"""PoC-3 Stage 4.6C.3: full-training fit / shared-mapping diagnostic (TRAIN ONLY; approved diagnostic exception).

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s46c3_full_train_fit.py

Every row used is a Stage-2 TRAIN row (guarded; exactly 2240 = 1680 repairable + 560 negatives). No validation sample
is built for inference, no validation / test metric is computed, seed 61 is not generated, no checkpoint is written.
Frozen FeasibilityModel, dense scene-balanced BCE, threshold 0, admissible domain M = V = 1.
  Structure: per-scene SHIFT-masked quadratic LP strict separability (C.1 design / LP helpers, HiGHS) on stable dense
    F; unrestricted-Q LP only on failed scenes. R_sep = separable / 2240; STOP (Outcome SF) if R_sep < 0.99.
  Run A: seed 7, AdamW lr 1e-3, wd 1e-5, batch 32, 715 epochs = 50 050 steps, no early stopping; train-only metrics
    every 25 epochs and at the final epoch. FULL_TRAIN_FIT iff some record has BA >= 0.98 and both recalls >= 0.97.
  Run B (only if A fails): identical except lr 3e-3. No other setting exists.
  Plateau (per run without fit): tail = records from step >= 50 050 - 10 000 (epochs 575 .. 715); PLATEAU iff
    Delta BA_tail < 0.005 and (BCE_start - BCE_end) / BCE_start < 0.05, else STILL IMPROVING.
  Outcomes: SF if R_sep < 0.99; O if A fits; H if B fits; J if both runs plateau; U otherwise.
"""

import copy
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s46c1_gate_diagnosis as gd  # noqa: E402
import s46c_feasibility_supervision as sc  # noqa: E402

from poc3 import RunEnvironment  # noqa: E402
from poc3 import dataset as ds  # noqa: E402
from poc3 import feasibility as fz  # noqa: E402
from poc3 import train as tr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s46c3"
RUN_A = {"seed": 7, "lr": 1e-3, "weight_decay": 1e-5, "batch": 32, "epochs": 715}
RUN_B = RUN_A | {"lr": 3e-3}
RECORD_EVERY, SEP_MIN, TAIL_STEPS, FIT_BA, FIT_RECALL = 25, 0.99, 10_000, 0.98, 0.97
N_TRAIN, N_REP, N_NEG = 2240, 1680, 560
LABEL = {"SF": "FULL-DATA STRUCTURAL FORM BOTTLENECK", "O": "FULL TRAINING DISTRIBUTION IS FITTABLE WITH THE CURRENT MODEL",
         "H": "FULL-DATA FIT IS OPTIMIZER-SENSITIVE", "J": "SHARED SCENE-TO-COEFFICIENT FIT BOTTLENECK",
         "U": "FULL-DATA FIT REMAINS OPTIMIZATION-UNRESOLVED"}
NEXT = {"SF": "structural-form diagnosis (mask vs order) before any neural work",
        "O": "re-run dense-F generalization with a pre-registered longer training protocol, representation unchanged",
        "H": "test the train-only-selected 3e-3 protocol on validation, no architecture change",
        "J": "representation diagnostic (width / pooling / relational features / optimization landscape)",
        "U": "longer or better-resolved optimization diagnosis; no representation redesign yet"}


def guard_train(records, rows) -> list[int]:
    for j in rows:
        if records[j]["split"] != "train":
            raise RuntimeError("Stage 4.6C.3 is train-only: validation / test rows are forbidden")
    return list(rows)


def train_rows_checked(records) -> list[int]:
    rows = guard_train(records, [j for j, r in enumerate(records) if r["split"] == "train"])
    rep = sum(records[j]["intent"] == "repairable" for j in rows)
    if (len(rows), rep, len(rows) - rep) != (N_TRAIN, N_REP, N_NEG):
        raise RuntimeError("unexpected Stage-2 train composition")
    return rows


def _separable(args) -> bool:
    bits, cand, F, stable, masked = args
    return gd.lp_separable(gd.design(bits[stable], gd.pair_list(cand, masked)), F[stable])


def card_masks(s) -> dict:
    c = s.bits.numpy().sum(axis=1)
    return {"x0": c == 0, "x1": c == 1, "x2plus": c >= 2}


def scene_eval(lg: np.ndarray, s) -> dict:
    st_ = fz.state_metrics(lg, s)
    x = fz.derive_repair(lg, s)
    vf = x is not None and s.F.numpy()[int(np.flatnonzero(s.states == x)[0])] == 0
    return st_ | {f"{k}_{c}": v for c, m in card_masks(s).items() for k, v in fz.state_metrics(lg, s, m).items()} | {
        "hit": x in s.optimal, "valid_feasible": bool(vf), "empty_correct": x == 0, "x_hat": x}


def evaluate(model, samples) -> list[dict]:
    model.eval()
    with torch.no_grad():
        return [scene_eval(fz.feasibility_logits(model, s).numpy(), s) for s in samples]


def aggregate(per: list[dict], samples, idx=None) -> dict:
    idx = range(len(per)) if idx is None else idx
    rows = [(per[i], samples[i]) for i in idx]
    if not rows:
        return {"n": 0}
    mean = lambda k, sel=None: float(np.nanmean([p[k] for p, s in rows if sel is None or sel(s)]))  # noqa: E731
    rep, neg = (lambda s: s.meta["intent"] == "repairable"), (lambda s: s.meta["intent"] == "negative")
    out = {"n": len(rows)} | {k: mean(k) for k in ("bce", "balanced_accuracy", "recall_F0", "recall_F1", "false_feasible",
                                                   "false_infeasible")}
    if any(rep(s) for _, s in rows):
        out |= {"repairable_hit": mean("hit", rep), "repairable_valid_feasible": mean("valid_feasible", rep)}
    if any(neg(s) for _, s in rows):
        out["negative_correct_empty"] = mean("empty_correct", neg)
    return out


def run(samples, cfg: dict, record_every: int = RECORD_EVERY) -> dict:
    """tr.train's exact seeding / init / shuffling / AdamW step; no early stopping; train-only records."""
    tr.set_determinism(cfg["seed"])
    model = fz.FeasibilityModel()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    order_rng = np.random.default_rng(cfg["seed"])
    steps_per_epoch = math.ceil(len(samples) / cfg["batch"])
    losses, records, best = [], [], (-1.0, None, None)
    t0 = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        model.train()
        order, total = order_rng.permutation(len(samples)), 0.0
        for k in range(0, len(order), cfg["batch"]):
            chunk = [samples[i] for i in order[k:k + cfg["batch"]]]
            opt.zero_grad()
            loss = fz.dense_loss(model, chunk)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(chunk)
        losses.append(total / len(samples))
        if (epoch + 1) % record_every == 0 or epoch + 1 == cfg["epochs"]:
            per = evaluate(model, samples)
            agg = aggregate(per, samples) | {"epoch": epoch + 1, "step": (epoch + 1) * steps_per_epoch,
                                             "running_train_bce": losses[-1], "elapsed_s": time.perf_counter() - t0}
            agg["per_family_BA"] = {f: aggregate(per, samples, [i for i, s in enumerate(samples) if s.meta["family"] == f])
                                    .get("balanced_accuracy") for f in ds.FAMILIES}
            records.append(agg)
            print(json.dumps({k: agg[k] for k in ("epoch", "balanced_accuracy", "recall_F0", "recall_F1", "bce")}), flush=True)
            if agg["balanced_accuracy"] > best[0]:
                best = (agg["balanced_accuracy"], copy.deepcopy(model.state_dict()), epoch + 1)
    fit = [r for r in records if r["balanced_accuracy"] >= FIT_BA and min(r["recall_F0"], r["recall_F1"]) >= FIT_RECALL]
    return {"config": cfg, "steps_per_epoch": steps_per_epoch, "total_steps": cfg["epochs"] * steps_per_epoch,
            "train_bce_per_epoch": losses, "records": records, "best_state": best[1], "best_epoch": best[2],
            "full_train_fit": bool(fit), "first_fit": {k: fit[0][k] for k in ("epoch", "step")} if fit else None,
            "runtime_s": time.perf_counter() - t0}


def plateau(records, total_steps: int) -> dict:
    tail = [r for r in records if r["step"] >= total_steps - TAIL_STEPS]
    a, b = tail[0], tail[-1]
    d_ba, r_bce = b["balanced_accuracy"] - a["balanced_accuracy"], (a["bce"] - b["bce"]) / a["bce"]
    flat = d_ba < 0.005 and r_bce < 0.05
    return {"tail_epochs": [a["epoch"], b["epoch"]], "delta_BA_tail": d_ba, "R_BCE": r_bce, "plateau": flat,
            "label": "PLATEAU" if flat else "STILL IMPROVING"}


def anatomy(per, samples, shift_of) -> dict:
    pick = lambda f: [i for i, s in enumerate(samples) if f(s)]  # noqa: E731
    pops = {"all": list(range(len(samples))), "repairable": pick(lambda s: s.meta["intent"] == "repairable"),
            "feasible_negative": pick(lambda s: s.meta["intent"] == "negative")}
    pops |= {f"family={f}": pick(lambda s, f=f: s.meta["family"] == f) for f in ds.FAMILIES}
    pops |= {f"SHIFT={v}": pick(lambda s, v=v: shift_of[s.meta["scene_id"]] == v) for v in (True, False)}
    kstar = {i: min(int(x).bit_count() for x in s.optimal) for i, s in enumerate(samples) if s.meta["intent"] == "repairable"}
    pops |= {f"K*={k}": [i for i, v in kstar.items() if (v >= 4 if k == "4+" else v == k)] for k in (1, 2, 3, "4+")}
    card = {c: {k: float(np.nanmean([per[i][f"{k}_{c}"] for i in pops["repairable"]])) for k in
                ("balanced_accuracy", "recall_F0", "recall_F1")} for c in ("x0", "x1", "x2plus")}
    order = np.argsort([per[i]["balanced_accuracy"] for i in range(len(per))], kind="stable")[:50]
    worst = [{"scene_id": samples[i].meta["scene_id"], "family": samples[i].meta["family"], "P": samples[i].bits.shape[1],
              "n_admissible": len(samples[i].states), "fraction_feasible": float(np.mean(samples[i].F.numpy() == 0)),
              "K_star": kstar.get(i, 0), "BA": per[i]["balanced_accuracy"], "recall_F0": per[i]["recall_F0"],
              "recall_F1": per[i]["recall_F1"], "derived_repair_hit": per[i]["hit"]} for i in order]
    return {k: aggregate(per, samples, v) for k, v in pops.items()} | {"cardinality_repairable": card, "worst_50": worst}


def coefficients(model, samples) -> dict:
    b, q, Q = [], [], []
    model.eval()
    with torch.no_grad():
        for s in samples:
            bb, qq, QQ = model.terms(**s.inputs)
            b.append(abs(float(bb)))
            q += qq.abs().tolist()
            iu = torch.triu_indices(len(qq), len(qq), 1)
            Q += [v for v in QQ[iu[0], iu[1]].abs().tolist() if v != 0]
    stat = lambda v: {"n": len(v), "median_abs": float(np.median(v)), "p95_abs": float(np.percentile(v, 95)),  # noqa: E731
                      "max_abs": float(np.max(v))}
    return {"b": stat(b), "q": stat(q), "Q_nonzero_shift_pairs": stat(Q)}


def outcome(r_sep: float, A: dict, B: dict | None, plateaus: dict) -> str:
    if r_sep < SEP_MIN:
        return "SF"
    if A["full_train_fit"]:
        return "O"
    if B is not None and B["full_train_fit"]:
        return "H"
    return "J" if all(p["plateau"] for p in plateaus.values()) else "U"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = RunEnvironment.capture().to_json()
    records, arrays = ds.load(ROOT / "s2")
    rows = train_rows_checked(records)
    with ProcessPoolExecutor(12, mp_context=mp.get_context("spawn")) as pool:
        lab = list(pool.map(sc._labels, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                          arrays["optimal"][j], False) for j in rows], chunksize=8))
    labels = dict(zip(rows, lab))
    if not all(L["consistent"] for L in lab):
        sys.exit("STOP: oracle tables disagree with the stored Stage-2 labels")
    n_states, n_dis = sum(len(L["states"]) for L in lab), sum(int((L["F"] != L["F_fine"]).sum()) for L in lab)
    compact = {j: {k: labels[j][k] for k in ("states", "F", "F_fine")} for j in rows}
    samples = sc.dense_samples(records, arrays, guard_train(records, rows), compact)
    shift_of = {records[j]["scene_id"]: labels[j]["shift"] for j in rows}
    base = {"stage": "4.6C.3", "git_sha": env["git_sha"], "git_dirty": env["git_dirty"], "dataset_digest": ds.STAGE2_DIGEST,
            "environment": env, "validation_evaluated": False, "stage2_test_split_evaluated": False,
            "future_audit_seed_61_generated": False, "rows_used": {"train": len(rows), "val": 0, "test": 0},
            "rules": __doc__.strip(), "train_label_stability": {"admissible_states": n_states, "F_disagreements": n_dis}}
    if n_dis / n_states > sc.MAX_DISAGREEMENT:
        sys.exit("STOP: train dense labels unstable")
    args = [(s.bits.numpy(), s.inputs["candidates"].numpy(), s.F.numpy(), s.stable.numpy(), True) for s in samples]
    with ProcessPoolExecutor(12, mp_context=mp.get_context("spawn")) as pool:
        sep = list(pool.map(_separable, args, chunksize=16))
        failed = [i for i, ok in enumerate(sep) if not ok]
        unres = list(pool.map(_separable, [args[i][:4] + (False,) for i in failed]))
    r_sep = sum(sep) / len(sep)

    def counts(key):
        groups = {}
        for i, s in enumerate(samples):
            g = groups.setdefault(str(key(i, s)), [0, 0])
            g[0] += sep[i]
            g[1] += 1
        return {k: {"separable": a, "n": n} for k, (a, n) in sorted(groups.items())}
    base["structural_realizability_full_train"] = {
        "total": len(sep), "separable": sum(sep), "non_separable": len(failed), "R_sep": r_sep,
        "per_family": counts(lambda i, s: s.meta["family"]), "by_P": counts(lambda i, s: s.bits.shape[1]),
        "by_repair_size_or_negative": counts(lambda i, s: "negative" if s.meta["intent"] == "negative"
                                             else min(int(x).bit_count() for x in s.optimal))}
    base["unrestricted_control_on_failures"] = {
        "n_failed": len(failed), "rescued": sum(unres),
        "per_scene": [{"scene_id": samples[i].meta["scene_id"], "unrestricted_separable": u} for i, u in zip(failed, unres)],
        "case": None if not failed else ("A: SHIFT MASK IS NOT UNIVERSALLY SUFFICIENT FOR DENSE F" if all(unres) else
                                         "B: QUADRATIC SIGN FORM IS NOT UNIVERSALLY SUFFICIENT FOR DENSE F")}
    print(json.dumps({"R_sep": r_sep, "failed": len(failed), "rescued": sum(unres)}), flush=True)
    if r_sep < SEP_MIN:
        o = "SF"
        (OUT / "metrics.json").write_text(json.dumps(base | {"run_A": None, "run_B": None, "outcome": o, "outcome_label":
                                                             LABEL[o], "recommended_next_stage": NEXT[o]}, indent=1, default=float))
        sys.exit("STOP: full-data structural form bottleneck")
    A = run(samples, RUN_A)
    B = run(samples, RUN_B) if not A["full_train_fit"] else None
    plateaus = {name: plateau(r["records"], r["total_steps"]) for name, r in (("A", A), ("B", B))
                if r is not None and not r["full_train_fit"]}
    o = outcome(r_sep, A, B, plateaus)
    best_runs = {}
    for name, r in (("A", A), ("B", B)):
        if r is None:
            continue
        model = fz.FeasibilityModel()
        model.load_state_dict(r["best_state"])
        per = evaluate(model, samples)
        best_runs[name] = {"best_epoch": r["best_epoch"], "metrics": aggregate(per, samples),
                           "anatomy": anatomy(per, samples, shift_of), "coefficients": coefficients(model, samples)}
    strip = lambda r: None if r is None else {k: v for k, v in r.items() if k not in ("best_state", "train_bce_per_epoch")}  # noqa: E731
    out = base | {"run_A": strip(A), "run_B": strip(B),
                  "best_train_state": {k: {"epoch": v["best_epoch"], **v["metrics"]} for k, v in best_runs.items()},
                  "train_error_anatomy": {k: v["anatomy"] for k, v in best_runs.items()},
                  "coefficient_magnitudes": {k: v["coefficients"] for k, v in best_runs.items()},
                  "plateau_diagnostics": plateaus, "outcome": o, "outcome_label": LABEL[o], "recommended_next_stage": NEXT[o],
                  "historical_context": "Stage-4.6C.2 selected-checkpoint train BA .925-.938 (recorded scalars only)"}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"outcome": o, "label": LABEL[o], "A_fit": A["full_train_fit"],
                      "B_fit": None if B is None else B["full_train_fit"], "plateaus": plateaus}, indent=1, default=float))


if __name__ == "__main__":
    main()
