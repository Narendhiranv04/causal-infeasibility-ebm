"""PoC-3 Stage 4.6C: dense feasibility-supervision diagnostic (approved exception; train + validation only).

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s46c_feasibility_supervision.py

DENSE-F (feasibility.FeasibilityModel: frozen REL-CF-PAIR-SHIFT architecture + scene bias on g_0, trained from
scratch with dense oracle F over the admissible domain) vs the frozen Stage-4.6B REL-CF-PAIR-SHIFT set-NLL model
(argmin E over the admissible domain). DENSE-F selects the minimum-cost predicted-feasible state. Order: frozen
reference integrity -> Stage-4.6A categories 288 / 27 / 45 -> base / fine dense-F labels on train + val -> stability
audit (STOP if > 1 % disagreement on either split) -> 20-scene gate (STOP unless state BA >= 0.98 and >= 19 / 20
repair hits) -> 3 seeds -> validation-only evaluation. No test rows; seed 61 not generated.

FROZEN INTERPRETATION (validation U-EXACT; across-seed means; pre-registered before any result)
  S1 BA_F >= 0.95; S2 Recall_F0 >= 0.90 and Recall_F1 >= 0.90; S3 derived valid-feasible >= 0.90;
  S4 hit >= 0.85, or Delta hit vs frozen REL-CF-PAIR-SHIFT >= 0.08 with paired 95 % CI lower bound > 0.
  Outcome S (DENSE FEASIBILITY SUPERVISION SUCCEEDS) iff S1-S4; else D (FEASIBILITY SIGNAL LEARNED; DECISION
  CALIBRATION / COMPOSITION REMAINS) iff S1 and S2; else R (DENSE FEASIBILITY SUPERVISION DOES NOT RESCUE THE
  MODEL) iff BA_F < 0.90 or valid-feasible < 0.85; else M (MIXED). D precedes R: with S1 and S2 the feasibility
  signal did generalize, so a weak repair decision is a composition failure, not a representation failure.
"""

import json
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s46a_error_audit as au  # noqa: E402

from poc2 import dataset as pd  # noqa: E402
from poc2 import oracle as orc  # noqa: E402
from poc3 import RunEnvironment  # noqa: E402
from poc3 import dataset as ds  # noqa: E402
from poc3 import feasibility as fz  # noqa: E402
from poc3 import metrics as mt  # noqa: E402
from poc3 import relational as rl  # noqa: E402
from poc3 import train as tr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s46c"
EXPECTED_COUNTS = {"U-EXACT": 288, "U-OVERLAP": 27, "P-TOP1": 45}
MAX_DISAGREEMENT, GATE_BA, LOW_SAMPLE = 0.01, 0.98, 30
REF_ARCH = rl.__doc__.strip() + "\n\n" + rl.PAIR_ARCHITECTURE
HYPER = {"optimizer": "AdamW", "lr": tr.LR, "weight_decay": tr.WEIGHT_DECAY, "batch": tr.BATCH_SCENES,
         "max_epochs": tr.MAX_EPOCHS, "patience": tr.PATIENCE, "seeds": list(tr.SEEDS),
         "loss": "scene-balanced dense BCE on stable admissible states", "selection": "validation dense-F BCE"}
POLICY = "states whose base (2 mm) and fine (1 mm) F disagree are excluded from dense BCE and state metrics only"
STATE_KEYS = ("bce", "recall_F0", "recall_F1", "balanced_accuracy", "false_feasible", "false_infeasible")


def guard(records, rows) -> list[int]:
    for j in rows:
        if records[j]["split"] not in ("train", "val"):
            raise RuntimeError("Stage 4.6C never evaluates the Stage-2 test split")
    return list(rows)


def _labels(args) -> dict:
    """Base table (canonical) + fine F on the admissible states; full tables kept for validation rows."""
    spec_json, P, adm, opt, keep_full = args
    spec = pd.spec_from_json(spec_json)
    lab = ds.oracle(spec)
    T, (scene, _, options) = lab["table"], ds.build(spec)
    Tf = orc.repair_table(scene, options, ds.FINE_STEP)
    states = np.array(sorted(lab["admissible_x"]), dtype=np.int64)
    out = {"states": states, "F": T.F[states].astype(np.int8), "F_fine": Tf.F[states].astype(np.int8),
           "consistent": lab["admissible_x"] == ds.unpack_states(adm, P) and lab["S_star_x"] == ds.unpack_states(opt, P),
           "shift": any(o.intervention.kind.value == "shift_target" for o in options), "K_star": int(T.K[min(lab["S_star_x"])])}
    return out | ({"M": T.M, "V": T.V, "Ffull": T.F, "K": T.K} if keep_full else {})


def dense_samples(records, arrays, rows, labels) -> list:
    out = []
    for j in guard(records, rows):
        r, P, L = records[j], int(arrays["P"][j]), labels[j]
        out.append(fz.make_dense_sample(*ds.build(pd.spec_from_json(r["spec"])), L["states"], L["F"], L["F"] == L["F_fine"],
                                        ds.unpack_states(arrays["optimal"][j], P), arrays["cost"][j][:P],
                                        {k: r[k] for k in ("scene_id", "family", "intent")}))
    return out


def _train(job) -> dict:
    seed, labels = job
    records, arrays = ds.load(ROOT / "s2")
    train_set = dense_samples(records, arrays, tr.split_rows(records, "train"), labels)
    val_set = dense_samples(records, arrays, tr.split_rows(records, "val"), labels)
    return tr.train(train_set, val_set, seed, pairwise=True, loss_fn=fz.dense_loss, name="dense_f_bce",
                    make_model=fz.FeasibilityModel)


def gate(samples, seed: int = 7, epochs: int = tr.GATE_EPOCHS) -> dict:
    run = tr.train(samples, samples, seed, max_epochs=epochs, patience=epochs, batch=tr.GATE_BATCH, pairwise=True,
                   loss_fn=fz.dense_loss, name="dense_f_bce", make_model=fz.FeasibilityModel)
    model = fz.load(run["state_dict"])
    with torch.no_grad():
        lg = [fz.feasibility_logits(model, s).numpy() for s in samples]
    ba = float(np.nanmean([fz.state_metrics(l_, s)["balanced_accuracy"] for l_, s in zip(lg, samples)]))
    hits = sum(fz.derive_repair(l_, s) in s.optimal for l_, s in zip(lg, samples))
    return {"state_balanced_accuracy": ba, "hits": int(hits), "n": len(samples),
            "passed": ba >= GATE_BA and hits * 20 >= tr.GATE_MIN_HITS * len(samples), "runtime_s": run["runtime_s"]}


def repair_row(x, L) -> dict:
    """Repair outcome against the canonical oracle table; x is None for NO_PREDICTED_FEASIBLE_STATE."""
    if x is fz.NO_PREDICTED_FEASIBLE_STATE:
        return {"hit": False, "valid_feasible": False, "minimal": False, "M_violation": False, "V_violation": False,
                "infeasible": False, "excess_cost": None, "n_pred": 0, "empty": False, "no_feasible": True}
    return mt.score_state(x, L["M"], L["V"], L["Ffull"], L["K"], L["optimal"]) | {"no_feasible": False}


def paired(a: dict, b: dict, idx, key: str) -> dict:
    d = np.mean([[float(b[s][i][key]) - float(a[s][i][key]) for i in idx] for s in tr.SEEDS], axis=0)
    return mt.paired_bootstrap(d) | {"low_sample_exploratory": len(idx) < LOW_SAMPLE}


def outcome(u: dict) -> dict:
    s = {"S1": u["ba"] >= 0.95, "S2": u["r0"] >= 0.90 and u["r1"] >= 0.90, "S3": u["vf"] >= 0.90,
         "S4": u["hit"] >= 0.85 or (u["dhit"] >= 0.08 and u["dhit_ci_lo"] > 0)}
    if all(s.values()):
        o = "S"
    elif s["S1"] and s["S2"]:
        o = "D"
    elif u["ba"] < 0.90 or u["vf"] < 0.85:
        o = "R"
    else:
        o = "M"
    return {"criteria": s, "inputs": u, "outcome": o, "failed": [k for k, v in s.items() if not v]}


NEXT = {"S": "Stage 4.6D (with approval): test incorporating feasibility supervision into the structured repair learner",
        "D": "diagnose decision calibration / min-cost composition on top of the learned feasibility signal",
        "R": "representation / inductive-bias diagnosis (no loss engineering)",
        "M": "mixed: review the failed criteria before choosing supervision or representation work"}


def strata_mean(per_scene: dict, idx, keys) -> dict:
    """Scene-balanced (nan-skipping) mean per seed, then mean / sample std across seeds."""
    vals = {k: [float(np.nanmean([per_scene[s][i][k] for i in idx])) if idx else math.nan for s in tr.SEEDS] for k in keys}
    return {"n_scenes": len(idx)} | {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for k, v in vals.items()}


def main() -> None:
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = RunEnvironment.capture().to_json()
    records, arrays = ds.load(ROOT / "s2")
    train_rows, val_rows = guard(records, tr.split_rows(records, "train")), guard(records, tr.split_rows(records, "val"))
    ref = {}
    for seed in tr.SEEDS:                                                        # 3. frozen reference integrity
        path = ROOT / "s46b" / "checkpoints" / f"rel_pair_shift_seed{seed}.pt"
        if not path.exists():
            sys.exit(f"STOP: missing frozen checkpoint {path}")
        ck = torch.load(path)
        if (ck["variant"], ck["pair"], ck["seed"], ck["dataset_digest"], ck["architecture"]) != \
                ("REL-CF-PAIR-SHIFT", "shift", seed, ds.STAGE2_DIGEST, REF_ARCH):
            sys.exit(f"STOP: frozen REL-CF-PAIR-SHIFT seed {seed} failed integrity")
        ref[seed] = rl.load(ck["state_dict"], "cf", "shift")
    spawn = mp.get_context("spawn")
    rep_val = au.guard(records, [j for j in val_rows if records[j]["intent"] == "repairable"])
    with ProcessPoolExecutor(12, mp_context=spawn) as pool:                     # 4. categories
        cat_scenes = list(pool.map(au._scene, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                                arrays["optimal"][j]) for j in rep_val], chunksize=4))
        rows = train_rows + val_rows                                             # 5. base / fine dense labels
        lab = list(pool.map(_labels, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                       arrays["optimal"][j], records[j]["split"] == "val") for j in rows], chunksize=8))
    category = {j: o["category"] for j, o in zip(rep_val, cat_scenes)}
    counts = {c: list(category.values()).count(c) for c in au.CATEGORIES}
    if counts != EXPECTED_COUNTS:
        sys.exit(f"STOP: Stage-4.6A categories not reproduced: {counts}")
    labels = dict(zip(rows, lab))
    if not all(L["consistent"] for L in lab):
        sys.exit("STOP: oracle tables disagree with the stored Stage-2 labels")
    stab = {}                                                                    # 6. stability audit
    for name, sel in [("train", train_rows), ("val", val_rows)] + [(f"family={f}", [j for j in rows if records[j]["family"] == f])
                                                                  for f in ds.FAMILIES]:
        n = sum(len(labels[j]["states"]) for j in sel)
        d = sum(int((labels[j]["F"] != labels[j]["F_fine"]).sum()) for j in sel)
        stab[name] = {"admissible_states": n, "F_disagreements": d, "fraction": d / n}
    base = {"stage": "4.6C", "git_sha": env["git_sha"], "git_dirty": env["git_dirty"], "dataset_digest": ds.STAGE2_DIGEST,
            "environment": env, "stage2_test_split_evaluated": False, "future_audit_seed_61_generated": False,
            "rows_used": {"train": len(train_rows), "val": len(val_rows), "test": 0}, "category_counts": counts,
            "dense_label_stability": stab | {"policy": POLICY, "max_disagreement": MAX_DISAGREEMENT},
            "note": "l(x) is a feasibility polynomial, not the final repair-selection energy; K never enters it"}
    if stab["train"]["fraction"] > MAX_DISAGREEMENT or stab["val"]["fraction"] > MAX_DISAGREEMENT:
        (OUT / "metrics.json").write_text(json.dumps(base | {"stopped": "dense labels unstable"}, indent=1))
        sys.exit("STOP: dense F labels unstable (> 1 % base / fine disagreement)")
    g = gate(dense_samples(records, arrays, tr.gate_rows(records), labels))      # 9. implementation gate
    print(json.dumps({"stability": {k: stab[k] for k in ("train", "val")}, "gate": g}, default=float), flush=True)
    base["implementation_gate"] = g
    if not g["passed"]:
        (OUT / "metrics.json").write_text(json.dumps(base | {"stopped": "implementation gate failed"}, indent=1))
        sys.exit("STOP: 20-scene dense-F gate failed")
    compact = {j: {k: labels[j][k] for k in ("states", "F", "F_fine")} for j in rows}
    with ProcessPoolExecutor(3, mp_context=spawn) as pool:                      # 11-12. train, select on val BCE
        runs = {r["seed"]: r for r in pool.map(_train, [(s, compact) for s in tr.SEEDS])}
    for seed, r in runs.items():
        torch.save({"state_dict": r["state_dict"], "seed": seed, "dataset_digest": ds.STAGE2_DIGEST,
                    "architecture": fz.__doc__.strip() + "\n\n" + REF_ARCH, "dense_target": "F", "threshold_logit": 0.0,
                    "best_val_dense_f_bce": r["best_val_dense_f_bce"], "training_hyperparameters": HYPER,
                    "stability_exclusion_policy": POLICY}, OUT / "checkpoints" / f"dense_f_seed{seed}.pt")
    val = dense_samples(records, arrays, val_rows, labels)                      # 13. validation only
    for s, j in zip(val, val_rows):
        labels[j]["optimal"] = s.optimal
    state, dense, refr = {}, {}, {}
    for seed in tr.SEEDS:
        model = fz.load(runs[seed]["state_dict"])
        state[seed], dense[seed], refr[seed] = [], [], []
        for s, j in zip(val, val_rows):
            with torch.no_grad():
                lg = fz.feasibility_logits(model, s).numpy()
            card = s.bits.numpy().sum(axis=1)
            state[seed].append(fz.state_metrics(lg, s) | {f"{k}_le1": v for k, v in fz.state_metrics(lg, s, card <= 1).items()}
                               | {f"{k}_ge2": v for k, v in fz.state_metrics(lg, s, card >= 2).items()})
            dense[seed].append(repair_row(fz.derive_repair(lg, s), labels[j]) | {"family": s.meta["family"],
                                                                                 "intent": s.meta["intent"]})
            ss = tr.StructuredSample(s.inputs, torch.as_tensor(s.cost), s.states, s.bits, torch.as_tensor(
                np.isin(s.states, sorted(s.optimal))), s.optimal, s.meta)
            refr[seed].append(repair_row(tr.exact_inference(tr.energies(ref[seed], ss), ss)[0], labels[j])
                              | {"family": s.meta["family"], "intent": s.meta["intent"]})
    N = len(val_rows)
    pick = lambda f: [i for i in range(N) if f(i, val_rows[i])]  # noqa: E731
    rep = pick(lambda i, j: records[j]["intent"] == "repairable")
    pops = {"overall": list(range(N)), "repairable": rep, "feasible_negative": pick(lambda i, j: records[j]["intent"] == "negative")}
    pops |= {c: [i for i in rep if category[val_rows[i]] == c] for c in au.CATEGORIES}
    fam = {f"family={f}": [i for i in rep if records[val_rows[i]]["family"] == f] for f in ds.FAMILIES}
    shift = {f"SHIFT={v}": [i for i in rep if labels[val_rows[i]]["shift"] == v] for v in (True, False)}
    size = {f"K*={k}": [i for i in rep if labels[val_rows[i]]["K_star"] == k] for k in (1, 2, 3, 4)}
    sk = STATE_KEYS + tuple(f"{k}_{c}" for c in ("le1", "ge2") for k in STATE_KEYS)
    sf = {k: strata_mean(state, v, sk) for k, v in (pops | fam | shift | size).items()}
    summ = lambda rows_, idx: (mt.across_seeds([mt.summarize([rows_[s][i] for i in idx], ("no_feasible",))  # noqa: E731
                                                for s in tr.SEEDS]) if idx else {"n": 0})
    sel = {"all_repairable": rep, "feasible_negative": pops["feasible_negative"]} | {c: pops[c] for c in au.CATEGORIES}
    sel |= fam | size
    repair = {k: {"DENSE-F": summ(dense, v), "REL-CF-PAIR-SHIFT": summ(refr, v)} for k, v in sel.items()}
    boots = {k: {m: paired(refr, dense, v, m) for m in ("hit", "valid_feasible")} for k, v in
             ({"ALL repairable": rep} | {c: pops[c] for c in au.CATEGORIES} | {k: size[k] for k in ("K*=1", "K*=2")} | fam).items()}
    U = pops["U-EXACT"]
    u = {"ba": sf["U-EXACT"]["balanced_accuracy"]["mean"], "r0": sf["U-EXACT"]["recall_F0"]["mean"],
         "r1": sf["U-EXACT"]["recall_F1"]["mean"], "vf": repair["U-EXACT"]["DENSE-F"]["valid_feasible"]["mean"],
         "hit": repair["U-EXACT"]["DENSE-F"]["hit"]["mean"], "dhit": boots["U-EXACT"]["hit"]["mean_diff"],
         "dhit_ci_lo": boots["U-EXACT"]["hit"]["ci95"][0], "n_U_EXACT": len(U)}
    verdict = outcome(u)
    out = base | {
        "frozen_reference": {"model": "REL-CF-PAIR-SHIFT set-NLL (Stage 4.6B), argmin E over the admissible domain",
                             "checkpoints": "PoC-3/out/s46b/checkpoints/rel_pair_shift_seed{7,17,27}.pt"},
        "dense_f_per_seed": {s: {k: runs[s][k] for k in ("best_epoch", "epochs_run", "best_val_dense_f_bce", "runtime_s")}
                             for s in tr.SEEDS},
        "histories": {s: runs[s]["history"] for s in tr.SEEDS},
        "state_feasibility": {k: sf[k] for k in ("overall", "repairable", "feasible_negative", *au.CATEGORIES)}
        | {"by_family": {k: sf[k] for k in fam}, "by_shift": {k: sf[k] for k in shift}, "by_repair_size": {k: sf[k] for k in size},
           "singleton_vs_multi": {p: {c: {k: sf[p][f"{k}_{c}"] for k in STATE_KEYS} for c in ("le1", "ge2")}
                                  for p in ("overall", "U-EXACT")}},
        "repair_selection": repair, "paired_bootstraps": boots, "interpretation_criteria": verdict,
        "outcome": verdict["outcome"], "recommended_next_stage": NEXT[verdict["outcome"]],
    }
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"outcome": verdict, "next": out["recommended_next_stage"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
