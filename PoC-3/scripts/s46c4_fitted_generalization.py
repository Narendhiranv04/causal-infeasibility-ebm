"""PoC-3 Stage 4.6C.4: fitted-model generalization diagnostic (approved diagnostic exception).

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s46c4_fitted_generalization.py

Frozen FeasibilityModel / dense BCE / threshold 0 / admissible domain. Train seeds 7, 17, 27 on the 2240 TRAIN rows
only: AdamW lr 1e-3, wd 1e-5, batch 32, exactly 700 epochs = 49 000 steps, NO early stopping; train-only records every
25 epochs. Per seed two states are frozen in memory with zero validation information: FIT = first 25-epoch record
with train BA >= 0.98 and both train recalls >= 0.97 (primary), FINAL = epoch 700 (secondary). Validation (480 rows)
is evaluated only after all states are frozen; the frozen C.2 dense-F and Stage-4.6B REL-CF-PAIR-SHIFT checkpoints are
the references. No test rows, no seed 61.
PRE-REGISTERED OUTCOMES (FIT, validation U-EXACT; across fitted seeds):
  INSTABILITY (TRAIN-FIT INSTABILITY) if any seed never fits (no clean conclusion; fitted seeds still reported);
  else T (FITTED DENSE-F GENERALIZES) iff the original Outcome-S criteria S1-S4 hold (vs Stage-4.6B for S4);
  else G (TRAINING FITS; VALIDATION DOES NOT - GENERALIZATION BOTTLENECK) iff BA_F < 0.90 or valid-feasible < 0.85;
  else M (LONGER TRAINING HELPS, BUT DOES NOT SOLVE GENERALIZATION) iff FIT materially improves C.2: paired
  bootstrap FIT - C.2 U-EXACT hit has CI lower bound > 0, or U-EXACT state BA rises by >= 0.02;
  else N (NO MATERIAL IMPROVEMENT; BELOW SUCCESS BAR).
  OVERFITTING_AFTER_FIT iff FINAL - FIT repairable hit <= -0.03 with paired CI upper bound < 0, or validation BA
  (all 480) falls by >= 0.02.
"""

import copy
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s46a_error_audit as au  # noqa: E402
import s46c2_dense_f_full as c2  # noqa: E402
import s46c3_full_train_fit as c3  # noqa: E402
import s46c_feasibility_supervision as sc  # noqa: E402

from poc3 import RunEnvironment  # noqa: E402
from poc3 import dataset as ds  # noqa: E402
from poc3 import feasibility as fz  # noqa: E402
from poc3 import metrics as mt  # noqa: E402
from poc3 import train as tr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s46c4"
SEEDS = (7, 17, 27)
CFG = {"lr": 1e-3, "weight_decay": 1e-5, "batch": 32, "epochs": 700}
RECORD_EVERY, FIT_BA, FIT_RECALL = 25, 0.98, 0.97
LABEL = {"INSTABILITY": "TRAIN-FIT INSTABILITY", "T": "FITTED DENSE-F GENERALIZES",
         "G": "TRAINING FITS; VALIDATION DOES NOT - GENERALIZATION BOTTLENECK",
         "M": "LONGER TRAINING HELPS, BUT DOES NOT SOLVE GENERALIZATION", "N": "NO MATERIAL IMPROVEMENT; BELOW SUCCESS BAR"}
SELECTION = "first 25-epoch TRAIN record with BA >= 0.98 and recall_F0, recall_F1 >= 0.97; no validation information"


def is_fit(rec: dict) -> bool:
    return rec["balanced_accuracy"] >= FIT_BA and min(rec["recall_F0"], rec["recall_F1"]) >= FIT_RECALL


def run_fit_final(samples, seed: int, cfg: dict = CFG, record_every: int = RECORD_EVERY) -> dict:
    """tr.train's exact seeding / init / shuffling / AdamW step on TRAIN samples only; freezes FIT and FINAL."""
    tr.set_determinism(seed)
    model = fz.FeasibilityModel()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    order_rng = np.random.default_rng(seed)
    steps = -(-len(samples) // cfg["batch"])
    losses, records, fit = [], [], None
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
            rec = c3.aggregate(c3.evaluate(model, samples), samples) | {"epoch": epoch + 1, "step": (epoch + 1) * steps}
            records.append(rec)
            if fit is None and is_fit(rec):
                fit = {"epoch": epoch + 1, "step": (epoch + 1) * steps, "train": rec, "state": copy.deepcopy(model.state_dict())}
    return {"seed": seed, "records": records, "train_bce_per_epoch": losses, "fit": fit,
            "final": {"epoch": cfg["epochs"], "step": cfg["epochs"] * steps, "train": records[-1],
                      "state": copy.deepcopy(model.state_dict())}, "runtime_s": time.perf_counter() - t0}


def _train_seed(args) -> dict:
    seed, compact = args
    records, arrays = ds.load(ROOT / "s2")
    return run_fit_final(sc.dense_samples(records, arrays, c3.train_rows_checked(records), compact), seed)


def load_c2() -> dict:
    out = {}
    for seed in SEEDS:
        ck = torch.load(ROOT / "s46c2" / "checkpoints" / f"dense_f_seed{seed}.pt")
        if (ck["seed"], ck["dataset_digest"], ck["dense_target"], ck["threshold_logit"], ck["gate_budget"]) != \
                (seed, ds.STAGE2_DIGEST, "F", 0.0, 1000) or ck["architecture"] != fz.__doc__.strip() + "\n\n" + sc.REF_ARCH:
            sys.exit(f"STOP: C.2 dense-F checkpoint seed {seed} failed integrity")
        out[seed] = fz.load(ck["state_dict"])
    return out


def evaluate_dense(model, val, labels, rows) -> tuple[list, list]:
    state, rep = [], []
    model.eval()
    with torch.no_grad():
        for s, j in zip(val, rows):
            lg = fz.feasibility_logits(model, s).numpy()
            state.append(fz.state_metrics(lg, s) | {f"{k}_{c}": v for c, m in c2.card_masks(s).items()
                                                    for k, v in fz.state_metrics(lg, s, m).items()})
            x = fz.derive_repair(lg, s)
            rep.append(sc.repair_row(x, labels[j]) | {"x_hat": x, "family": s.meta["family"], "intent": s.meta["intent"]})
    return state, rep


def evaluate_ref(model, val, labels, rows) -> list:
    out = []
    for s, j in zip(val, rows):
        ss = tr.StructuredSample(s.inputs, torch.as_tensor(s.cost), s.states, s.bits,
                                 torch.as_tensor(np.isin(s.states, sorted(s.optimal))), s.optimal, s.meta)
        x = tr.exact_inference(tr.energies(model, ss), ss)[0]
        out.append(sc.repair_row(x, labels[j]) | {"x_hat": x, "family": s.meta["family"], "intent": s.meta["intent"]})
    return out


def paired(a: dict, b: dict, idx, key: str) -> dict:
    """b - a, each model averaged over its own available seeds per scene, then bootstrapped over scenes."""
    mean = lambda rows: np.mean([[float(rows[s][i][key]) for i in idx] for s in rows], axis=0)  # noqa: E731
    return mt.paired_bootstrap(mean(b) - mean(a)) | {"low_sample_exploratory": len(idx) < sc.LOW_SAMPLE}


def stat(rows: dict, idx, keys) -> dict:
    vals = {k: [float(np.nanmean([rows[s][i][k] for i in idx])) for s in rows] for k in keys}
    return {"n_scenes": len(idx)} | {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                                     for k, v in vals.items()}


def summ(rows: dict, idx) -> dict:
    return mt.across_seeds([mt.summarize([rows[s][i] for i in idx], ("no_feasible",)) for s in rows]) if idx else {"n": 0}


def outcome(all_fit: bool, crit: dict, u: dict, improves_c2: bool) -> str:
    if not all_fit:
        return "INSTABILITY"
    if crit["outcome"] == "S":
        return "T"
    if u["ba"] < 0.90 or u["vf"] < 0.85:
        return "G"
    return "M" if improves_c2 else "N"


def main() -> None:
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = RunEnvironment.capture().to_json()
    records, arrays = ds.load(ROOT / "s2")
    train_rows = c3.train_rows_checked(records)
    val_rows = sc.guard(records, tr.split_rows(records, "val"))
    ref46b, refc2 = c2.check_reference(), load_c2()
    rows = train_rows + val_rows
    rep_val = au.guard(records, [j for j in val_rows if records[j]["intent"] == "repairable"])
    with ProcessPoolExecutor(12, mp_context=mp.get_context("spawn")) as pool:
        lab = list(pool.map(sc._labels, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                          arrays["optimal"][j], records[j]["split"] == "val") for j in rows], chunksize=8))
        cat_scenes = list(pool.map(au._scene, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                                arrays["optimal"][j]) for j in rep_val], chunksize=4))
    labels = dict(zip(rows, lab))
    stab = {n: {"admissible_states": sum(len(labels[j]["states"]) for j in sel),
                "F_disagreements": sum(int((labels[j]["F"] != labels[j]["F_fine"]).sum()) for j in sel)}
            for n, sel in (("train", train_rows), ("val", val_rows))}
    category = {j: o["category"] for j, o in zip(rep_val, cat_scenes)}
    counts = {c: list(category.values()).count(c) for c in au.CATEGORIES}
    if not all(L["consistent"] for L in lab) or counts != sc.EXPECTED_COUNTS or \
            any(v["F_disagreements"] / v["admissible_states"] > sc.MAX_DISAGREEMENT for v in stab.values()):
        sys.exit(f"STOP: labels / stability / categories not reproduced: {counts} {stab}")
    compact = {j: {k: labels[j][k] for k in ("states", "F", "F_fine")} for j in train_rows}  # train labels only
    with ProcessPoolExecutor(3, mp_context=mp.get_context("spawn")) as pool:   # training sees no validation row
        runs = {r["seed"]: r for r in pool.map(_train_seed, [(s, compact) for s in SEEDS])}
    fitted = [s for s in SEEDS if runs[s]["fit"] is not None]
    for s in SEEDS:                                                           # freeze before any validation
        for kind in ("fit", "final"):
            st_ = runs[s][kind]
            if st_ is None:
                continue
            torch.save({"state_dict": st_["state"], "seed": s, "kind": kind.upper(), "epoch": st_["epoch"], "step": st_["step"],
                        "selection": SELECTION if kind == "fit" else "epoch 700", "validation_used_for_selection": False,
                        "train_metrics_at_selection": {k: v for k, v in st_["train"].items() if k != "per_family_BA"},
                        "dataset_digest": ds.STAGE2_DIGEST, "architecture": fz.__doc__.strip() + "\n\n" + sc.REF_ARCH,
                        "training_hyperparameters": CFG | {"early_stopping": None}},
                       OUT / "checkpoints" / f"{kind}_seed{s}.pt")
    full = {j: {k: labels[j][k] for k in ("states", "F", "F_fine")} for j in val_rows}
    val = sc.dense_samples(records, arrays, val_rows, full)
    for s_, j in zip(val, val_rows):
        labels[j]["optimal"] = s_.optimal
    st, rp = {"FIT": {}, "FINAL": {}, "C2": {}}, {"FIT": {}, "FINAL": {}, "C2": {}, "REL-CF-PAIR-SHIFT": {}}
    for s in SEEDS:
        for kind in ("FIT", "FINAL"):
            if runs[s][kind.lower()] is not None:
                st[kind][s], rp[kind][s] = evaluate_dense(fz.load(runs[s][kind.lower()]["state"]), val, labels, val_rows)
        st["C2"][s], rp["C2"][s] = evaluate_dense(refc2[s], val, labels, val_rows)
        rp["REL-CF-PAIR-SHIFT"][s] = evaluate_ref(ref46b[s], val, labels, val_rows)
    N = len(val_rows)
    rep = [i for i in range(N) if records[val_rows[i]]["intent"] == "repairable"]
    neg = [i for i in range(N) if records[val_rows[i]]["intent"] == "negative"]
    cats = {c: [i for i in rep if category[val_rows[i]] == c] for c in au.CATEGORIES}
    shift = {f"SHIFT={v}": [i for i in rep if labels[val_rows[i]]["shift"] == v] for v in (True, False)}
    size = {f"K*={k}": [i for i in rep if labels[val_rows[i]]["K_star"] == k] for k in (1, 2, 3, 4)}
    fam = {f"family={f}": [i for i in rep if records[val_rows[i]]["family"] == f] for f in ds.FAMILIES}
    keys = sc.STATE_KEYS + tuple(f"{k}_{c}" for c in c2.CARD for k in sc.STATE_KEYS)
    pops = {"all_validation": list(range(N)), "repairable": rep, "feasible_negative": neg} | cats | shift | fam
    state = {m: {k: stat(st[m], v, keys) for k, v in pops.items()} for m in st if st[m]}
    repair = {k: {m: summ(rp[m], v) for m in rp if rp[m]} for k, v in ({"ALL": rep} | cats | size | fam | shift).items()}
    repair["feasible_negative"] = {m: c2.negative_rates(rp[m], neg) if set(rp[m]) == set(SEEDS) else None for m in rp}
    strata = {"ALL repairable": rep} | cats | {k: size[k] for k in ("K*=1", "K*=2")} | fam
    boots = {f"FIT - {ref}": {k: {m: paired(rp[ref], rp["FIT"], v, m) for m in ("hit", "valid_feasible")} for k, v in strata.items()}
             for ref in ("C2", "REL-CF-PAIR-SHIFT")} if rp["FIT"] else {}
    gap = {}
    for kind in ("FIT", "FINAL"):
        for s in st[kind]:
            tr_ = runs[s][kind.lower()]["train"]
            va = stat({s: st[kind][s]}, list(range(N)), ("balanced_accuracy", "recall_F0", "recall_F1"))
            gap.setdefault(kind, {})[s] = {"epoch": runs[s][kind.lower()]["epoch"]} | {
                k: {"train": tr_[k], "val": va[k]["mean"], "delta": tr_[k] - va[k]["mean"]} for k in ("balanced_accuracy", "recall_F0", "recall_F1")}
    train_samples = sc.dense_samples(records, arrays, train_rows, compact)   # same scenes as C.3's magnitudes
    coef = {kind: {s: c3.coefficients(fz.load(runs[s][kind.lower()]["state"]), train_samples) for s in st[kind]}
            for kind in ("FIT", "FINAL")}
    out = {"stage": "4.6C.4", "git_sha": env["git_sha"], "git_dirty": env["git_dirty"], "dataset_digest": ds.STAGE2_DIGEST,
           "environment": env, "stage2_test_split_evaluated": False, "future_audit_seed_61_generated": False,
           "rows_used": {"train": len(train_rows), "val": N, "test": 0}, "rules": __doc__.strip(), "dense_label_stability": stab,
           "category_counts": counts, "training": {s: {"fit_epoch": None if runs[s]["fit"] is None else runs[s]["fit"]["epoch"],
                                                        "fit_step": None if runs[s]["fit"] is None else runs[s]["fit"]["step"],
                                                        "runtime_s": runs[s]["runtime_s"], "records": runs[s]["records"]} for s in SEEDS},
           "fitted_seeds": fitted, "state_feasibility": state, "repair_selection": repair, "paired_bootstraps": boots,
           "train_validation_gap": gap, "coefficient_magnitudes_train_scenes": coef}
    if not fitted:
        (OUT / "metrics.json").write_text(json.dumps(out | {"outcome": "INSTABILITY", "outcome_label": LABEL["INSTABILITY"]},
                                                     indent=1, default=float))
        sys.exit("TRAIN-FIT INSTABILITY: no seed fitted")
    U = state["FIT"]["U-EXACT"]
    u = {"ba": U["balanced_accuracy"]["mean"], "r0": U["recall_F0"]["mean"], "r1": U["recall_F1"]["mean"],
         "vf": repair["U-EXACT"]["FIT"]["valid_feasible"]["mean"], "hit": repair["U-EXACT"]["FIT"]["hit"]["mean"],
         "dhit": boots["FIT - REL-CF-PAIR-SHIFT"]["U-EXACT"]["hit"]["mean_diff"],
         "dhit_ci_lo": boots["FIT - REL-CF-PAIR-SHIFT"]["U-EXACT"]["hit"]["ci95"][0]}
    crit = sc.outcome(u)
    c2u = boots["FIT - C2"]["U-EXACT"]["hit"]
    improves = c2u["ci95"][0] > 0 or u["ba"] - state["C2"]["U-EXACT"]["balanced_accuracy"]["mean"] >= 0.02
    o = outcome(len(fitted) == len(SEEDS), crit, u, improves)
    ff = paired(rp["FIT"], rp["FINAL"], rep, "hit") if fitted else None
    dba = state["FINAL"]["all_validation"]["balanced_accuracy"]["mean"] - state["FIT"]["all_validation"]["balanced_accuracy"]["mean"]
    overfit = bool(ff and ((ff["mean_diff"] <= -0.03 and ff["ci95"][1] < 0) or dba <= -0.02))
    out |= {"interpretation_criteria": crit | {"improves_c2": improves, "c2_U_EXACT_hit_bootstrap": c2u},
            "final_minus_fit": {"repairable_hit": ff, "validation_BA_delta": dba}, "OVERFITTING_AFTER_FIT": overfit,
            "outcome": o, "outcome_label": LABEL[o]}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"fitted": fitted, "fit_epochs": {s: out["training"][s]["fit_epoch"] for s in SEEDS}, "u": u,
                      "criteria": crit["criteria"], "outcome": o, "label": LABEL[o], "overfit": overfit}, indent=1, default=float))


if __name__ == "__main__":
    main()
