"""PoC-3 Stage 4.6C.2: dense-F full generalization test (approved diagnostic exception; train + validation only).

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s46c2_dense_f_full.py

Exactly Stage 4.6C (frozen feasibility.FeasibilityModel, dense-F loss, threshold 0, min-cost predicted-feasible
repair, full-training protocol 200 epochs / patience 20 / lr 1e-3 / batch 32, validation dense-BCE selection, frozen
S / D / R / M rules incl. D-before-R precedence) with ONE change justified by train-only Stage 4.6C.1 (Outcome O):
the 20-scene implementation gate runs 1000 epochs = 1000 single-batch steps (patience 1000) instead of 400. Gate
criteria unchanged (BA >= 0.98, >= 19 / 20). Crossing epochs are replayed with the C.1 probe (verified to reproduce
tr.train). The 4.6C functions are imported, not copied. No test rows; seed 61 not generated; no calibration.
"""

import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s46a_error_audit as au  # noqa: E402
import s46c1_gate_diagnosis as gd  # noqa: E402
import s46c_feasibility_supervision as sc  # noqa: E402

from poc3 import RunEnvironment  # noqa: E402
from poc3 import dataset as ds  # noqa: E402
from poc3 import feasibility as fz  # noqa: E402
from poc3 import metrics as mt  # noqa: E402
from poc3 import relational as rl  # noqa: E402
from poc3 import train as tr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s46c2"
GATE_EPOCHS = 1000
LABEL = {"S": "DENSE FEASIBILITY SUPERVISION SUCCEEDS", "R": "DENSE FEASIBILITY SUPERVISION DOES NOT RESCUE GENERALIZATION",
         "D": "FEASIBILITY SIGNAL LEARNED; REPAIR DECISION REMAINS", "M": "MIXED"}
CARD = ("x0", "x1", "x2plus")


def card_masks(s) -> dict:
    c = s.bits.numpy().sum(axis=1)
    return {"x0": c == 0, "x1": c == 1, "x2plus": c >= 2}


def check_reference() -> dict:
    ref = {}
    for seed in tr.SEEDS:
        path = ROOT / "s46b" / "checkpoints" / f"rel_pair_shift_seed{seed}.pt"
        if not path.exists():
            sys.exit(f"STOP: missing frozen checkpoint {path}")
        ck = torch.load(path)
        if (ck["variant"], ck["pair"], ck["seed"], ck["dataset_digest"], ck["architecture"]) != \
                ("REL-CF-PAIR-SHIFT", "shift", seed, ds.STAGE2_DIGEST, sc.REF_ARCH):
            sys.exit(f"STOP: frozen REL-CF-PAIR-SHIFT seed {seed} failed integrity")
        ref[seed] = rl.load(ck["state_dict"], "cf", "shift")
    return ref


def check_c1() -> dict:
    m = json.loads((ROOT / "s46c1" / "metrics.json").read_text())
    rec = m["joint_optimization_runs"]["A"]["records"]
    first = next(r["epoch"] for r in rec if r["ba"] >= 0.98 and r["hits"] >= 19)
    if m["outcome"] != "O" or m["git_dirty"]:
        sys.exit("STOP: Stage-4.6C.1 Outcome O not present from a clean commit")
    return {"outcome": m["outcome"], "git_sha": m["git_sha"], "setting_A_first_gate_epoch_recorded_every_25": first}


def first_epoch(records, pred):
    return next((r["epoch"] for r in records if pred(r)), None)


def negative_rates(rows_by_seed: dict, idx) -> dict:
    """Feasible negatives: correct empty prediction, false repair (non-empty), no predicted feasible state."""
    def rate(f):
        v = [float(np.mean([f(rows_by_seed[s][i]) for i in idx])) for s in tr.SEEDS]
        return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
    return {"n": len(idx), "correct_empty": rate(lambda r: r["x_hat"] == 0),
            "false_repair": rate(lambda r: r["x_hat"] not in (0, None)), "no_predicted_feasible": rate(lambda r: r["x_hat"] is None)}


def summ(rows_by_seed: dict, idx) -> dict:
    return mt.across_seeds([mt.summarize([rows_by_seed[s][i] for i in idx], ("no_feasible",)) for s in tr.SEEDS]) \
        if idx else {"n": 0}


def main() -> None:
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = RunEnvironment.capture().to_json()
    records, arrays = ds.load(ROOT / "s2")
    train_rows, val_rows = sc.guard(records, tr.split_rows(records, "train")), sc.guard(records, tr.split_rows(records, "val"))
    ref, c1 = check_reference(), check_c1()
    rows = train_rows + val_rows
    rep_val = au.guard(records, [j for j in val_rows if records[j]["intent"] == "repairable"])
    with ProcessPoolExecutor(12, mp_context=mp.get_context("spawn")) as pool:
        lab = list(pool.map(sc._labels, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                          arrays["optimal"][j], records[j]["split"] == "val") for j in rows], chunksize=8))
        cat_scenes = list(pool.map(au._scene, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                                arrays["optimal"][j]) for j in rep_val], chunksize=4))
    labels = dict(zip(rows, lab))
    if not all(L["consistent"] for L in lab):
        sys.exit("STOP: oracle tables disagree with the stored Stage-2 labels")
    stab = {}
    for name, sel in [("train", train_rows), ("val", val_rows)]:
        n, d = sum(len(labels[j]["states"]) for j in sel), sum(int((labels[j]["F"] != labels[j]["F_fine"]).sum()) for j in sel)
        stab[name] = {"admissible_states": n, "F_disagreements": d, "fraction": d / n}
    category = {j: o["category"] for j, o in zip(rep_val, cat_scenes)}
    counts = {c: list(category.values()).count(c) for c in au.CATEGORIES}
    base = {"stage": "4.6C.2", "git_sha": env["git_sha"], "git_dirty": env["git_dirty"], "dataset_digest": ds.STAGE2_DIGEST,
            "environment": env, "stage2_test_split_evaluated": False, "future_audit_seed_61_generated": False,
            "rows_used": {"train": len(train_rows), "val": len(val_rows), "test": 0}, "stage46c1_check": c1,
            "dense_label_stability": stab | {"policy": sc.POLICY}, "category_counts": counts,
            "note": "l(x) is a feasibility polynomial, not the final repair energy; no temperature calibration"}
    if max(stab["train"]["fraction"], stab["val"]["fraction"]) > sc.MAX_DISAGREEMENT:
        (OUT / "metrics.json").write_text(json.dumps(base | {"stopped": "dense labels unstable"}, indent=1))
        sys.exit("STOP: dense F labels unstable")
    if counts != sc.EXPECTED_COUNTS:
        sys.exit(f"STOP: Stage-4.6A categories not reproduced: {counts}")
    compact = {j: {k: labels[j][k] for k in ("states", "F", "F_fine")} for j in rows}
    gate_set = sc.dense_samples(records, arrays, gd.gate_rows_checked(records), compact)
    g = sc.gate(gate_set, epochs=GATE_EPOCHS)                                    # unchanged 4.6C gate, 1000 steps
    traj = gd.probe_train(gate_set, tr.LR, tr.WEIGHT_DECAY, GATE_EPOCHS, tr.GATE_BATCH, record_every=1)["records"]
    base["corrected_gate"] = g | {"epochs": GATE_EPOCHS, "optimizer_steps": GATE_EPOCHS, "lr": tr.LR,
                                  "first_epoch_ba_ge_0.98": first_epoch(traj, lambda r: r["ba"] >= sc.GATE_BA),
                                  "first_epoch_hits_ge_19": first_epoch(traj, lambda r: r["hits"] >= tr.GATE_MIN_HITS),
                                  "first_epoch_both": first_epoch(traj, lambda r: r["ba"] >= sc.GATE_BA and r["hits"] >= 19),
                                  "best_ba": max(r["ba"] for r in traj), "best_hits": max(r["hits"] for r in traj)}
    print(json.dumps({"stability": stab, "gate": base["corrected_gate"]}, default=float), flush=True)
    if not g["passed"]:
        (OUT / "metrics.json").write_text(json.dumps(base | {"stopped": "corrected implementation gate failed"}, indent=1))
        sys.exit("STOP: corrected 1000-step dense-F gate failed")
    with ProcessPoolExecutor(3, mp_context=mp.get_context("spawn")) as pool:   # unchanged 4.6C full training
        runs = {r["seed"]: r for r in pool.map(sc._train, [(s, compact) for s in tr.SEEDS])}
    for seed, r in runs.items():
        torch.save({"state_dict": r["state_dict"], "seed": seed, "dataset_digest": ds.STAGE2_DIGEST,
                    "architecture": fz.__doc__.strip() + "\n\n" + sc.REF_ARCH, "dense_target": "F", "threshold_logit": 0.0,
                    "best_val_dense_f_bce": r["best_val_dense_f_bce"], "best_epoch": r["best_epoch"], "calibrated": False,
                    "training_hyperparameters": sc.HYPER, "gate_budget": GATE_EPOCHS, "stability_policy": sc.POLICY},
                   OUT / "checkpoints" / f"dense_f_seed{seed}.pt")
    val, train = sc.dense_samples(records, arrays, val_rows, compact), sc.dense_samples(records, arrays, train_rows, compact)
    for s, j in zip(val, val_rows):
        labels[j]["optimal"] = s.optimal
    state, dense, refr, gap = {}, {}, {}, {}
    for seed in tr.SEEDS:
        model = fz.load(runs[seed]["state_dict"])
        state[seed], dense[seed], refr[seed] = [], [], []
        with torch.no_grad():
            for s, j in zip(val, val_rows):
                lg = fz.feasibility_logits(model, s).numpy()
                state[seed].append(fz.state_metrics(lg, s) | {f"{k}_{c}": v for c, m in card_masks(s).items()
                                                               for k, v in fz.state_metrics(lg, s, m).items()})
                x = fz.derive_repair(lg, s)
                dense[seed].append(sc.repair_row(x, labels[j]) | {"x_hat": x, "family": s.meta["family"], "intent": s.meta["intent"]})
                ss = tr.StructuredSample(s.inputs, torch.as_tensor(s.cost), s.states, s.bits,
                                         torch.as_tensor(np.isin(s.states, sorted(s.optimal))), s.optimal, s.meta)
                xr = tr.exact_inference(tr.energies(ref[seed], ss), ss)[0]
                refr[seed].append(sc.repair_row(xr, labels[j]) | {"x_hat": xr, "family": s.meta["family"], "intent": s.meta["intent"]})
            tr_ba = [fz.state_metrics(fz.feasibility_logits(model, s).numpy(), s)["balanced_accuracy"] for s in train]
        h = runs[seed]["history"]
        va_ba = [r["balanced_accuracy"] for r in state[seed]]
        gap[seed] = {"best_epoch": runs[seed]["best_epoch"], "epochs_run": runs[seed]["epochs_run"],
                     "train_bce_at_best_epoch": h["train_dense_f_bce"][runs[seed]["best_epoch"]],
                     "final_train_bce": h["train_dense_f_bce"][-1], "best_train_bce": min(h["train_dense_f_bce"]),
                     "best_val_bce": runs[seed]["best_val_dense_f_bce"], "train_BA_selected": float(np.nanmean(tr_ba)),
                     "val_BA_selected": float(np.nanmean(va_ba)), "runtime_s": runs[seed]["runtime_s"],
                     "per_family_BA": {f: {"train": float(np.nanmean([b for b, s in zip(tr_ba, train) if s.meta["family"] == f])),
                                           "val": float(np.nanmean([b for b, s in zip(va_ba, val) if s.meta["family"] == f]))}
                                       for f in ds.FAMILIES}}
    N = len(val_rows)
    rep = [i for i in range(N) if records[val_rows[i]]["intent"] == "repairable"]
    neg = [i for i in range(N) if records[val_rows[i]]["intent"] == "negative"]
    cats = {c: [i for i in rep if category[val_rows[i]] == c] for c in au.CATEGORIES}
    shift = {f"SHIFT={v}": [i for i in rep if labels[val_rows[i]]["shift"] == v] for v in (True, False)}
    size = {f"K*={k}": [i for i in rep if labels[val_rows[i]]["K_star"] == k] for k in (1, 2, 3, 4)}
    fam = {f"family={f}": [i for i in rep if records[val_rows[i]]["family"] == f] for f in ds.FAMILIES}
    keys = sc.STATE_KEYS + tuple(f"{k}_{c}" for c in CARD for k in sc.STATE_KEYS)
    pops = {"all_validation": list(range(N)), "repairable": rep, "feasible_negative": neg} | cats | shift | size | fam
    sf = {k: sc.strata_mean(state, v, keys) for k, v in pops.items()}
    cardinality = {p: {c: {k: sf[p][f"{k}_{c}"] for k in sc.STATE_KEYS} for c in CARD} for p in ("repairable", "U-EXACT")}
    sel = {"all_repairable": rep} | cats | shift | size | fam
    repair = {k: {"DENSE-F": summ(dense, v), "REL-CF-PAIR-SHIFT": summ(refr, v)} for k, v in sel.items()}
    repair["feasible_negative"] = {"DENSE-F": negative_rates(dense, neg), "REL-CF-PAIR-SHIFT": negative_rates(refr, neg)}
    boots = {k: {m: sc.paired(refr, dense, v, m) for m in ("hit", "valid_feasible")}
             for k, v in ({"ALL repairable": rep} | cats | {k: size[k] for k in ("K*=1", "K*=2")} | fam).items()}
    u = {"ba": sf["U-EXACT"]["balanced_accuracy"]["mean"], "r0": sf["U-EXACT"]["recall_F0"]["mean"],
         "r1": sf["U-EXACT"]["recall_F1"]["mean"], "vf": repair["U-EXACT"]["DENSE-F"]["valid_feasible"]["mean"],
         "hit": repair["U-EXACT"]["DENSE-F"]["hit"]["mean"], "dhit": boots["U-EXACT"]["hit"]["mean_diff"],
         "dhit_ci_lo": boots["U-EXACT"]["hit"]["ci95"][0], "n_U_EXACT": len(cats["U-EXACT"]),
         "reference_hit_U_EXACT": repair["U-EXACT"]["REL-CF-PAIR-SHIFT"]["hit"]["mean"]}
    verdict = sc.outcome(u)
    out = base | {
        "full_training": {s: {k: runs[s][k] for k in ("best_epoch", "epochs_run", "best_val_dense_f_bce", "runtime_s")}
                          for s in tr.SEEDS} | {"hyperparameters": sc.HYPER},
        "histories": {s: runs[s]["history"] for s in tr.SEEDS}, "train_validation_gap": gap,
        "state_feasibility": {k: sf[k] for k in ("all_validation", "repairable", "feasible_negative", *au.CATEGORIES)}
        | {"by_shift": {k: sf[k] for k in shift}, "by_family": {k: sf[k] for k in fam}, "by_repair_size": {k: sf[k] for k in size},
           "cardinality": cardinality},
        "repair_selection": repair,
        "frozen_set_nll_reference": {"model": "REL-CF-PAIR-SHIFT (Stage 4.6B), argmin E over the admissible domain",
                                     "checkpoints": "PoC-3/out/s46b/checkpoints/rel_pair_shift_seed{7,17,27}.pt"},
        "paired_bootstraps": boots, "interpretation_criteria": verdict, "outcome": verdict["outcome"],
        "outcome_label": LABEL[verdict["outcome"]], "recommended_next_stage": sc.NEXT[verdict["outcome"]],
    }
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"outcome": verdict, "label": LABEL[verdict["outcome"]]}, indent=1, default=float))


if __name__ == "__main__":
    main()
