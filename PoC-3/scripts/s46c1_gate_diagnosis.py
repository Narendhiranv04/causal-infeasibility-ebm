"""PoC-3 Stage 4.6C.1: dense-F gate failure diagnosis (train-only, gate-only; approved diagnostic exception).

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s46c1_gate_diagnosis.py

Only the fixed 20 gate scenes (first 5 repairable TRAIN scenes per family) ever enter a prediction or metric; a guard
rejects any other row. No validation, no test, no seed 61, no checkpoint. Threshold fixed: F_hat = 1[l >= 0].
  A  free per-scene SHIFT-masked quadratic l = b + q.x + sum_{p<r, p or r SHIFT} Q_pr x_p x_r fitted to stable dense F
     (BCE, zeros init, float64 L-BFGS); STRUCTURALLY_REALIZABLE iff every scene reaches BA >= 0.999 and 20 / 20 derived
     repair hits. An exact LP separability certificate (scipy HiGHS) is reported as supporting evidence only.
  B  same with Q on every pair, only if A fails.
  C  (only if A holds) fresh FeasibilityModel per scene, seed 7, AdamW lr 1e-3 wd 1e-5, batch 1, 2000 epochs;
     memorized iff BA >= 0.99 and derived repair in S* at some recorded epoch (every 25 epochs).
  D  (only if A holds) joint 20-scene runs, seed 7, batch 20, 2000 epochs, no early stopping, settings
     lr / wd = 1e-3 / 1e-5, 3e-3 / 1e-5, 3e-4 / 1e-5. "Reaches the gate": some recorded epoch has BA >= 0.98 and
     >= 19 hits. "Still decreasing": final train BCE < 0.99 x the train BCE 200 epochs earlier. The best run is the
     setting with the highest best BA, analysed at its best-BA state.
OUTCOMES (frozen): Q if A fails (B separates mask vs order-2); else L if < 18 / 20 scenes memorized; else O if some
joint run reaches the gate; else J.
"""

import copy
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fnn
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s46c_feasibility_supervision as sc  # noqa: E402  (frozen 4.6C labels and dense samples)

from poc3 import RunEnvironment  # noqa: E402
from poc3 import dataset as ds  # noqa: E402
from poc3 import feasibility as fz  # noqa: E402
from poc3 import relational as rl  # noqa: E402
from poc3 import train as tr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s46c1"
SETTINGS = {"A": (1e-3, 1e-5), "B": (3e-3, 1e-5), "C": (3e-4, 1e-5)}
EPOCHS, RECORD_EVERY, SEED = 2000, 25, 7
REALIZABLE_BA, MEMO_BA, GATE_BA, GATE_HITS, MEMO_MIN = 0.999, 0.99, 0.98, 19, 18


def gate_rows_checked(records) -> list[int]:
    rows = tr.gate_rows(records)
    for j in rows:
        if records[j]["split"] != "train" or records[j]["intent"] != "repairable":
            raise RuntimeError("Stage 4.6C.1 uses only the 20 repairable TRAIN gate scenes")
    return rows


def pair_list(candidates: np.ndarray, masked: bool) -> list[tuple[int, int]]:
    shift = candidates[:, rl.SHIFT] == 1
    return [(p, r) for p, r in combinations(range(len(candidates)), 2) if not masked or shift[p] or shift[r]]


def design(bits: np.ndarray, pairs) -> np.ndarray:
    """Phi(x) = [1, x_p ..., x_p x_r for the allowed pairs] for every admissible state."""
    return np.hstack([np.ones((len(bits), 1)), bits] + [(bits[:, p] * bits[:, r])[:, None] for p, r in pairs])


def fit_free(Phi: np.ndarray, y: np.ndarray, iters: int = 500) -> np.ndarray:
    """Deterministic float64 L-BFGS on BCE from zeros (convex in the coefficients)."""
    X, t = torch.as_tensor(Phi), torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-15,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = fnn.binary_cross_entropy_with_logits(X @ w, t)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()


def lp_separable(Phi: np.ndarray, y: np.ndarray) -> bool:
    """Exact strict separability: exists w with s_i Phi_i w >= 1 (s = +1 infeasible, -1 feasible)."""
    s = np.where(y == 1, 1.0, -1.0)
    res = linprog(np.zeros(Phi.shape[1]), A_ub=-(s[:, None] * Phi), b_ub=-np.ones(len(y)), bounds=(None, None),
                  method="highs")
    return bool(res.status == 0)


def margins(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, logits, -logits)


def margin_summary(m: np.ndarray) -> dict:
    return {"n": int(len(m)), "min": float(m.min()), "p5": float(np.percentile(m, 5)), "median": float(np.median(m)),
            "frac_negative": float(np.mean(m < 0)), "frac_abs_below_0.1": float(np.mean(np.abs(m) < 0.1))}


def scene_score(logits, s) -> dict:
    st_ = fz.state_metrics(logits, s)
    return {k: st_[k] for k in ("balanced_accuracy", "recall_F0", "recall_F1", "bce")} | \
        {"hit": fz.derive_repair(logits, s) in s.optimal}


def free_control(samples, masked: bool) -> dict:
    per, allm = [], []
    for s in samples:
        keep = s.stable.numpy()
        Phi = design(s.bits.numpy(), pair_list(s.inputs["candidates"].numpy(), masked))
        y = s.F.numpy()
        lg = Phi @ fit_free(Phi[keep], y[keep])
        r = scene_score(lg, s) | {"lp_separable": lp_separable(Phi[keep], y[keep]), "n_coefficients": Phi.shape[1]}
        r["perfect"] = r["balanced_accuracy"] >= REALIZABLE_BA and r["hit"]
        per.append(r | {"scene_id": s.meta["scene_id"]})
        allm.append(margins(lg[keep], y[keep]))
    return {"per_scene": per, "realizable": all(r["perfect"] for r in per), "n_perfect": sum(r["perfect"] for r in per),
            "n_lp_separable": sum(r["lp_separable"] for r in per), "margins": margin_summary(np.concatenate(allm))}


def evaluate(model, samples) -> dict:
    model.eval()
    with torch.no_grad():
        lgs = [fz.feasibility_logits(model, s).numpy() for s in samples]
    per = [scene_score(lg, s) for lg, s in zip(lgs, samples)]
    return {"ba": float(np.nanmean([p["balanced_accuracy"] for p in per])), "r0": float(np.nanmean([p["recall_F0"] for p in per])),
            "r1": float(np.nanmean([p["recall_F1"] for p in per])), "hits": int(sum(p["hit"] for p in per)), "per": per,
            "logits": lgs}


def probe_train(samples, lr: float, wd: float, epochs: int, batch: int, seed: int = SEED, record_every: int = RECORD_EVERY):
    """tr.train's exact seeding / init / shuffling / AdamW step, recording metrics every `record_every` epochs."""
    tr.set_determinism(seed)
    model = fz.FeasibilityModel()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    order_rng = np.random.default_rng(seed)
    losses, records, best = [], [], (-1.0, None, -1)
    for epoch in range(epochs):
        model.train()
        order, total, gn = order_rng.permutation(len(samples)), 0.0, 0.0
        for k in range(0, len(order), batch):
            chunk = [samples[i] for i in order[k:k + batch]]
            opt.zero_grad()
            loss = fz.dense_loss(model, chunk)
            loss.backward()
            gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)))
            opt.step()
            total += float(loss.detach()) * len(chunk)
        losses.append(total / len(samples))
        if (epoch + 1) % record_every == 0:
            ev = evaluate(model, samples)
            records.append({"epoch": epoch + 1, "train_bce": losses[-1], "ba": ev["ba"], "recall_F0": ev["r0"],
                            "recall_F1": ev["r1"], "hits": ev["hits"], "grad_norm": gn})
            if ev["ba"] > best[0]:
                best = (ev["ba"], copy.deepcopy(model.state_dict()), epoch + 1)
    return {"train_bce": losses, "records": records, "best_state": best[1], "best_epoch": best[2], "model": model}


def _single(args) -> dict:
    j, compact = args
    records, arrays = ds.load(ROOT / "s2")
    s = sc.dense_samples(records, arrays, [j], compact)[0]
    run = probe_train([s], *SETTINGS["A"], EPOCHS, 1)
    hit = [r for r in run["records"] if r["ba"] >= MEMO_BA and r["hits"] == 1]
    return {"scene_id": s.meta["scene_id"], "family": s.meta["family"], "memorized": bool(hit),
            "first_memorized_epoch": hit[0]["epoch"] if hit else None, "best_ba": max(r["ba"] for r in run["records"]),
            "final": run["records"][-1], "min_repair_size": min(int(x).bit_count() for x in s.optimal)}


def _joint(args) -> dict:
    name, compact, rows = args
    records, arrays = ds.load(ROOT / "s2")
    samples = sc.dense_samples(records, arrays, rows, compact)
    t0 = time.perf_counter()
    run = probe_train(samples, *SETTINGS[name], EPOCHS, tr.GATE_BATCH)
    rec, L = run["records"], run["train_bce"]
    best = max(rec, key=lambda r: r["ba"])
    return {"setting": name, "lr": SETTINGS[name][0], "weight_decay": SETTINGS[name][1], "records": rec,
            "best_ba": best["ba"], "best_ba_epoch": best["epoch"], "best_hits": max(r["hits"] for r in rec),
            "final_ba": rec[-1]["ba"], "final_hits": rec[-1]["hits"],
            "reaches_gate": any(r["ba"] >= GATE_BA and r["hits"] >= GATE_HITS for r in rec),
            "still_decreasing_final_200": L[-1] < 0.99 * L[-201], "best_state": run["best_state"],
            "runtime_s": time.perf_counter() - t0}


def per_scene_errors(model, samples, rows, arrays) -> tuple[list, dict]:
    ev, per, card_err = evaluate(model, samples), [], {"0": [0, 0], "1": [0, 0], ">=2": [0, 0]}
    for s, lg, sco, j in zip(samples, ev["logits"], ev["per"], rows):
        keep, y = s.stable.numpy(), s.F.numpy()
        card = s.bits.numpy().sum(axis=1).astype(int)
        wrong = (fz.predicted_infeasible(lg) != (y == 1)) & keep
        for c in range(len(card)):
            if keep[c]:
                key = "0" if card[c] == 0 else ("1" if card[c] == 1 else ">=2")
                card_err[key][0] += int(wrong[c])
                card_err[key][1] += 1
        per.append({"scene_id": s.meta["scene_id"], "family": s.meta["family"], "P": int(arrays["P"][j]),
                    "n_optimal": len(s.optimal), "min_repair_size": min(int(x).bit_count() for x in s.optimal),
                    "n_admissible": len(s.states), "fraction_feasible": float(np.mean(y == 0))} | sco
                   | {"misclassified_cardinalities": sorted(card[wrong].tolist())})
    return per, {k: {"errors": e, "states": n, "rate": e / n if n else None} for k, (e, n) in card_err.items()}


def outcome(A: dict, n_mem: int | None, any_gate: bool | None) -> str:
    if not A["realizable"]:
        return "Q"
    if n_mem < MEMO_MIN:
        return "L"
    return "O" if any_gate else "J"


LABEL = {"Q": "QUADRATIC / MASK STRUCTURE FAILURE", "L": "LOCAL REPRESENTATION / PARAMETERIZATION FAILURE",
         "O": "ORIGINAL 400-EPOCH GATE WAS OPTIMIZATION-LIMITED", "J": "SHARED REPRESENTATION / MULTI-SCENE FIT BOTTLENECK"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = RunEnvironment.capture().to_json()
    records, arrays = ds.load(ROOT / "s2")
    rows = gate_rows_checked(records)
    lab = [sc._labels((records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j], arrays["optimal"][j], False))
           for j in rows]
    if not all(L["consistent"] for L in lab):
        sys.exit("STOP: oracle tables disagree with the stored Stage-2 labels")
    compact = {j: {k: L[k] for k in ("states", "F", "F_fine")} for j, L in zip(rows, lab)}
    samples = sc.dense_samples(records, arrays, rows, compact)
    A = free_control(samples, masked=True)                                       # Diagnostic A
    B = free_control(samples, masked=False) if not A["realizable"] else None      # Diagnostic B (only if A fails)
    out = {"stage": "4.6C.1", "git_sha": env["git_sha"], "git_dirty": env["git_dirty"], "dataset_digest": ds.STAGE2_DIGEST,
           "environment": env, "stage2_test_split_evaluated": False, "validation_evaluated": False,
           "future_audit_seed_61_generated": False, "gate_scene_ids": [s.meta["scene_id"] for s in samples],
           "dense_label_disagreements": int(sum((L["F"] != L["F_fine"]).sum() for L in lab)),
           "rules": __doc__.strip(),
           "structural_realizability": A, "unrestricted_quadratic_control": B}
    n_mem = any_gate = None
    if A["realizable"]:
        spawn = mp.get_context("spawn")
        with ProcessPoolExecutor(12, mp_context=spawn) as pool:                 # Diagnostic C
            single = list(pool.map(_single, [(j, compact) for j in rows]))
        n_mem = sum(r["memorized"] for r in single)
        with ProcessPoolExecutor(3, mp_context=spawn) as pool:                  # Diagnostic D
            joint = {r["setting"]: r for r in pool.map(_joint, [(n, compact, rows) for n in SETTINGS])}
        any_gate = any(r["reaches_gate"] for r in joint.values())
        best = max(joint.values(), key=lambda r: r["best_ba"])
        model = fz.FeasibilityModel()
        model.load_state_dict(best["best_state"])
        per, card = per_scene_errors(model, samples, rows, arrays)
        ev = evaluate(model, samples)
        keep_m = np.concatenate([margins(lg[s.stable.numpy()], s.F.numpy()[s.stable.numpy()])
                                 for lg, s in zip(ev["logits"], samples)])
        out |= {"individual_scene_memorization": {"n_memorized": n_mem, "n": len(single), "per_scene": single,
                                                  "failed": [r for r in single if not r["memorized"]]},
                "joint_optimization_runs": {k: {kk: v for kk, v in r.items() if kk != "best_state"} for k, r in joint.items()},
                "best_joint_setting": best["setting"], "per_scene_errors": per, "cardinality_errors": card,
                "margin_diagnostics": {"best_joint_neural": margin_summary(keep_m), "free_shift_masked": A["margins"]}}
    else:
        out |= {"individual_scene_memorization": None, "joint_optimization_runs": None, "per_scene_errors": None,
                "cardinality_errors": None, "margin_diagnostics": {"free_shift_masked": A["margins"],
                                                                     "free_unrestricted": B["margins"]}}
    o = outcome(A, n_mem, any_gate)
    out |= {"outcome": o, "outcome_label": LABEL[o]}
    if o == "Q":
        out["mask_vs_order2"] = ("pair terms beyond the SHIFT mask are needed" if B["realizable"]
                                 else "quadratic sign structure is insufficient for dense F")
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"A": {k: A[k] for k in ("realizable", "n_perfect", "n_lp_separable", "margins")},
                      "B": None if B is None else {k: B[k] for k in ("realizable", "n_perfect", "n_lp_separable")},
                      "n_memorized": n_mem, "any_joint_gate": any_gate, "outcome": o, "label": LABEL[o]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
