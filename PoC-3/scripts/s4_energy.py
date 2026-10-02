"""PoC-3 Stage 4: unary vs pairwise learned intervention energy under exact inference (plan3.md section 51).

Run from the repository root:

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s4_energy.py

Strict order (test isolation): load the frozen dataset -> train + validation samples only -> 20-scene
overfit gate (pairwise, seed 7; STOP on failure) -> 3 unary + 3 pairwise trainings (spawned processes,
one CPU thread each) with validation-only checkpoint selection and temperature calibration -> checkpoints
frozen -> only then the test samples and test oracle tables are built and the test set is evaluated once.
Writes PoC-3/out/s4/metrics.json and PoC-3/out/s4/checkpoints/{unary,pairwise}_seed{7,17,27}.pt.
"""

import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from poc2 import dataset as pd
from poc3 import RunEnvironment
from poc3 import dataset as ds
from poc3 import metrics as mt
from poc3 import model as md
from poc3 import train as tr

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s4"
KINDS = {"unary": False, "pairwise": True}
EXTRA = mt.SET_KEYS + tuple(f"{k}_{t}" for t in ("T1", "cal") for k in ("nll", "kl", "mass", "brier", "mae"))
HYPER = {"optimizer": "AdamW", "lr": tr.LR, "weight_decay": tr.WEIGHT_DECAY, "max_epochs": tr.MAX_EPOCHS,
         "patience": tr.PATIENCE, "batch_scenes": tr.BATCH_SCENES, "seeds": list(tr.SEEDS), "T_train": 1.0,
         "loss": "exact set NLL over the admissible domain, mean over S* ties, equal weight per scene",
         "selection": "validation mean scene-level set NLL", "energy_tie_tol": tr.ENERGY_TIE_TOL,
         "T_grid": "logspace(log10(0.05), log10(20), 241)"}


def _load():
    return ds.load(ROOT / "s2")  # verifies the pinned Stage-2 content digest


def _train_kind(job) -> dict:
    kind, seed = job
    records, arrays = _load()
    train_set = tr.structured_samples_from_dataset(records, arrays, tr.split_rows(records, "train"))
    val_set = tr.structured_samples_from_dataset(records, arrays, tr.split_rows(records, "val"))
    run = tr.train(train_set, val_set, seed, pairwise=KINDS[kind], loss_fn=tr.structured_loss, name="set_nll")
    model = tr.load_model(run["state_dict"], KINDS[kind])
    run["calibration"] = tr.calibrate_temperature([tr.energies(model, s) for s in val_set],
                                                  [s.target.numpy() for s in val_set])
    return run | {"kind": kind}


def _oracle_table(args) -> tuple:
    spec_json, admissible, optimal, P = args
    lab = ds.oracle(pd.spec_from_json(spec_json))
    t = lab["table"]
    ok = lab["admissible_x"] == ds.unpack_states(admissible, P) and lab["S_star_x"] == ds.unpack_states(optimal, P)
    return t.M, t.V, t.F, t.K, ok


def evaluate(run: dict, test_set, tables) -> list[dict]:
    model, T = tr.load_model(run["state_dict"], KINDS[run["kind"]]), run["calibration"]["T"]
    rows = []
    for s, (M, V, F, K, _) in zip(test_set, tables):
        E = tr.energies(model, s)
        top, learned = tr.exact_inference(E, s)
        row = mt.score_state(top, M, V, F, K, s.optimal) | mt.set_scores(learned, s.optimal)
        for tag, temp in (("T1", 1.0), ("cal", T)):
            d = mt.distribution_scores(E, s.states, s.optimal, len(s.cost), temp)
            assert abs(d["prob_sum"] - 1.0) < 1e-9
            row |= {f"{k}_{tag}": d[k] for k in ("nll", "kl", "mass", "brier", "mae")}
        rows.append(row | {"x_hat": top, "n_learned_min": len(learned)} | {k: s.meta[k] for k in ("scene_id", "family",
                                                                                                    "intent")})
    return rows


def comparisons(rows: dict, test_set) -> dict:
    """Paired bootstrap of pairwise - unary, seed-averaged per test scene."""
    out = {}
    for key in ("hit", "valid_feasible"):
        d = np.mean([[float(rows["pairwise"][sd][i][key]) - float(rows["unary"][sd][i][key])
                      for i in range(len(test_set))] for sd in tr.SEEDS], axis=0)
        pops = {"all": np.arange(len(test_set)),
                "repairable": np.array([i for i, s in enumerate(test_set) if s.meta["intent"] == "repairable"])}
        pops |= {f"repairable/{f}": np.array([i for i, s in enumerate(test_set) if s.meta["intent"] == "repairable"
                                              and s.meta["family"] == f]) for f in ds.FAMILIES}
        out[key] = {name: mt.paired_bootstrap(d[idx]) for name, idx in pops.items()}
    return out


def learning_gate(agg: dict, collapse: dict, stage3: dict) -> dict:
    """plan3.md Stage-4 gate on the pairwise model (thresholds fixed in the plan, applied unchanged)."""
    o = agg["pairwise"]["test"]["overall"]
    s3_all, s3_rep = stage3["all"]["hit"]["mean"], stage3["repairable"]["hit"]["mean"]
    checks = {"hit_all_ge_0.90": o["all"]["hit"]["mean"] >= 0.90,
              "valid_feasible_all_ge_0.99": o["all"]["valid_feasible"]["mean"] >= 0.99,
              "improves_over_stage3_all_and_repairable": o["all"]["hit"]["mean"] > s3_all
              and o["repairable"]["hit"]["mean"] > s3_rep,
              "no_collapse_any_seed": not any(c["collapsed"] for c in collapse["pairwise"].values())}
    report = {"pairwise_hit_all": o["all"]["hit"], "pairwise_hit_repairable": o["repairable"]["hit"],
              "pairwise_valid_feasible_all": o["all"]["valid_feasible"],
              "pairwise_valid_feasible_repairable": o["repairable"]["valid_feasible"],
              "stage3_hit_all": s3_all, "stage3_hit_repairable": s3_rep,
              "repairable_hit_ge_0.90": o["repairable"]["hit"]["mean"] >= 0.90,
              "repairable_valid_feasible_ge_0.99": o["repairable"]["valid_feasible"]["mean"] >= 0.99}
    return {"checks": checks, "passed": all(checks.values()), "repairable_view": report}


def main() -> None:
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = RunEnvironment.capture().to_json() | {"device": "cpu", "threads_per_run": 1, "deterministic_algorithms": True}
    records, arrays = _load()                                                   # 1. frozen dataset
    gate_set = tr.structured_samples_from_dataset(records, arrays, tr.gate_rows(records))  # train rows only
    gate = tr.overfit_gate(gate_set) | {"scene_ids": [s.meta["scene_id"] for s in gate_set]}  # 3. overfit gate
    print(json.dumps({"overfit_gate": {k: gate[k] for k in ("hits", "n", "passed", "runtime_s")}}), flush=True)
    base = {"stage": 4, "dataset_digest": ds.STAGE2_DIGEST, "hyperparameters": HYPER, "environment": env,
            "overfit_gate": gate}
    if not gate["passed"]:
        (OUT / "metrics.json").write_text(json.dumps(base | {"stopped": "overfit gate failed"}, indent=1))
        sys.exit("STOP: 20-scene overfit gate failed")
    t0 = time.perf_counter()                                                    # 4-7. train, select, calibrate
    jobs = [(k, sd) for k in KINDS for sd in tr.SEEDS]
    with ProcessPoolExecutor(len(jobs), mp_context=mp.get_context("spawn")) as pool:
        runs = {(r["kind"], r["seed"]): r for r in pool.map(_train_kind, jobs)}
    t_train = time.perf_counter() - t0
    for (kind, seed), r in runs.items():
        torch.save({"state_dict": r["state_dict"], "model_kind": kind, "seed": seed, "best_epoch": r["best_epoch"],
                    "best_val_set_nll": r["best_val_set_nll"], "calibrated_temperature": r["calibration"]["T"],
                    "hyperparameters": HYPER, "dataset_digest": ds.STAGE2_DIGEST},
                   OUT / "checkpoints" / f"{kind}_seed{seed}.pt")
    t1 = time.perf_counter()                                                    # 8. only now: the test split
    test_rows = tr.split_rows(records, "test")
    with ProcessPoolExecutor(12, mp_context=mp.get_context("spawn")) as pool:
        tables = list(pool.map(_oracle_table, [(records[j]["spec"], arrays["admissible"][j], arrays["optimal"][j],
                                                int(arrays["P"][j])) for j in test_rows], chunksize=4))
    if not all(t[4] for t in tables):
        sys.exit("STOP: test oracle tables disagree with the stored Stage-2 labels")
    test_set = tr.structured_samples_from_dataset(records, arrays, test_rows)
    rows = {k: {sd: evaluate(runs[(k, sd)], test_set, tables) for sd in tr.SEEDS} for k in KINDS}  # 9. once
    t_eval = time.perf_counter() - t1
    per_seed = {k: {sd: {"test": mt.grouped(rows[k][sd], ds.FAMILIES, EXTRA), "collapse": mt.collapse(rows[k][sd]),
                         "best_epoch": runs[(k, sd)]["best_epoch"], "epochs_run": runs[(k, sd)]["epochs_run"],
                         "best_val_set_nll": runs[(k, sd)]["best_val_set_nll"],
                         "calibration": runs[(k, sd)]["calibration"], "train_runtime_s": runs[(k, sd)]["runtime_s"]}
                    for sd in tr.SEEDS} for k in KINDS}
    agg = {k: {"test": mt.across_seeds([per_seed[k][sd]["test"] for sd in tr.SEEDS]),
               "best_val_set_nll": mt.across_seeds([per_seed[k][sd]["best_val_set_nll"] for sd in tr.SEEDS]),
               "T_cal": mt.across_seeds([per_seed[k][sd]["calibration"]["T"] for sd in tr.SEEDS])} for k in KINDS}
    s3 = json.loads((ROOT / "s3" / "metrics.json").read_text())["across_seeds"]["test"]["overall"]
    collapse = {k: {sd: per_seed[k][sd]["collapse"] for sd in tr.SEEDS} for k in KINDS}
    comp = comparisons(rows, test_set)
    rep = comp["hit"]
    fam_up = sum(rep[f"repairable/{f}"]["mean_diff"] > 0 for f in ds.FAMILIES)
    out = base | {
        "parameters": {k: md.n_parameters(md.EnergyModel(pairwise=v)) for k, v in KINDS.items()},
        "runtime_s": {"training_wall_clock_6_runs_parallel": t_train, "test_tables_and_evaluation": t_eval,
                      "per_run": {f"{k}_seed{sd}": runs[(k, sd)]["runtime_s"] for k, sd in runs}},
        "histories": {f"{k}_seed{sd}": runs[(k, sd)]["history"] for k, sd in runs},
        "per_seed": per_seed, "across_seeds": agg,
        "stage3_independent_reference": {g: {m: s3[g][m] for m in ("hit", "valid_feasible", "empty", "mean_n_pred")}
                                         for g in ("all", "repairable", "negative")},
        "unary_vs_pairwise_paired_bootstrap": comp,
        "preliminary_pairwise_indicator_section56": {
            "delta_hit_all": rep["all"]["mean_diff"], "ci95": rep["all"]["ci95"],
            "families_repairable_improved": fam_up,
            "meets_5pp_ci_and_two_families": rep["all"]["mean_diff"] >= 0.05 and rep["all"]["ci95"][0] > 0
            and fam_up >= 2, "note": "final verdict belongs to Stage 7"},
        "learning_gate": learning_gate(agg, collapse, s3),
        "test_predictions": {"scene_ids": [s.meta["scene_id"] for s in test_set],
                             **{f"{k}_seed{sd}": [r["x_hat"] for r in rows[k][sd]] for k in KINDS for sd in tr.SEEDS}},
    }
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1))
    keys = ("hit", "valid_feasible", "set_recovery", "empty", "mean_n_pred", "nll_cal", "mass_cal", "brier_cal")
    print(json.dumps({"runtime_s": out["runtime_s"]["training_wall_clock_6_runs_parallel"],
                      **{k: {g: {m: round(agg[k]["test"]["overall"][g][m]["mean"], 4) for m in keys}
                             for g in ("all", "repairable")} for k in KINDS},
                      "bootstrap_hit": {n: [round(v["mean_diff"], 4), [round(c, 4) for c in v["ci95"]]]
                                        for n, v in rep.items()},
                      "learning_gate": out["learning_gate"]["checks"]}, indent=1))


if __name__ == "__main__":
    main()
