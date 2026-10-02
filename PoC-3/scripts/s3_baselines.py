"""PoC-3 Stage 3: independent non-energy candidate baseline on the frozen Stage-2 dataset.

Run from the repository root:

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s3_baselines.py

Writes exactly PoC-3/out/s3/{metrics.json, best_seed7.pt, best_seed17.pt, best_seed27.pt}. The three seeds
train in parallel processes (spawned, one CPU thread each). Checkpoints are selected on validation BCE only;
the test split is evaluated once, after training, with the exact oracle table of every test scene (M, V,
F, K over all states; evaluation only, cross-checked against the stored Stage-2 masks).
"""

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from poc2 import dataset as pd
from poc3 import RunEnvironment
from poc3 import dataset as ds
from poc3 import metrics as mt
from poc3 import train as tr

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s3"


def _load():
    return ds.load(ROOT / "s2")  # verifies the pinned Stage-2 content digest


def _train_seed(seed: int) -> dict:
    records, arrays = _load()
    train_set = tr.samples_from_dataset(records, arrays, tr.split_rows(records, "train"))
    val_set = tr.samples_from_dataset(records, arrays, tr.split_rows(records, "val"))
    return tr.train(train_set, val_set, seed)


def _oracle_table(args) -> tuple:
    spec_json, admissible, optimal, P = args
    lab = ds.oracle(pd.spec_from_json(spec_json))  # frozen base-resolution PoC-2 table
    t = lab["table"]
    consistent = lab["admissible_x"] == ds.unpack_states(admissible, P) and lab["S_star_x"] == ds.unpack_states(optimal, P)
    return t.M, t.V, t.F, t.K, consistent


def evaluate(state_dict, test_set, tables) -> tuple[list[dict], dict]:
    model = tr.load_model(state_dict)
    rows = []
    for s, (M, V, F, K, _) in zip(test_set, tables):
        pi, x = tr.predict(model, s)
        rows.append(mt.score_state(x, M, V, F, K, s.optimal) | {"brier": mt.brier(pi, s.y.numpy()), "x_hat": x}
                    | {k: s.meta[k] for k in ("scene_id", "family", "intent")})
    return rows, mt.grouped(rows, ds.FAMILIES)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, arrays = _load()
    test_rows = tr.split_rows(records, "test")
    t0 = time.perf_counter()
    spawn = mp.get_context("spawn")
    with ProcessPoolExecutor(12, mp_context=spawn) as pool:
        tables = list(pool.map(_oracle_table, [(records[j]["spec"], arrays["admissible"][j], arrays["optimal"][j],
                                                int(arrays["P"][j])) for j in test_rows], chunksize=4))
    t_oracle = time.perf_counter() - t0
    if not all(t[4] for t in tables):
        raise SystemExit("STOP: test oracle tables disagree with the stored Stage-2 labels")
    t1 = time.perf_counter()
    with ProcessPoolExecutor(len(tr.SEEDS), mp_context=spawn) as pool:
        runs = list(pool.map(_train_seed, tr.SEEDS))
    t_train = time.perf_counter() - t1
    test_set = tr.samples_from_dataset(records, arrays, test_rows)
    per_seed, predictions = {}, {}
    for run in runs:
        seed = run["seed"]
        rows, groups = evaluate(run["state_dict"], test_set, tables)
        per_seed[seed] = {"best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
                          "best_val_bce": run["best_val_bce"], "train_runtime_s": run["runtime_s"], "test": groups}
        predictions[seed] = [r["x_hat"] for r in rows]
        torch.save({"state_dict": run["state_dict"], "seed": seed, "best_epoch": run["best_epoch"],
                    "best_val_bce": run["best_val_bce"], "history": run["history"],
                    "hyperparameters": tr.HYPERPARAMETERS, "dataset_digest": ds.STAGE2_DIGEST},
                   OUT / f"best_seed{seed}.pt")
    seeds = sorted(per_seed)
    out = {
        "stage": 3, "model": "independent candidate classifier = EnergyModel(pairwise=False), pi = sigmoid(q)",
        "conventions": tr.__doc__.strip(), "metric_definitions": mt.__doc__.strip(),
        "hyperparameters": tr.HYPERPARAMETERS, "dataset_digest": ds.STAGE2_DIGEST,
        "n_test": len(test_rows), "n_train": len(tr.split_rows(records, "train")),
        "n_val": len(tr.split_rows(records, "val")), "normalization": "fixed L0 scaling only (no normalization.json)",
        "environment": RunEnvironment.capture().to_json() | {"device": "cpu", "threads_per_seed": 1,
                                                             "deterministic_algorithms": True},
        "runtime_s": {"oracle_tables_test": t_oracle, "training_wall_clock_3_seeds_parallel": t_train,
                      "per_seed_training": {s: per_seed[s]["train_runtime_s"] for s in seeds}},
        "per_seed": per_seed,
        "across_seeds": {"test": mt.across_seeds([per_seed[s]["test"] for s in seeds]),
                         "best_val_bce": mt.across_seeds([per_seed[s]["best_val_bce"] for s in seeds]),
                         "best_epoch": mt.across_seeds([per_seed[s]["best_epoch"] for s in seeds])},
        "test_predictions": {"scene_ids": [records[j]["scene_id"] for j in test_rows],
                             **{f"seed{s}": predictions[s] for s in seeds}},
    }
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1))
    keys = ("hit", "valid_feasible", "M_violation", "V_violation", "infeasible", "empty", "mean_n_pred", "brier")
    agg = out["across_seeds"]["test"]["overall"]
    print(json.dumps({"runtime_s": out["runtime_s"], "best_epoch": {s: per_seed[s]["best_epoch"] for s in seeds},
                      **{g: {k: [round(agg[g][k]["mean"], 4), round(agg[g][k]["std"], 4)] for k in keys}
                         for g in ("all", "repairable", "negative")},
                      "negative_FP": agg["negative"]["false_positive_repair_rate"]}, indent=1))


if __name__ == "__main__":
    main()
