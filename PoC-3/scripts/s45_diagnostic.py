"""PoC-3 Stage 4.5 diagnostic: oracle pairwise structure, memorisation, and relational representation.

Run from the repository root:

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s45_diagnostic.py

TRAIN + VALIDATION ONLY: no oracle or model metric is computed on a split == "test" row (guarded). Order:
(B) frozen-oracle structure audit of every repairable train / val scene; (C) 200-scene train memorisation
(original unary, R1, R2; seed 7, 400 epochs, no early stopping); (I) R1 / R2 x seeds 7, 17, 27 on the full
training split with validation-only selection and calibration, plus the frozen Stage-4 unary checkpoints
evaluated on validation; (J) fixed selection rule. Writes PoC-3/out/s45/metrics.json and
PoC-3/out/s45/checkpoints/rel_{static,cf}_seed{7,17,27}.pt. A future fresh audit split (seed namespace 61,
90 + 30 per family) is predeclared here but NOT generated.
"""

import functools
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from poc.types import InterventionKind
from poc2 import dataset as pd
from poc2 import oracle as orc
from poc2 import structure as st
from poc3 import RunEnvironment
from poc3 import dataset as ds
from poc3 import metrics as mt
from poc3 import model as md
from poc3 import relational as rl
from poc3 import train as tr

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s45"
MEMO_PER_FAMILY, MEMO_EPOCHS, MEMO_SEED = 50, 400, 7
SELECTION_MARGIN = 0.01
VARIANTS = {"rel_static": "static", "rel_cf": "cf"}
FUTURE_AUDIT = {"seed_namespace": 61, "family_codes": [0, 1, 2, 3], "per_family": [90, 30], "total": 480,
                "status": "predeclared, not generated, not used by any Stage-4.5 result"}
EXTRA = mt.SET_KEYS + tuple(f"{k}_{t}" for t in ("T1", "cal") for k in ("nll", "kl", "mass", "brier", "mae"))


def _load():
    return ds.load(ROOT / "s2")  # verifies the pinned Stage-2 content digest


def rows_of(records, split: str) -> list[int]:
    if split not in ("train", "val"):
        raise RuntimeError("Stage 4.5 never evaluates the Stage-2 test split")
    return tr.split_rows(records, split)


def memo_rows(records) -> list[int]:
    out = []
    for fam in ds.FAMILIES:
        out += [j for j in rows_of(records, "train") if records[j]["family"] == fam
                and records[j]["intent"] == "repairable"][:MEMO_PER_FAMILY]
    return out


def _scene(args) -> dict:
    """Base table (M, V, F, K) and, for repairable scenes, the frozen-oracle decision / structure audit."""
    spec_json, P, adm, opt, intent = args
    spec = pd.spec_from_json(spec_json)
    lab = ds.oracle(spec)
    T = lab["table"]
    out = {"M": T.M, "V": T.V, "F": T.F, "K": T.K,
           "consistent": lab["admissible_x"] == ds.unpack_states(adm, P) and lab["S_star_x"] == ds.unpack_states(opt, P)}
    if intent != "repairable":
        return out
    scene, _, options = ds.build(spec)
    Tf = orc.repair_table(scene, options, ds.FINE_STEP)
    a_b, a_f = st.compatible_mobius(T.G, T.M), st.compatible_mobius(Tf.G, Tf.M)
    exact = lab["S_star_x"]
    ok = {k: st.order_decision(T, st.reconstruct(a_b, k)) == exact for k in range(1, P + 1)}  # exact coefficients
    edges = st.interaction_edges(T, a_b, st.significant(a_b, a_f))["geometric"]
    shifts = {p for p, o in enumerate(options) if o.intervention.kind is InterventionKind.SHIFT_TARGET}
    out["audit"] = {"unary_ok": ok[1], "pair_ok": ok.get(2, True), "third_ok": ok.get(3, True),
                    "decision_order": min(k for k in ok if ok[k]),
                    "representation_order": st.representation_order(T.G, T.M, a_b, st.eps_repr(T.G, Tf.G, T.M)),
                    "shift_offered": bool(shifts), "repair_size": min(int(x).bit_count() for x in exact),
                    "n_geo_edges": len(edges), "n_geo_edges_shift": sum(p in shifts or q in shifts for p, q, _, _ in edges)}
    return out


def audit_summary(audits: list[dict]) -> dict:
    def block(sel):
        if not sel:
            return {"n": 0}
        pair_needed = [a for a in sel if not a["unary_ok"] and a["pair_ok"]]
        edges = sum(a["n_geo_edges"] for a in sel)
        return {"n": len(sel), "unary_sufficient": float(np.mean([a["unary_ok"] for a in sel])),
                "pairwise_sufficient": float(np.mean([a["pair_ok"] for a in sel])),
                "third_order_sufficient": float(np.mean([a["third_ok"] for a in sel])),
                "R_pair": len(pair_needed) / len(sel), "n_decision_order_gt2": sum(a["decision_order"] > 2 for a in sel),
                "decision_order_hist": {str(k): sum(a["decision_order"] == k for a in sel) for k in range(1, 11)},
                "representation_order_hist": {str(k): sum(a["representation_order"] == k for a in sel) for k in range(11)},
                "n_significant_geo_pair_edges": edges,
                "frac_geo_edges_involving_shift": sum(a["n_geo_edges_shift"] for a in sel) / edges if edges else math.nan,
                "frac_pair_needed_with_shift": float(np.mean([a["shift_offered"] for a in pair_needed]))
                if pair_needed else math.nan}
    return {"overall": block(audits), "per_split": {s: block([a for a in audits if a["split"] == s]) for s in ("train", "val")},
            "per_family": {f: block([a for a in audits if a["family"] == f]) for f in ds.FAMILIES},
            "by_shift_offered": {str(v): block([a for a in audits if a["shift_offered"] == v]) for v in (True, False)},
            "by_repair_size": {str(k): block([a for a in audits if a["repair_size"] == k]) for k in range(1, 6)},
            "pair_needed_scene_ids": [a["scene_id"] for a in audits if not a["unary_ok"] and a["pair_ok"]],
            "decision_order_gt2_scene_ids": [a["scene_id"] for a in audits if a["decision_order"] > 2]}


def maker(kind: str):
    return None if kind == "original" else functools.partial(rl.RelationalEnergy, VARIANTS[kind])


def load_model(kind: str, state_dict):
    return tr.load_model(state_dict, False) if kind == "original" else rl.load(state_dict, VARIANTS[kind])


def _memo(kind: str) -> dict:
    records, arrays = _load()
    memo = tr.structured_samples_from_dataset(records, arrays, memo_rows(records))
    return tr.train(memo, memo, MEMO_SEED, max_epochs=MEMO_EPOCHS, patience=MEMO_EPOCHS, loss_fn=tr.structured_loss,
                    name="set_nll", make_model=maker(kind)) | {"kind": kind}


def _full(job) -> dict:
    kind, seed = job
    records, arrays = _load()
    train_set = tr.structured_samples_from_dataset(records, arrays, rows_of(records, "train"))
    val_set = tr.structured_samples_from_dataset(records, arrays, rows_of(records, "val"))
    run = tr.train(train_set, val_set, seed, loss_fn=tr.structured_loss, name="set_nll", make_model=maker(kind))
    model = load_model(kind, run["state_dict"])
    run["calibration"] = tr.calibrate_temperature([tr.energies(model, s) for s in val_set],
                                                  [s.target.numpy() for s in val_set])
    return run | {"kind": kind}


def evaluate(model, samples, tables, T: float) -> list[dict]:
    rows = []
    for s in samples:
        t = tables[s.meta["scene_id"]]
        E = tr.energies(model, s)
        top, learned = tr.exact_inference(E, s)
        row = mt.score_state(top, t["M"], t["V"], t["F"], t["K"], s.optimal) | mt.set_scores(learned, s.optimal)
        for tag, temp in (("T1", 1.0), ("cal", T)):
            d = mt.distribution_scores(E, s.states, s.optimal, len(s.cost), temp)
            row |= {f"{k}_{tag}": d[k] for k in ("nll", "kl", "mass", "brier", "mae")}
        rows.append(row | {k: s.meta[k] for k in ("scene_id", "family", "intent")} | {"floor": math.log(len(s.optimal))})
    return rows


def report(rows: list[dict]) -> dict:
    g = mt.grouped(rows, ds.FAMILIES, EXTRA + ("floor",))
    for blk in [g["overall"], *g["per_family"].values()]:
        for pop in blk.values():
            if pop.get("n"):
                pop["nll_gap_T1"] = pop["nll_T1"] - pop["floor"]
    return g


def paired(rows_a: dict, rows_b: dict, idx) -> dict:
    """Bootstrap of b - a hit, seed-averaged per scene (same scene, same training seed)."""
    d = np.mean([[float(rows_b[s][i]["hit"]) - float(rows_a[s][i]["hit"]) for i in idx] for s in tr.SEEDS], axis=0)
    return mt.paired_bootstrap(d)


def main() -> None:
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = RunEnvironment.capture().to_json() | {"device": "cpu", "threads_per_run": 1, "deterministic_algorithms": True}
    records, arrays = _load()
    used = {"train": rows_of(records, "train"), "val": rows_of(records, "val")}
    jobs = [j for s in ("train", "val") for j in used[s] if s == "val" or records[j]["intent"] == "repairable"]
    spawn, t0 = mp.get_context("spawn"), time.perf_counter()
    with ProcessPoolExecutor(12, mp_context=spawn) as pool:                       # (B) oracle audit + tables
        res = list(pool.map(_scene, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                      arrays["optimal"][j], records[j]["intent"]) for j in jobs], chunksize=8))
    if not all(r["consistent"] for r in res):
        sys.exit("STOP: oracle tables disagree with the stored Stage-2 labels")
    tables = {records[j]["scene_id"]: r for j, r in zip(jobs, res)}
    audits = [r["audit"] | {k: records[j][k] for k in ("scene_id", "family", "split")} for j, r in zip(jobs, res)
              if "audit" in r]
    t_audit = time.perf_counter() - t0
    t1 = time.perf_counter()                                                      # (C) memorisation
    with ProcessPoolExecutor(3, mp_context=spawn) as pool:
        memo_runs = {r["kind"]: r for r in pool.map(_memo, ["original", "rel_static", "rel_cf"])}
    memo = tr.structured_samples_from_dataset(records, arrays, memo_rows(records))
    memorisation = {k: {"best_epoch": r["best_epoch"], "runtime_s": r["runtime_s"],
                        "train": report(evaluate(load_model(k, r["state_dict"]), memo, tables, 1.0))["overall"]["all"]}
                    for k, r in memo_runs.items()}
    t_memo = time.perf_counter() - t1
    t2 = time.perf_counter()                                                      # (I) full train / validation
    with ProcessPoolExecutor(6, mp_context=spawn) as pool:
        runs = {(r["kind"], r["seed"]): r for r in pool.map(_full, [(k, s) for k in VARIANTS for s in tr.SEEDS])}
    t_full = time.perf_counter() - t2
    val = tr.structured_samples_from_dataset(records, arrays, used["val"])
    rows, per_seed = {}, {}
    for (kind, seed), r in runs.items():
        torch.save({"state_dict": r["state_dict"], "model_kind": kind, "seed": seed, "best_epoch": r["best_epoch"],
                    "best_val_set_nll": r["best_val_set_nll"], "calibrated_temperature": r["calibration"]["T"],
                    "architecture": rl.__doc__.strip(), "dataset_digest": ds.STAGE2_DIGEST},
                   OUT / "checkpoints" / f"{kind}_seed{seed}.pt")
        rows.setdefault(kind, {})[seed] = evaluate(load_model(kind, r["state_dict"]), val, tables, r["calibration"]["T"])
        per_seed.setdefault(kind, {})[seed] = {"best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                                               "best_val_set_nll": r["best_val_set_nll"], "calibration": r["calibration"],
                                               "train_runtime_s": r["runtime_s"]}
    for seed in tr.SEEDS:                                                         # frozen Stage-4 unary reference
        ck = torch.load(ROOT / "s4" / "checkpoints" / f"unary_seed{seed}.pt")
        if ck["dataset_digest"] != ds.STAGE2_DIGEST or ck["model_kind"] != "unary" or ck["seed"] != seed:
            sys.exit(f"STOP: frozen Stage-4 unary checkpoint seed {seed} failed its check")
        rows.setdefault("original", {})[seed] = evaluate(load_model("original", ck["state_dict"]), val, tables,
                                                         ck["calibrated_temperature"])
        per_seed.setdefault("original", {})[seed] = {"best_epoch": ck["best_epoch"], "best_val_set_nll": ck["best_val_set_nll"],
                                      "calibrated_temperature": ck["calibrated_temperature"], "source": "frozen Stage 4"}
    for kind in rows:
        for seed in tr.SEEDS:
            per_seed[kind][seed]["validation"] = report(rows[kind][seed])
    across = {k: mt.across_seeds([per_seed[k][s]["validation"] for s in tr.SEEDS]) for k in rows}
    rep = [i for i, s in enumerate(val) if s.meta["intent"] == "repairable"]
    boot = {"rel_cf_minus_rel_static": paired(rows["rel_static"], rows["rel_cf"], rep),
            "rel_static_minus_original": paired(rows["original"], rows["rel_static"], rep),
            "rel_cf_minus_original": paired(rows["original"], rows["rel_cf"], rep)}
    hit = {k: across[k]["overall"]["repairable"]["hit"]["mean"] for k in rows}
    selected = "rel_cf" if hit["rel_cf"] - hit["rel_static"] >= SELECTION_MARGIN else "rel_static"
    out = {
        "stage": "4.5", "git_sha": env["git_sha"], "git_dirty": env["git_dirty"], "dataset_digest": ds.STAGE2_DIGEST,
        "environment": env, "stage2_test_split_evaluated": False,
        "rows_evaluated_per_split": {"train": sum(records[j]["split"] == "train" for j in jobs), "val": len(used["val"]),
                                     "test": 0},
        "future_audit_split": FUTURE_AUDIT,
        "architecture": {"relational": rl.__doc__.strip(), "parameters": {k: md.n_parameters(rl.RelationalEnergy(v))
                                                                          for k, v in VARIANTS.items()}
                         | {"original_unary": md.n_parameters(md.EnergyModel(pairwise=False))}},
        "hyperparameters": {"optimizer": "AdamW", "lr": tr.LR, "weight_decay": tr.WEIGHT_DECAY, "batch": tr.BATCH_SCENES,
                            "max_epochs": tr.MAX_EPOCHS, "patience": tr.PATIENCE, "seeds": list(tr.SEEDS), "T_train": 1.0,
                            "memorisation": {"scenes": len(memo), "epochs": MEMO_EPOCHS, "seed": MEMO_SEED,
                                             "early_stopping": "none (best memo-set NLL state)"},
                            "selection_rule": f"rel_cf iff mean repairable val hit exceeds rel_static by >= {SELECTION_MARGIN}"},
        "runtime_s": {"oracle_audit_and_tables": t_audit, "memorisation_3_parallel": t_memo,
                      "full_training_6_parallel": t_full},
        "oracle_structure_audit": audit_summary(audits),
        "memorisation_200": memorisation | {"target_entropy_floor": float(np.mean([math.log(len(s.optimal)) for s in memo]))},
        "validation": {"per_seed": per_seed, "across_seeds": across},
        "paired_bootstrap_val_repairable_hit": boot,
        "selection": {"mean_repairable_val_hit": hit, "selected": selected, "margin": SELECTION_MARGIN},
        "histories": {f"{k}_seed{s}": runs[(k, s)]["history"] for k, s in runs}
        | {f"memo_{k}": r["history"] for k, r in memo_runs.items()},
    }
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1))
    a = out["oracle_structure_audit"]["overall"]
    print(json.dumps({"audit": {k: a[k] for k in ("n", "unary_sufficient", "pairwise_sufficient", "R_pair",
                                                     "n_decision_order_gt2", "frac_geo_edges_involving_shift")},
                      "memo": {k: {m: round(v["train"][m], 4) for m in ("hit", "valid_feasible", "nll_T1", "floor")}
                               for k, v in memorisation.items()},
                      "val_repairable_hit": hit, "selected": selected,
                      "boot": {k: [round(v["mean_diff"], 4), [round(c, 4) for c in v["ci95"]]] for k, v in boot.items()},
                      "runtime_s": out["runtime_s"]}, indent=1))


if __name__ == "__main__":
    main()
