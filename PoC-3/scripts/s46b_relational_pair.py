"""PoC-3 Stage 4.6B: relational pairwise diagnostic (approved diagnostic exception; validation only).

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s46b_relational_pair.py

REL-CF + learned symmetric Q (relational.PAIR_ARCHITECTURE): P1 REL-CF-PAIR-ALL, P2 REL-CF-PAIR-SHIFT (Q kept only
for pairs containing a SHIFT_TARGET; identical parameters). Frozen Stage-4 objective, exact admissible domain and
inference, Stage-3/4 protocol. Order: integrity of the frozen REL-CF-UNARY reference -> Stage-4.6A categories must
reproduce 288 / 27 / 45 -> 20-scene P2 implementation gate (STOP if < 19 / 20) -> 6 trainings (train split,
validation-NLL selection, validation calibration) -> validation-repairable evaluation. Zero test rows; seed 61 is
not generated. Writes PoC-3/out/s46b/metrics.json and checkpoints/rel_pair_{all,shift}_seed{7,17,27}.pt.

FROZEN RULES (pre-registered before any aggregate result; deltas vs frozen REL-CF-UNARY, seed-averaged per scene)
  Delta_P = hit(P-TOP1) difference, Delta_U = hit(U-EXACT) difference, Delta_all = all-repairable difference.
  Selection: discard a variant with Delta_U < -0.02; among the rest pick the higher mean P-TOP1 hit if the
  difference is >= 0.01, otherwise REL-CF-PAIR-SHIFT (simpler, physically structured); none if both discarded.
  RELATIONAL PAIRWISE HELPS P-TOP1 iff the selected variant has Delta_P >= 0.10 and paired 95 % CI lower bound > 0.
  Unary strength preserved iff Delta_U >= -0.02. Overall improvement iff Delta_all >= 0.03 and its CI excludes 0.
  UNARY FEASIBILITY BOTTLENECK REMAINS iff the selected (else every) variant has U-EXACT hit < 0.85.
"""

import functools
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s46a_error_audit as au  # noqa: E402  (frozen Stage-4.6A categories, integrity, scoring, statistics)

from poc3 import RunEnvironment  # noqa: E402
from poc3 import dataset as ds  # noqa: E402
from poc3 import relational as rl  # noqa: E402
from poc3 import train as tr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s46b"
VARIANTS = {"REL-CF-PAIR-ALL": "all", "REL-CF-PAIR-SHIFT": "shift"}
REFERENCE = "REL-CF-UNARY"
EXPECTED_COUNTS = {"U-EXACT": 288, "U-OVERLAP": 27, "P-TOP1": 45}
HYPER = {"optimizer": "AdamW", "lr": tr.LR, "weight_decay": tr.WEIGHT_DECAY, "batch": tr.BATCH_SCENES,
         "max_epochs": tr.MAX_EPOCHS, "patience": tr.PATIENCE, "seeds": list(tr.SEEDS), "T_train": 1.0,
         "loss": "exact Stage-4 set NLL over the admissible domain", "selection": "validation set NLL"}
ARCHITECTURE = rl.__doc__.strip() + "\n\n" + rl.PAIR_ARCHITECTURE


def train_guard(records, rows) -> list[int]:
    """Training / selection rows: train or val only; any test row raises (same guard intent as Stage 4.6A)."""
    for j in rows:
        if records[j]["split"] == "test":
            raise RuntimeError("Stage 4.6B never evaluates the Stage-2 test split")
    return list(rows)


def make(pair: str):
    return functools.partial(rl.RelationalEnergy, "cf", pair)


def gate(samples, seed: int = 7, epochs: int = tr.GATE_EPOCHS) -> dict:
    """20-scene P2 implementation gate: fit and exact-evaluate the same scenes; pass iff >= 19 / 20 hits."""
    run = tr.train(samples, samples, seed, max_epochs=epochs, patience=epochs, batch=tr.GATE_BATCH,
                   pairwise=True, loss_fn=tr.structured_loss, name="set_nll", make_model=make("shift"))
    model = rl.load(run["state_dict"], "cf", "shift")
    hits = sum(tr.exact_inference(tr.energies(model, s), s)[0] in s.optimal for s in samples)
    return {"hits": int(hits), "n": len(samples), "passed": hits * 20 >= tr.GATE_MIN_HITS * len(samples),
            "best_epoch": run["best_epoch"], "runtime_s": run["runtime_s"]}


def _train(job) -> dict:
    name, seed = job
    records, arrays = ds.load(ROOT / "s2")
    train_set = tr.structured_samples_from_dataset(records, arrays, train_guard(records, tr.split_rows(records, "train")))
    val_set = tr.structured_samples_from_dataset(records, arrays, train_guard(records, tr.split_rows(records, "val")))
    run = tr.train(train_set, val_set, seed, pairwise=True, loss_fn=tr.structured_loss, name="set_nll",
                   make_model=make(VARIANTS[name]))
    model = rl.load(run["state_dict"], "cf", VARIANTS[name])
    run["calibration"] = tr.calibrate_temperature([tr.energies(model, s) for s in val_set],
                                                  [s.target.numpy() for s in val_set])
    return run | {"name": name}


def checkpoint_payload(run: dict) -> dict:
    return {"state_dict": run["state_dict"], "variant": run["name"], "pair": VARIANTS[run["name"]], "seed": run["seed"],
            "dataset_digest": ds.STAGE2_DIGEST, "best_val_set_nll": run["best_val_set_nll"],
            "calibrated_temperature": run["calibration"]["T"], "architecture": ARCHITECTURE, "hyperparameters": HYPER}


def select(deltas: dict, p_hits: dict) -> str | None:
    keep = [n for n in VARIANTS if deltas[n]["U-EXACT"] >= -0.02]
    if not keep:
        return None
    if len(keep) == 1:
        return keep[0]
    a, b = (p_hits[n] for n in keep)
    if abs(a - b) >= 0.01:
        return keep[0] if a > b else keep[1]
    return "REL-CF-PAIR-SHIFT"


def rules(dP: float, ci_P: list, dU: float, d_all: float, ci_all: list) -> dict:
    return {"RELATIONAL_PAIRWISE_HELPS_P_TOP1": dP >= 0.10 and ci_P[0] > 0, "unary_strength_preserved": dU >= -0.02,
            "overall_improvement": d_all >= 0.03 and (ci_all[0] > 0 or ci_all[1] < 0)}


def main() -> None:
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = RunEnvironment.capture().to_json()
    records, arrays = ds.load(ROOT / "s2")
    rows = au.guard(records, [j for j in tr.split_rows(records, "val") if records[j]["intent"] == "repairable"])
    ref = {}
    for seed in tr.SEEDS:                                                    # frozen reference integrity first
        path = ROOT / "s45" / "checkpoints" / f"rel_cf_seed{seed}.pt"
        if not path.exists():
            sys.exit(f"STOP: missing frozen checkpoint {path}")
        ck = torch.load(path)
        ref[seed] = (au.check_checkpoint(ck, "rel_cf", seed), ck["calibrated_temperature"])
    spawn = mp.get_context("spawn")
    with ProcessPoolExecutor(12, mp_context=spawn) as pool:
        scenes = list(pool.map(au._scene, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                            arrays["optimal"][j]) for j in rows], chunksize=4))
    cats = [o["category"] for o in scenes]
    counts = {c: cats.count(c) for c in au.CATEGORIES}
    if counts != EXPECTED_COUNTS or not all(o["consistent"] and o["S2_exact"] for o in scenes):
        sys.exit(f"STOP: Stage-4.6A categories not reproduced: {counts}")
    g = gate(tr.structured_samples_from_dataset(records, arrays, train_guard(records, tr.gate_rows(records))))
    print(json.dumps({"gate": g}), flush=True)
    base = {"stage": "4.6B", "git_sha": env["git_sha"], "git_dirty": env["git_dirty"], "dataset_digest": ds.STAGE2_DIGEST,
            "environment": env, "stage2_test_split_evaluated": False, "future_audit_seed_61_generated": False,
            "rows_evaluated_per_split": {"val": len(rows), "test": 0}, "category_counts": counts,
            "architecture": ARCHITECTURE, "hyperparameters": HYPER, "frozen_rules": __doc__.split("FROZEN RULES")[1].strip(),
            "implementation_gate": g}
    if not g["passed"]:
        (OUT / "metrics.json").write_text(json.dumps(base | {"stopped": "implementation gate failed"}, indent=1))
        sys.exit("STOP: 20-scene REL-CF-PAIR-SHIFT gate failed")
    with ProcessPoolExecutor(6, mp_context=spawn) as pool:
        runs = {(r["name"], r["seed"]): r for r in pool.map(_train, [(n, s) for n in VARIANTS for s in tr.SEEDS])}
    for (name, seed), r in runs.items():
        torch.save(checkpoint_payload(r), OUT / "checkpoints" / f"rel_pair_{VARIANTS[name]}_seed{seed}.pt")
    samples = tr.structured_samples_from_dataset(records, arrays, rows)
    res = {REFERENCE: {s: au.score(ref[s][0], samples, scenes, ref[s][1]) for s in tr.SEEDS}}
    res |= {n: {s: au.score(rl.load(runs[(n, s)]["state_dict"], "cf", VARIANTS[n]), samples, scenes,
                            runs[(n, s)]["calibration"]["T"]) for s in tr.SEEDS} for n in VARIANTS}
    N, fam = len(rows), [records[j]["family"] for j in rows]
    idx = {c: [i for i in range(N) if cats[i] == c] for c in au.CATEGORIES} | {"ALL": list(range(N))}
    idx |= {f"SHIFT={v}": [i for i in range(N) if scenes[i]["shift"] == v] for v in (True, False)}
    idx |= {f"K*={k}": [i for i in range(N) if scenes[i]["K_star"] == k] for k in (1, 2, 3, 4)}
    idx |= {f"K*=2 {c}": [i for i in idx[c] if scenes[i]["K_star"] == 2] for c in ("U-EXACT", "P-TOP1")}
    idx |= {f"family={f}": [i for i in range(N) if fam[i] == f] for f in ds.FAMILIES}
    strata = {m: {k: au.summary(res[m], v) for k, v in idx.items() if v} for m in res}
    boot_keys = ("ALL", "U-EXACT", "U-OVERLAP", "P-TOP1", "K*=2 U-EXACT", "K*=2 P-TOP1")
    boots = {f"{n} - {REFERENCE}": {k: au.paired(res[n], res[REFERENCE], idx[k]) for k in boot_keys} for n in VARIANTS}
    boots["REL-CF-PAIR-SHIFT - REL-CF-PAIR-ALL"] = {k: au.paired(res["REL-CF-PAIR-SHIFT"], res["REL-CF-PAIR-ALL"], idx[k])
                                                   for k in boot_keys}
    hit = lambda m, k: au.mean_of(strata[m][k], "hit")  # noqa: E731
    deltas = {n: {k: hit(n, k) - hit(REFERENCE, k) for k in ("ALL", "U-EXACT", "U-OVERLAP", "P-TOP1")} for n in VARIANTS}
    per_variant_rules = {n: rules(deltas[n]["P-TOP1"], boots[f"{n} - {REFERENCE}"]["P-TOP1"]["ci95"], deltas[n]["U-EXACT"],
                                  deltas[n]["ALL"], boots[f"{n} - {REFERENCE}"]["ALL"]["ci95"]) for n in VARIANTS}
    selected = select(deltas, {n: hit(n, "P-TOP1") for n in VARIANTS})
    remaining = {}
    for n in VARIANTS:
        conc = au.error_concentration(res[n], cats)
        H, h_P = hit(n, "ALL"), hit(n, "P-TOP1")
        remaining[n] = conc | {"hit_if_all_P_TOP1_errors_fixed": au.perfect_pair(H, counts["P-TOP1"] / N, h_P),
                               "U_EXACT_hit": hit(n, "U-EXACT"), "U_EXACT_below_0.85": hit(n, "U-EXACT") < 0.85}
    bottleneck = (remaining[selected]["U_EXACT_below_0.85"] if selected
                  else all(r["U_EXACT_below_0.85"] for r in remaining.values()))
    out = base | {
        "reference": {"model": REFERENCE, "checkpoints": "PoC-3/out/s45/checkpoints/rel_cf_seed{7,17,27}.pt (frozen)"},
        "training": {f"{n}_seed{s}": {k: runs[(n, s)][k] for k in ("best_epoch", "epochs_run", "best_val_set_nll",
                                                                     "calibration", "runtime_s")} for n, s in runs},
        "histories": {f"{n}_seed{s}": runs[(n, s)]["history"] for n, s in runs},
        "per_model_per_stratum": strata, "calibration_note": "nll_cal / mass_cal: in-sample validation calibration",
        "deltas_vs_reference": deltas, "paired_bootstraps": boots, "fixed_rule_results": per_variant_rules,
        "selected_pairwise_variant": selected,
        "selected_rule_results": per_variant_rules[selected] if selected else None,
        "remaining_error_concentration": remaining, "unary_feasibility_bottleneck_remains": bottleneck,
    }
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"deltas": deltas, "rules": per_variant_rules, "selected": selected,
                      "unary_feasibility_bottleneck_remains": bottleneck}, indent=1, default=float))


if __name__ == "__main__":
    main()
