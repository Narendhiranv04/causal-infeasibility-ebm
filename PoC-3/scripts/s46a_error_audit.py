"""PoC-3 Stage 4.6A: zero-training error localization (analysis only; approved file-table exception).

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s46a_error_audit.py

Frozen checkpoints only (Stage-4 unary / pairwise, Stage-4.5 REL-CF; seeds 7, 17, 27) evaluated with the
unchanged Stage-4 exact admissible inference on the 360 VALIDATION repairable scenes. No training, no
optimizer, no checkpoint is written; a guard rejects any split == "test" row. Writes PoC-3/out/s46a/metrics.json.

CATEGORIES (frozen oracle, base resolution, exact M, V, K; S1 / S2 = order-1 / order-2 decision sets)
  U-EXACT   S1 == S*             U-OVERLAP   S1 != S* and S1 & S* != {}             P-TOP1   S1 & S* == {}
  R_pair_set = P(S1 != S*) (complete tied-set recovery); R_pair_top1 = P(S1 & S* == {}) (no order-1 optimum).
  x1 = min S1 (lowest state index; undefined -> miss when the order-1 surrogate admits no feasible state).
PRE-REGISTERED DECISION RULES (fixed before any Stage-4.6A result was computed; REL-CF unless stated)
  A1 R_pair_top1 >= 0.10
  A2 hit(U-EXACT) - hit(P-TOP1) >= 0.10 (across-seed means)
  A3 ORIGINAL-PAIRWISE - ORIGINAL-UNARY hit on P-TOP1 has paired-bootstrap 95 % CI lower bound > 0,
     or error enrichment P(err | P-TOP1) / P(err | not P-TOP1) >= 1.5 (pooled over seeds)
  A4 some oracle repair-size stratum with n >= 20 in both P-TOP1 and U-EXACT keeps a deficit >= 0.10
  A5 perfect-pair upper bound raises repairable hit by >= 0.05
  Outcome A iff A1-A5. Otherwise, in order: B if R_pair_top1 < 0.10; C (mixed) if the error rate is >= 0.10
  in both U-EXACT and P-TOP1; B if > 50 % of errors lie in U-EXACT or U-OVERLAP, or A2 holds but A4 fails;
  else unclassified. Calibrated NLL / P(S*) are in-sample calibration diagnostics and drive nothing.
"""

import json
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from poc.types import InterventionKind
from poc2 import dataset as pd
from poc2 import structure as st
from poc3 import RunEnvironment
from poc3 import dataset as ds
from poc3 import metrics as mt
from poc3 import model as md
from poc3 import relational as rl
from poc3 import train as tr

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s46a"
CATEGORIES = ("U-EXACT", "U-OVERLAP", "P-TOP1")
MODELS = {"ORIGINAL-UNARY": ("s4", "unary"), "ORIGINAL-PAIRWISE": ("s4", "pairwise"), "REL-CF-UNARY": ("s45", "rel_cf")}
EXTRA = ("nll_T1", "nll_cal", "mass_cal")
MIN_STRATUM, LOW_SAMPLE = 20, 30


def guard(records, rows) -> list[int]:
    """Only validation repairable rows may enter; any test row raises."""
    for j in rows:
        if records[j]["split"] == "test":
            raise RuntimeError("Stage 4.6A never evaluates the Stage-2 test split")
        if records[j]["split"] != "val" or records[j]["intent"] != "repairable":
            raise RuntimeError("Stage 4.6A uses validation repairable scenes only")
    return list(rows)


def category(S1, S_star) -> str:
    if set(S1) == set(S_star):
        return "U-EXACT"
    return "U-OVERLAP" if set(S1) & set(S_star) else "P-TOP1"


def oracle_unary_top1(S1):
    return min(S1) if S1 else None


def check_checkpoint(ck: dict, kind: str, seed: int):
    """Integrity: kind, seed, Stage-2 digest, architecture and exact state-dict keys / shapes -> loaded model."""
    if ck.get("model_kind") != kind or ck.get("seed") != seed or ck.get("dataset_digest") != ds.STAGE2_DIGEST:
        raise ValueError(f"checkpoint {kind} seed {seed} failed kind / seed / digest check")
    if kind == "rel_cf":
        if ck.get("architecture") != rl.__doc__.strip():
            raise ValueError("REL-CF architecture description differs from the frozen relational module")
        model = rl.RelationalEnergy("cf")
    else:
        model = md.EnergyModel(pairwise=kind == "pairwise")
    ref = model.state_dict()
    sd = ck["state_dict"]
    if set(sd) != set(ref) or any(sd[k].shape != ref[k].shape for k in ref):
        raise ValueError(f"checkpoint {kind} seed {seed}: missing / unexpected / mis-shaped state-dict keys")
    model.load_state_dict(sd, strict=True)
    return model.eval()


def _scene(args) -> dict:
    spec_json, P, adm, opt = args
    lab = ds.oracle(pd.spec_from_json(spec_json))
    T, S = lab["table"], lab["S_star_x"]
    a = st.compatible_mobius(T.G, T.M)
    S1, S2 = st.order_decision(T, st.reconstruct(a, 1)), st.order_decision(T, st.reconstruct(a, 2))
    _, _, options = ds.build(pd.spec_from_json(spec_json))
    x1 = oracle_unary_top1(S1)
    return {"M": T.M, "V": T.V, "F": T.F, "K": T.K, "S1": sorted(S1), "S2_exact": S2 == S, "x1_hit": x1 in S,
            "category": category(S1, S), "shift": any(o.intervention.kind is InterventionKind.SHIFT_TARGET for o in options),
            "K_star": int(T.K[min(S)]), "S1_empty": not S1,
            "consistent": lab["admissible_x"] == ds.unpack_states(adm, P) and S == ds.unpack_states(opt, P)}


def score(model, samples, scenes, T_cal: float) -> list[dict]:
    rows = []
    for s, o in zip(samples, scenes):
        E = tr.energies(model, s)
        top, _ = tr.exact_inference(E, s)
        d1 = mt.distribution_scores(E, s.states, s.optimal, len(s.cost), 1.0)
        dc = mt.distribution_scores(E, s.states, s.optimal, len(s.cost), T_cal)
        rows.append(mt.score_state(top, o["M"], o["V"], o["F"], o["K"], s.optimal)
                    | {"nll_T1": d1["nll"], "nll_cal": dc["nll"], "mass_cal": dc["mass"], "x_hat": top,
                       "family": s.meta["family"], "intent": s.meta["intent"]})
    return rows


def summary(by_seed: dict, idx) -> dict:
    return mt.across_seeds([mt.summarize([by_seed[s][i] for i in idx], EXTRA) for s in tr.SEEDS])


def mean_of(block: dict, metric: str, empty: float = math.nan) -> float:
    """Across-seed mean of a metric, or `empty` for a population without scenes."""
    return block[metric]["mean"] if metric in block else empty


def error_concentration(by_seed: dict, cats: list[str]) -> dict:
    """P(error | C) pooled over seeds, P(C | error), and enrichment of P-TOP1 errors (inf if denominator 0)."""
    err = np.array([[not by_seed[s][i]["hit"] for i in range(len(cats))] for s in tr.SEEDS])
    cats = np.array(cats)
    given = {c: float(err[:, cats == c].mean()) if (cats == c).any() else math.nan for c in CATEGORIES}
    share = {c: float(err[:, cats == c].sum() / err.sum()) if err.sum() else math.nan for c in CATEGORIES}
    p, rest = err[:, cats == "P-TOP1"], err[:, cats != "P-TOP1"]
    num, den = (p.mean() if p.size else math.nan), (rest.mean() if rest.size else math.nan)
    return {"P_error_given_C": given, "P_C_given_error": share, "n_errors_pooled": int(err.sum()),
            "enrichment_P_TOP1": math.inf if den == 0 else float(num / den)}


def perfect_pair(H: float, p_P: float, h_P: float) -> float:
    """Diagnostic upper bound if all top-1 pair-dependent scenes were solved: H + p_P (1 - h_P)."""
    return H + p_P * (1.0 - h_P)


def paired(rows_b: dict, rows_a: dict, idx) -> dict:
    d = np.mean([[float(rows_b[s][i]["hit"]) - float(rows_a[s][i]["hit"]) for i in idx] for s in tr.SEEDS], axis=0)
    return mt.paired_bootstrap(d) | {"low_sample_exploratory": len(idx) < LOW_SAMPLE}


def decide(r: dict) -> dict:
    a = {"A1": r["R_top1"] >= 0.10, "A2": r["deficit"] >= 0.10, "A3": r["pair_ci_lo"] > 0 or r["enrichment"] >= 1.5,
         "A4": r["stratified_deficit"], "A5": r["perfect_gain"] >= 0.05}
    mixed = r["err_U_EXACT"] >= 0.10 and r["err_P_TOP1"] >= 0.10
    b_unary = r["share_unary_errors"] > 0.5 or (a["A2"] and not a["A4"])
    if all(a.values()):
        outcome = "A"
    elif r["R_top1"] < 0.10:
        outcome = "B"
    elif mixed:
        outcome = "C"
    elif b_unary:
        outcome = "B"
    else:
        outcome = "unclassified"
    return {"criteria_A": a, "mixed_condition": mixed, "B_unary_errors_or_vanishing_deficit": b_unary, "outcome": outcome}


NEXT = {"A": "relational pairwise diagnostic (do not build without approval)",
        "B": "unary feasibility learning / supervision / missing raw geometry diagnostic; do not add Q yet",
        "C": "mixed bottleneck: target both unary feasibility and pair-relevant scenes, sized by their error shares",
        "unclassified": "evidence ambiguous; no recommendation forced"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = RunEnvironment.capture().to_json()
    records, arrays = ds.load(ROOT / "s2")
    rows = guard(records, [j for j in tr.split_rows(records, "val") if records[j]["intent"] == "repairable"])
    models = {}
    for name, (stage, kind) in MODELS.items():                      # integrity first: STOP before any analysis
        for seed in tr.SEEDS:
            path = ROOT / stage / "checkpoints" / f"{kind}_seed{seed}.pt"
            if not path.exists():
                sys.exit(f"STOP: missing frozen checkpoint {path}")
            ck = torch.load(path)
            models[(name, seed)] = (check_checkpoint(ck, kind, seed), ck["calibrated_temperature"])
    with ProcessPoolExecutor(12, mp_context=mp.get_context("spawn")) as pool:
        scenes = list(pool.map(_scene, [(records[j]["spec"], int(arrays["P"][j]), arrays["admissible"][j],
                                         arrays["optimal"][j]) for j in rows], chunksize=4))
    if not all(o["consistent"] for o in scenes):
        sys.exit("STOP: oracle tables disagree with the stored Stage-2 labels")
    if not all(o["S2_exact"] for o in scenes):
        sys.exit("STOP: a validation repairable scene needs decision order > 2")
    samples = tr.structured_samples_from_dataset(records, arrays, rows)
    res = {n: {s: score(models[(n, s)][0], samples, scenes, models[(n, s)][1]) for s in tr.SEEDS} for n in MODELS}
    cats = [o["category"] for o in scenes]
    N, sel = len(rows), lambda f: [i for i in range(len(rows)) if f(i)]
    fam = [records[j]["family"] for j in rows]
    by_cat = {c: sel(lambda i, c=c: cats[i] == c) for c in CATEGORIES} | {"ALL": list(range(N))}
    counts = {c: len(by_cat[c]) for c in CATEGORIES}
    R_set, R_top1 = 1 - counts["U-EXACT"] / N, counts["P-TOP1"] / N
    per_model = {n: {c: summary(res[n], idx) for c, idx in by_cat.items()} for n in MODELS}
    conc = {n: error_concentration(res[n], cats) for n in MODELS}
    upper = {}
    for n in MODELS:
        H, h = mean_of(per_model[n]["ALL"], "hit"), mean_of(per_model[n]["P-TOP1"], "hit", 1.0)  # empty: nothing to fix
        V, v = mean_of(per_model[n]["ALL"], "valid_feasible"), mean_of(per_model[n]["P-TOP1"], "valid_feasible", 1.0)
        hp, vp = perfect_pair(H, R_top1, h), perfect_pair(V, R_top1, v)
        upper[n] = {"label": "diagnostic upper bound if all top-1 pair-dependent scenes were solved", "H": H,
                    "H_perfect_pair": hp, "VF": V, "VF_perfect_pair": vp, "reaches_0.90_hit": hp >= 0.90,
                    "reaches_0.99_vf": vp >= 0.99}
    shift = {str(v): sel(lambda i, v=v: scenes[i]["shift"] == v) for v in (True, False)}
    sizes = sorted({o["K_star"] for o in scenes})
    strata, stratified = {}, False
    for k in sizes:
        p_idx = [i for i in by_cat["P-TOP1"] if scenes[i]["K_star"] == k]
        u_idx = [i for i in by_cat["U-EXACT"] if scenes[i]["K_star"] == k]
        pu_idx = [i for i in range(N) if cats[i] != "U-EXACT" and scenes[i]["K_star"] == k]
        row = {"n_P_TOP1": len(p_idx), "n_U_EXACT": len(u_idx), "n_pair_relevant_set": len(pu_idx)}
        if len(p_idx) >= MIN_STRATUM and len(u_idx) >= MIN_STRATUM:
            hp_, hu = summary(res["REL-CF-UNARY"], p_idx)["hit"]["mean"], summary(res["REL-CF-UNARY"], u_idx)["hit"]["mean"]
            row |= {"REL_CF_hit_P_TOP1": hp_, "REL_CF_hit_U_EXACT": hu, "deficit": hu - hp_}
            stratified = stratified or hu - hp_ >= 0.10
        if len(pu_idx) >= MIN_STRATUM and len(u_idx) >= MIN_STRATUM:
            row["REL_CF_hit_pair_relevant_set"] = summary(res["REL-CF-UNARY"], pu_idx)["hit"]["mean"]
        strata[str(k)] = row
    boots = {c: {"REL-CF - ORIGINAL-UNARY": paired(res["REL-CF-UNARY"], res["ORIGINAL-UNARY"], idx),
                 "ORIGINAL-PAIRWISE - ORIGINAL-UNARY": paired(res["ORIGINAL-PAIRWISE"], res["ORIGINAL-UNARY"], idx),
                 "REL-CF - ORIGINAL-PAIRWISE": paired(res["REL-CF-UNARY"], res["ORIGINAL-PAIRWISE"], idx)}
             for c, idx in by_cat.items() if idx}
    rc = per_model["REL-CF-UNARY"]
    rule_inputs = {"R_top1": R_top1, "deficit": mean_of(rc["U-EXACT"], "hit") - mean_of(rc["P-TOP1"], "hit"),
                   "pair_ci_lo": boots["P-TOP1"]["ORIGINAL-PAIRWISE - ORIGINAL-UNARY"]["ci95"][0] if "P-TOP1" in boots
                   else -math.inf, "enrichment": conc["REL-CF-UNARY"]["enrichment_P_TOP1"],
                   "stratified_deficit": stratified,
                   "perfect_gain": upper["REL-CF-UNARY"]["H_perfect_pair"] - upper["REL-CF-UNARY"]["H"],
                   "err_U_EXACT": conc["REL-CF-UNARY"]["P_error_given_C"]["U-EXACT"],
                   "err_P_TOP1": conc["REL-CF-UNARY"]["P_error_given_C"]["P-TOP1"],
                   "share_unary_errors": conc["REL-CF-UNARY"]["P_C_given_error"]["U-EXACT"]
                   + conc["REL-CF-UNARY"]["P_C_given_error"]["U-OVERLAP"]}
    verdict = decide(rule_inputs)
    out = {
        "stage": "4.6A", "git_sha": env["git_sha"], "git_dirty": env["git_dirty"], "dataset_digest": ds.STAGE2_DIGEST,
        "environment": env, "stage2_test_split_evaluated": False, "future_audit_seed_61_generated": False,
        "n_validation_repairable": N, "rows_evaluated_per_split": {"val": N, "train": 0, "test": 0},
        "category_definitions": __doc__.split("CATEGORIES")[1].split("PRE-REGISTERED")[0].strip(),
        "decision_rules": __doc__.split("PRE-REGISTERED DECISION RULES")[1].strip(),
        "category_counts": counts, "R_pair_set": R_set, "R_pair_top1": R_top1,
        "S1_empty_count": sum(o["S1_empty"] for o in scenes),
        "deterministic_oracle_unary_top1": {"x1_hit_rate": float(np.mean([o["x1_hit"] for o in scenes])),
                                            "S1_overlap_rate": 1 - R_top1},
        "shift_contingency": {c: {s: sum(scenes[i]["shift"] == (s == "True") for i in by_cat[c]) for s in shift}
                              for c in CATEGORIES},
        "per_model_by_shift": {n: {s: summary(res[n], idx) for s, idx in shift.items()} for n in MODELS},
        "repair_size_contingency": {c: {str(k): sum(scenes[i]["K_star"] == k for i in by_cat[c]) for k in sizes}
                                    for c in CATEGORIES},
        "per_model_by_repair_size": {n: {str(k): {m: summary(res[n], sel(lambda i, k=k: scenes[i]["K_star"] == k))[m]
                                                  for m in ("n", "hit", "valid_feasible")} for k in sizes} for n in MODELS},
        "repair_size_stratified_REL_CF": strata,
        "per_model_per_category": per_model, "error_concentration": conc, "perfect_pair_upper_bounds": upper,
        "per_family": {f: {"n": fam.count(f), "category_counts": {c: sum(fam[i] == f for i in by_cat[c]) for c in CATEGORIES},
                           "R_pair_set": 1 - sum(fam[i] == f for i in by_cat["U-EXACT"]) / fam.count(f),
                           "R_pair_top1": sum(fam[i] == f for i in by_cat["P-TOP1"]) / fam.count(f),
                           "hit_by_category": {n: {c: summary(res[n], [i for i in by_cat[c] if fam[i] == f]).get("hit")
                                                   for c in CATEGORIES} for n in ("REL-CF-UNARY", "ORIGINAL-PAIRWISE")}}
                       for f in ds.FAMILIES},
        "paired_bootstraps": boots,
        "interpretation_rule_results": {"inputs": rule_inputs} | verdict,
        "recommended_next_diagnostic": NEXT[verdict["outcome"]],
        "calibration_note": "nll_cal / mass_cal use each checkpoint's validation-calibrated T: in-sample calibration "
                            "diagnostics that drive no Stage-4.6A decision",
    }
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("category_counts", "R_pair_set", "R_pair_top1", "S1_empty_count",
                                          "deterministic_oracle_unary_top1", "interpretation_rule_results",
                                          "recommended_next_diagnostic")}, indent=1, default=float))


if __name__ == "__main__":
    main()
