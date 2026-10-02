"""PoC-2 Stage 6: structured-optimization evaluation and final verdict (plan2.md sections 22, 23).

Run from the repository root (after Stage 4 wrote PoC-2/out/s4/scenes.jsonl):

    PYTHONPATH=PoC-1/src:PoC-2/src python PoC-2/scripts/s6_validate.py

Writes PoC-2/out/s6/verdict.json, solver_accuracy.png, overview.png. Fixed, digest-verified 100-scene
dataset only. Solvers: unary structured selection, branch-on-global-macro, exact pairwise QUBO
enumeration, PoC-1 Hopfield (unchanged) with R in {1, 4, 16}. Rules: poc2/optimize.py docstring.
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from poc import energy as en
from poc.envelope import ENVELOPE_STEP
from poc.oracle import CONTACT_TOL_3D
from poc.types import InterventionKind
from poc2 import dataset as ds
from poc2 import optimize as op
from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2 import structure as st

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s6"
FINE = ENVELOPE_STEP / 2
FAMILIES = ("make_space", "storage_insertion", "storage_extraction", "articulated_opening")
SOLVERS = ("unary", "branch", "qubo", "hopfield_R1", "hopfield_R4", "hopfield_R16")


def build(spec):
    return tasks.build(spec) if isinstance(spec, tasks.TaskSpec) else scenes.build(spec)


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


def judge(T, chosen: frozenset[int], exact: frozenset[int]) -> dict:
    ok = all(T.M[x] == 1 and T.V[x] == 1 and T.F[x] == 0 for x in chosen) and bool(chosen)
    k_star = min(int(T.K[x]) for x in exact)
    return {"exact_match": chosen == exact, "valid_feasible": ok,
            "minimal": ok and all(int(T.K[x]) == k_star for x in chosen)}


def analyse(record: dict, idx: int) -> dict:
    spec = ds.spec_from_json(record["spec"])
    scene, _, options = build(spec)
    T, Tf = orc.repair_table(scene, options), orc.repair_table(scene, options, FINE)
    exact = st.exact_decision(T)
    (venc, qb), t_build = timed(lambda: (lambda v: (v, op.build_qubo(T, options, v)))(op.validity_encoding(scene, options)))
    chk = op.constraint_check(T, qb)
    H, t_qubo = timed(lambda: en.qubo_energy(qb["Q"], qb["q"], qb["c"]))
    qubo_opt, t_arg = timed(lambda: en.argmin_sets(H))
    a_b, a_f = st.compatible_mobius(T.G, T.M), st.compatible_mobius(Tf.G, Tf.M)
    unary, t_unary = timed(lambda: st.order_decision(T, st.reconstruct(a_b, 1)))
    pair_dec = st.order_decision(T, st.reconstruct(a_b, 2))
    branch, t_branch = timed(lambda: op.branch_on_macro(T, options))
    masks, energies, t_restart = op.hopfield_runs(qb, [ds.SEED, idx])
    edges = st.interaction_edges(T, a_b, st.significant(a_b, a_f))
    shifts = [p for p, o in enumerate(options) if o.intervention.kind is InterventionKind.SHIFT_TARGET]
    solvers = {"unary": judge(T, unary, exact), "branch": judge(T, branch["decision"], exact),
               "qubo": judge(T, qubo_opt, exact)}
    hop = {}
    for R in op.BUDGETS:
        j = int(np.argmin(energies[:R]))
        m = masks[j]
        solvers[f"hopfield_R{R}"] = judge(T, frozenset({m}), exact) | {"exact_match": m in exact}
        hop[R] = {"in_qubo_optimum": m in qubo_opt, "energy_gap": float(H[m] - H.min())}
    blockers = set(orc.causes(T.original).blockers)
    trivial = all(S.bit_count() == len(blockers) and {options[p].intervention.entity_id for p in range(len(options)) if S >> p & 1}
                  == blockers and not any(S >> p & 1 for p in shifts) for S in exact)
    levels = op.validity_levels(T)
    g = T.G[T.M == 1]
    return {
        "scene_id": record["scene_id"], "family": record["family"], "P": len(options),
        "repositioning_offered": bool(shifts), "unary_failure": unary != exact, "pairwise_correct": pair_dec == exact,
        "constraint": chk, "min_positive_G": float(g[g > 0].min(initial=np.inf)),
        "bounds": {k: qb[k] for k in ("B", "kappa", "lambda_V", "lambda_M", "L_G", "W")},
        "solvers": solvers, "hopfield": hop, "branch_work": [branch["subsets_examined"], branch["of_2P"]],
        "branch_conditional_unary_exact": branch["conditional_unary_exact"],
        "runtime_s": {"unary": t_unary, "branch": t_branch, "qubo": t_qubo + t_arg, "qubo_build": t_build,
                      **{f"hopfield_R{R}": t_restart * R for R in op.BUDGETS}},
        "geometric_density": st.density(len(edges["geometric"]), len(options)),
        "n_physical_edges": sum(k == st.PHYSICAL for *_, k in edges["geometric"]),
        "mechanisms": {"envelope_changing": any(k == st.PHYSICAL for p, q, _, k in edges["geometric"]
                                                if any(x >> p & 1 and x >> q & 1 for x in exact)),
                       "substitutable": any(k == st.SUBSTITUTABLE for p, q, _, k in edges["geometric"]
                                            if any(x >> p & 1 or x >> q & 1 for x in exact)),
                       "static_compatibility": levels["binding"]},
        "validity": levels, "trivial_repair": trivial and not levels["binding"],
        "stable": {"C_star": orc.causes(Tf.original) == orc.causes(T.original),
                   "S_star": st.exact_decision(Tf) == exact},
        "nontrivial_causal_sets": orc.causes(T.original).minimal_causes not in (frozenset(), frozenset({frozenset(blockers)})),
    }


def rates(rows) -> dict:
    return {s: {k: float(np.mean([r["solvers"][s][k] for r in rows])) for k in ("exact_match", "valid_feasible", "minimal")}
            for s in SOLVERS}


def by(rows, key, buckets) -> dict:
    return {name: {"n": len(sel), **{s: float(np.mean([r["solvers"][s]["exact_match"] for r in sel])) for s in SOLVERS}}
            for name, f in buckets.items() if (sel := [r for r in rows if f(r[key])])}


def plots(rows, overall) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, colors = "#fcfcfb", ("#2a78d6", "#1baf7a", "#eb6834", "#c3c2b7", "#9a9890", "#52514e")
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=surface)
    x = np.arange(len(SOLVERS))
    for i, (key, label) in enumerate((("exact_match", "exact S* (all ties)"), ("valid_feasible", "valid + feasible"),
                                      ("minimal", "minimal cost"))):
        ax.bar(x + (i - 1) * 0.26, [overall[s][key] for s in SOLVERS], 0.25, color=colors[i], label=label)
    ax.set_xticks(x, SOLVERS, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of 75 repairable scenes [-]")
    ax.set_title("Solver / decision-rule accuracy against the exact oracle repair", loc="left", fontsize=10)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "solver_accuracy.png", dpi=200, facecolor=surface)
    plt.close(fig)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=surface)
    for s, color in zip(("unary", "branch", "qubo", "hopfield_R16"), (colors[0], colors[1], colors[2], colors[5])):
        P = np.array([r["P"] for r in rows])
        t = np.array([r["runtime_s"][s] for r in rows]) * 1e3
        a.scatter(P + {"unary": -0.2, "branch": -0.07, "qubo": 0.07, "hopfield_R16": 0.2}[s], t, s=14, color=color, label=s)
    a.set_yscale("log")
    a.set_xlabel("candidate interventions P [-]")
    a.set_ylabel("solver runtime per scene [ms]")
    a.set_title("Runtime vs P (exact table construction excluded)", loc="left", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    fams = list(FAMILIES)
    vals = [[np.mean([r["unary_failure"] for r in rows if r["family"] == f and r["repositioning_offered"]] or [0]),
             np.mean([r["validity"]["binding"] for r in rows if r["family"] == f]),
             np.mean([r["validity"]["critical"] for r in rows if r["family"] == f])] for f in fams]
    for i, (label, color) in enumerate(zip(("P(unary fail | repositioning)", "V binding (tie set changes)",
                                            "V critical (no V-free optimum valid)"), colors)):
        b.bar(np.arange(len(fams)) + (i - 1) * 0.26, [v[i] for v in vals], 0.25, color=color, label=label)
    b.set_xticks(np.arange(len(fams)), fams, fontsize=8)
    b.set_ylim(0, 1.05)
    b.set_ylabel("fraction [-]")
    b.set_title("Where structure matters, per family", loc="left", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=200, facecolor=surface)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = ds.load_dataset(ROOT / "s4" / "scenes.jsonl", ds.STAGE4_DIGEST)
    rows = [analyse(r, i) for i, r in enumerate(records) if r["intent"] == "repairable"]
    neg_stable = all(ds.labels_of(ds.label(ds.spec_from_json(r["spec"]), FINE)) == ds.labels_of(ds.label(ds.spec_from_json(r["spec"])))
                     for r in records if r["intent"] == "negative")
    overall = rates(rows)
    fam = {f: rates([r for r in rows if r["family"] == f]) for f in FAMILIES}
    rep = [r for r in rows if r["repositioning_offered"]]
    norep = [r for r in rows if not r["repositioning_offered"]]
    pair_cases = [r for r in rows if r["unary_failure"] and r["pairwise_correct"]]
    r_pair = {f: float(np.mean([r["unary_failure"] and r["pairwise_correct"] for r in rows if r["family"] == f])) for f in FAMILIES}
    kinds = {k for r in pair_cases for k, v in r["mechanisms"].items() if v}
    mean_rt = {s: float(np.mean([r["runtime_s"][s] for r in rows])) for s in ("unary", "branch", "qubo", "hopfield_R16")}
    evidence = {
        "C_star_stable": all(r["stable"]["C_star"] for r in rows) and neg_stable,
        "nontrivial_causal_sets": any(r["nontrivial_causal_sets"] for r in rows),
        "S_star_stable": all(r["stable"]["S_star"] for r in rows) and neg_stable,
        "all_S_star_valid_feasible": all(r["solvers"]["qubo"]["valid_feasible"] and r["solvers"]["qubo"]["exact_match"]
                                         for r in rows),
        "trivial_repair_fraction": float(np.mean([r["trivial_repair"] for r in rows])),
        "pairwise_failure_rate": float(np.mean([not r["pairwise_correct"] for r in rows])),
        "families_with_R_pair_min": sum(v >= op.R_PAIR_MIN for v in r_pair.values()),
        "n_mechanism_kinds": len(kinds), "mechanism_kinds": sorted(kinds),
        "hopfield_exact_R16": float(np.mean([r["hopfield"][16]["in_qubo_optimum"] for r in rows])),
        "hopfield_feasible_R16": overall["hopfield_R16"]["valid_feasible"],
        "hopfield_faster_than_exact_and_branch": mean_rt["hopfield_R16"] < min(mean_rt["qubo"], mean_rt["branch"]),
    }
    verdict = op.verdicts(evidence)
    out = {
        "stage": 6, "dataset_digest": ds.STAGE4_DIGEST, "n_repairable": len(rows),
        "factorization_note": op.__doc__.split("EXACT QUADRATIC")[0].strip(),
        "rules": op.__doc__.split("PRE-REGISTERED FINAL VERDICT RULES")[1].strip(),
        "constraint_encoding": {"zero_penalty_iff_M_and_V_all_scenes": all(r["constraint"]["zero_iff_M_and_V"] for r in rows),
                                "vpen_counts_violations_all_scenes": all(r["constraint"]["vpen_counts_violations"] for r in rows),
                                "min_penalty_over_B": min(r["constraint"]["min_penalty_when_violated"] / r["constraint"]["B"]
                                                          for r in rows),
                                "kappa_bound_assumption_min_positive_G_over_tol": min(r["min_positive_G"] for r in rows)
                                / st.G_FEAS_TOL, "CONTACT_TOL_3D": CONTACT_TOL_3D},
        "solver_accuracy_overall": overall, "solver_accuracy_per_family": fam,
        "hopfield_vs_exact_qubo": {R: {"in_qubo_optimum": float(np.mean([r["hopfield"][R]["in_qubo_optimum"] for r in rows])),
                                       "mean_energy_gap": float(np.mean([r["hopfield"][R]["energy_gap"] for r in rows])),
                                       "max_energy_gap": float(max(r["hopfield"][R]["energy_gap"] for r in rows))}
                                   for R in op.BUDGETS},
        "mean_runtime_s": mean_rt,
        "branch": {"conditional_unary_exact_all": all(r["branch_conditional_unary_exact"] for r in rows),
                   "mean_fraction_of_2P_examined": float(np.mean([w / n for w, n in (r["branch_work"] for r in rows)])),
                   "still_combinatorial": "per macro branch the geometry is a unary sum, but M (one destination per "
                                          "object) and V (placement compatibility) still require searching relocation subsets"},
        "vs_P": by(rows, "P", {"5-6": lambda p: p <= 6, "7-8": lambda p: 7 <= p <= 8, "9-10": lambda p: p >= 9}),
        "vs_geometric_density": by(rows, "geometric_density", {"0": lambda d: d == 0, "(0,0.2]": lambda d: 0 < d <= 0.2,
                                                               ">0.2": lambda d: d > 0.2}),
        "vs_physical_edges": by(rows, "n_physical_edges", {"0": lambda n: n == 0, "1-2": lambda n: 1 <= n <= 2,
                                                           ">=3": lambda n: n >= 3}),
        "P_unary_failure_given_repositioning": {"offered": [sum(r["unary_failure"] for r in rep), len(rep)],
                                                "not_offered": [sum(r["unary_failure"] for r in norep), len(norep)]},
        "validity": {k: float(np.mean([r["validity"][k] for r in rows])) for k in ("binding", "critical", "cost_raising")}
        | {"per_family": {f: {k: float(np.mean([r["validity"][k] for r in rows if r["family"] == f]))
                              for k in ("binding", "critical", "cost_raising")} for f in FAMILIES}},
        "R_pair_per_family": r_pair, "R_pair_cases": [r["scene_id"] for r in pair_cases],
        "R_pair_mechanism_counts": {k: sum(r["mechanisms"][k] for r in pair_cases) for k in pair_cases[0]["mechanisms"]},
        "evidence": evidence, "verdict": verdict, "scenes": rows,
    }
    (OUT / "verdict.json").write_text(json.dumps(out, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    plots(rows, overall)
    print(json.dumps({k: out[k] for k in ("constraint_encoding", "solver_accuracy_overall", "hopfield_vs_exact_qubo",
                                          "mean_runtime_s", "branch", "P_unary_failure_given_repositioning", "validity",
                                          "R_pair_per_family", "R_pair_mechanism_counts", "evidence",
                                          "verdict")}, indent=1, default=str))


if __name__ == "__main__":
    main()
