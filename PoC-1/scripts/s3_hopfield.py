"""Stage 3 experiment: Hopfield as a candidate solver for already-validated QUBOs.

Run from the repository root:

    PYTHONPATH=src python scripts/s3_hopfield.py

Writes out/s3/metrics.json. Exhaustive enumeration of every QUBO is the ground
truth; a converged Hopfield state is only a fixed point (one-flip local minimum).

Distributions, reported separately:
  A  scene:     the 11 Stage-1/2 pairwise-representable scene QUBOs, unchanged.
  B1 stress:    seeded random mixed-sign pairwise QUBOs (frustrated, many local
                minima). Not representative of manipulation physics.
  B2 motif:     synthetic repair QUBOs composed of the Stage-1 motifs (unary
                blocker, substitutable pair as T5, coupled shift as T6) with
                k = 1..8 required interventions. Synthetic, not physical scenes.

Pre-registered gate (fixed before evaluation): random update order, R_GATE
restarts; exact-optimum recovery >= 95% on every distribution and true-feasible
repair >= 99% on A (and B2, which has a feasibility oracle).
"""

import json
import time
from pathlib import Path

import numpy as np

from poc import energy as e
from poc import hopfield as hf
from poc import toy2d as t

OUT = Path("out/s3")
SEED = 7
BUDGETS = (1, 4, 16, 64)   # restart budgets R, evaluated as nested prefixes of one 64-restart run
R_GATE = 16                # "reasonable" budget: 16 descents cost fewer energy evaluations than 2^12
ORDERS = ("random", "fixed")
TARGET_EXACT, TARGET_FEASIBLE = 0.95, 0.99
STRESS_P, STRESS_DENSITY, STRESS_PER_CELL = (6, 8, 10, 12), (0.3, 0.6, 1.0), 25
MOTIF_P, MOTIF_K, MOTIF_PER_K = 12, tuple(range(1, 9)), 10
DEPTH_RANGE = (0.005, 0.03)  # [m] motif conflict depths, the Stage-1 range


def _masks(case, id_sets) -> set[int]:
    names = [iv.intervention_id for iv in case.candidates]
    return {sum(1 << names.index(i) for i in s) for s in id_sets}


def scene_instances() -> list[dict]:
    out = []
    for case in t.all_cases():
        results = t.enumerate_subsets(case)
        G = np.array([r.G for r in results])
        kappa, lam = e.qubo_weights(t.oracle_repair_cost(1, 0, case.k_max), t.CONTACT_TOL)
        qubo = e.qubo(e.mobius(G), [iv.length for iv in case.candidates], kappa, lam)
        out.append({"name": case.name, "qubo": qubo, "F": np.array([r.F for r in results]),
                    "minimal": _masks(case, t.minimal_repairs(results)), "bins": {"family": case.family}})
    return out


def stress_instances() -> list[dict]:
    rng, out = np.random.default_rng([SEED, 1]), []
    for P in STRESS_P:
        for density in STRESS_DENSITY:
            for k in range(STRESS_PER_CELL):
                upper = np.triu(np.where(rng.random((P, P)) < density, rng.normal(size=(P, P)), 0.0), 1)
                out.append({"name": f"stress_P{P}_d{density}_{k}", "qubo": (upper + upper.T, rng.normal(size=P), 0.0),
                            "F": None, "minimal": None, "bins": {"P": P, "density": density}})
    return out


def motif_instance(rng, P: int, k: int, name: str) -> dict:
    X = e.binary_matrix(P)[:, rng.permutation(P)]  # motif variable v is column v of a permuted order
    G, v, need = np.zeros(2 ** P), 0, k
    while need > 0:
        kinds = ["unary"] + (["substitutable"] if P - v - 2 >= need - 1 else []) \
            + (["coupled"] if need >= 2 and P - v - 2 >= need - 2 else [])
        kind, (c1, c2) = rng.choice(kinds), rng.uniform(*DEPTH_RANGE, size=2)
        if kind == "unary":
            G, v, need = G + c1 * (1 - X[:, v]), v + 1, need - 1
        elif kind == "substitutable":
            G, v, need = G + c1 * (1 - X[:, v]) * (1 - X[:, v + 1]), v + 2, need - 1
        else:  # shift clears a fixed cause c1 but pushes into movable neighbour c2
            G, v, need = G + c1 * (1 - X[:, v]) + c2 * X[:, v] * (1 - X[:, v + 1]), v + 2, need - 2
    F, K = (G > t.CONTACT_TOL).astype(int), e.popcount(2 ** P)
    kappa, lam = e.qubo_weights(t.oracle_repair_cost(1, 0, P), t.CONTACT_TOL)
    J = [t.oracle_repair_cost(int(f), int(n), P) for f, n in zip(F, K)]
    return {"name": name, "qubo": e.qubo(e.mobius(G), [1] * P, kappa, lam), "F": F,
            "minimal": set(e.argmin_sets(J)), "bins": {"k": k}}


def motif_instances() -> list[dict]:
    rng = np.random.default_rng([SEED, 2])
    return [motif_instance(rng, MOTIF_P, k, f"motif_k{k}_{i}") for k in MOTIF_K for i in range(MOTIF_PER_K)]


def evaluate(inst: dict, index: int, order: str) -> dict:
    Q, q, c = inst["qubo"]
    P = len(q)
    t0 = time.perf_counter()
    H = e.qubo_energy(Q, q, c)
    opt = e.argmin_sets(H)
    t_exact = time.perf_counter() - t0
    net = hf.to_hopfield(*hf.rescale(Q, q, c, hf.normalising_factor(Q, q)))
    t0 = time.perf_counter()
    runs = hf.restarts(net, max(BUDGETS), np.random.default_rng([SEED, index]), order)
    t_hop = (time.perf_counter() - t0) / len(runs)
    weights = 1 << np.arange(P)
    masks = [int(hf.bits(d.s) @ weights) for d in runs]
    S = hf.spins(e.binary_matrix(P))
    row = {
        "name": inst["name"], "P": P, **inst["bins"], "optimum_cardinality": bin(min(opt)).count("1"),
        "n_fixed_points": int(np.sum(np.all(S * (S @ net.W + net.b) >= 0, axis=1))),
        "first_hit": next((r + 1 for r, m in enumerate(masks) if m in opt), None),
        "all_converged": all(d.converged for d in runs), "mean_flips": float(np.mean([d.flips for d in runs])),
        "mean_sweeps": float(np.mean([d.sweeps for d in runs])),
        "runtime_exact_s": t_exact, "runtime_per_restart_s": t_hop, "budgets": {},
    }
    span = H.max() - H.min()
    for R in BUDGETS:
        m = masks[min(range(R), key=lambda j: runs[j].energy)]  # same first-on-ties rule as hf.best
        row["budgets"][R] = {"exact": m in opt, "gap": float(H[m] - H.min()),
                             "rel_gap": float((H[m] - H.min()) / span) if span > 0 else 0.0,
                             "feasible": None if inst["F"] is None else bool(inst["F"][m] == 0),
                             "minimal": None if inst["minimal"] is None else m in inst["minimal"]}
    return row


def aggregate(rows: list[dict]) -> dict:
    out = {"n": len(rows), "never_hit_within_64": sum(r["first_hit"] is None for r in rows),
           "mean_first_hit": float(np.mean([r["first_hit"] for r in rows if r["first_hit"]] or [np.nan])),
           "max_first_hit": max((r["first_hit"] or 0) for r in rows),
           "mean_fixed_points": float(np.mean([r["n_fixed_points"] for r in rows])),
           "max_fixed_points": max(r["n_fixed_points"] for r in rows),
           "frac_multiple_fixed_points": float(np.mean([r["n_fixed_points"] > 1 for r in rows])),
           "all_converged": all(r["all_converged"] for r in rows),
           "mean_flips": float(np.mean([r["mean_flips"] for r in rows])),
           "mean_sweeps": float(np.mean([r["mean_sweeps"] for r in rows])), "by_R": {}}
    for R in BUDGETS:
        b = [r["budgets"][R] for r in rows]
        rate = lambda key: None if b[0][key] is None else float(np.mean([x[key] for x in b]))  # noqa: E731
        out["by_R"][R] = {"exact_optimum_rate": rate("exact"), "true_feasible_rate": rate("feasible"),
                          "minimal_repair_rate": rate("minimal"), "mean_gap": float(np.mean([x["gap"] for x in b])),
                          "max_gap": float(np.max([x["gap"] for x in b])),
                          "mean_rel_gap": float(np.mean([x["rel_gap"] for x in b])),
                          "hopfield_runtime_s": float(np.mean([r["runtime_per_restart_s"] for r in rows])) * R}
    out["exact_enumeration_runtime_s"] = float(np.mean([r["runtime_exact_s"] for r in rows]))
    return out


def binned(rows: list[dict], key: str) -> dict:
    values = sorted({r[key] for r in rows})
    return {str(v): {R: float(np.mean([r["budgets"][R]["exact"] for r in rows if r[key] == v])) for R in BUDGETS}
            | {"n": sum(r[key] == v for r in rows)} for v in values}


def gate(summary: dict) -> dict:
    checks = {}
    for dist in ("scene", "stress", "motif"):
        at = summary[dist]["random"]["by_R"][R_GATE]
        checks[f"{dist}_exact"] = at["exact_optimum_rate"] >= TARGET_EXACT
        if at["true_feasible_rate"] is not None:
            checks[f"{dist}_feasible"] = at["true_feasible_rate"] >= TARGET_FEASIBLE
    passed = all(checks.values())
    cost = {d: summary[d]["random"]["by_R"][R_GATE]["hopfield_runtime_s"] / summary[d]["random"]["exact_enumeration_runtime_s"]
            for d in ("scene", "stress", "motif")}
    return {"R_GATE": R_GATE, "order": "random", "checks": checks, "passed": passed,
            "runtime_ratio_hopfield_over_exact_enumeration": cost,
            "verdict": "QUBO VALID - HOPFIELD MEETS TARGETS" if passed else "QUBO VALID - HOPFIELD NOT JUSTIFIED"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    dists = {"scene": scene_instances(), "stress": stress_instances(), "motif": motif_instances()}
    rows = {d: {o: [evaluate(inst, i, o) for i, inst in enumerate(insts)] for o in ORDERS} for d, insts in dists.items()}
    summary = {d: {o: aggregate(rows[d][o]) for o in ORDERS} for d in dists}
    metrics = {
        "stage": 3, "seed": SEED,
        "seeding": "instances: default_rng([7, family]); descents: default_rng([7, instance_index]) per order",
        "constants": {"BUDGETS": BUDGETS, "R_GATE": R_GATE, "MAX_SWEEPS": hf.MAX_SWEEPS,
                      "ENERGY_TIE_TOL": e.ENERGY_TIE_TOL, "normalisation": "eta = 1/max|coef(Q, q)|, global"},
        "gate": gate(summary),
        "summary": summary,
        "bins_random_order": {
            "stress_by_P": binned(rows["stress"]["random"], "P"),
            "stress_by_density": binned(rows["stress"]["random"], "density"),
            "stress_by_optimum_cardinality": binned(rows["stress"]["random"], "optimum_cardinality"),
            "motif_by_k": binned(rows["motif"]["random"], "k"),
        },
        "scene_rows_random_order": rows["scene"]["random"],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({"gate": metrics["gate"], "bins": metrics["bins_random_order"],
                      "runtime_s": metrics["runtime_s"]}, indent=1))
    for d in dists:
        for o in ORDERS:
            s = summary[d][o]
            print(d, o, "max_first_hit:", s["max_first_hit"], "max_fixed_pts:", s["max_fixed_points"], {R: (v["exact_optimum_rate"], v["true_feasible_rate"], round(v["mean_gap"], 4))
                         for R, v in s["by_R"].items()}, "never_hit:", s["never_hit_within_64"],
                  "fixed_pts:", round(s["mean_fixed_points"], 2))


if __name__ == "__main__":
    main()
