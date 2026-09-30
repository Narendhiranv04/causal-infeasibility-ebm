"""Stage 2 experiment: interaction-order analysis of the Stage 1 tables + exact QUBO.

Run from the repository root:

    PYTHONPATH=src python scripts/s2_energy.py

Writes out/s2/metrics.json. The exhaustive Stage 1 table S -> G(S) (unchanged
oracle, G = L1) is ground truth. Deterministic; SEED recorded for the contract.
"""

import json
import time
from pathlib import Path

import numpy as np

from poc import energy as e
from poc import toy2d as t

OUT = Path("out/s2")
SEED = 7
ORDER_TOL = t.CONTACT_TOL  # [m] expansion residual below the oracle's contact resolution
ORDERS = (1, 2, 3)
SUBSTITUTION_DEPTH = 0.01  # [m] conflict removable by any one of m alternatives (synthetic check)


def _ids(case, x: int) -> frozenset[str]:
    return frozenset(iv.intervention_id for iv in t.subset(case.candidates, x))


def _sets(fs) -> list[list[str]]:
    return sorted(sorted(s) for s in fs)


def tables(case):
    """Exhaustive Stage 1 table as arrays G(x), c^S(x), F(x), K(x) indexed by bitmask x."""
    results = t.enumerate_subsets(case)
    G, F, K = (np.array([getattr(r, f) for r in results], dtype=float) for f in ("G", "F", "K"))
    return results, G, np.array([r.conflict for r in results]), F, K


def surrogate(case, G_hat, K):
    """Minimal repairs implied by an approximate G: F_hat = 1[G_hat > CONTACT_TOL], J = oracle cost."""
    F_hat = (G_hat > t.CONTACT_TOL).astype(int)
    J = [t.oracle_repair_cost(int(f), int(k), case.k_max) for f, k in zip(F_hat, K)]
    return F_hat, e.argmin_sets(J)


def approximation_report(case, G, F, K, s_star) -> dict:
    out = {}
    for k in ORDERS:
        F_hat, xs = surrogate(case, e.approximate(G, k), K)
        found = frozenset(_ids(case, x) for x in xs)
        out[f"order_{k}"] = {
            **e.error_metrics(G, e.approximate(G, k)),
            "feasibility_agreement": float(np.mean(F_hat == F)),
            "surrogate_minimal_repairs": _sets(found),
            "minimal_repair_match": found == s_star,
            "surrogate_repairs_truly_feasible": all(F[x] == 0 for x in xs),
        }
    return out


def qubo_report(case, coeffs, G, F, K, s_star) -> dict:
    B = t.oracle_repair_cost(1, 0, case.k_max)  # B = K_max + 1, owned by the Stage 1 oracle
    kappa, lam = e.qubo_weights(B, t.CONTACT_TOL)
    lengths = [iv.length for iv in case.candidates]
    Q, q, c = e.qubo(coeffs, lengths, kappa, lam)
    H = e.qubo_energy(Q, q, c)
    found = frozenset(_ids(case, x) for x in e.argmin_sets(H))
    feas, infeas = H[F == 0], H[F == 1]
    return {
        "B": B, "kappa": kappa, "lambda": lam,
        "Q": Q.tolist(), "q": q.tolist(), "c": c,
        "max_abs_qubo_minus_polynomial": float(np.max(np.abs(H - e.polynomial_energy(coeffs, lengths, kappa, lam)))),
        "max_abs_qubo_minus_exact_energy": float(np.max(np.abs(H - (kappa * G + lam * K)))),
        "energy_scale": float(np.max(np.abs(H))),
        "argmin": _sets(found), "argmin_matches_oracle": found == s_star,
        "feasible_below_infeasible": bool(infeas.size == 0 or feas.max() < infeas.min()),
        "feasible_energy_minus_K_max_abs": float(np.max(np.abs(feas - K[F == 0]))),
    }


def analyse(case) -> dict:
    results, G, C, F, K = tables(case)
    s_star = t.minimal_repairs(results)
    coeffs = e.mobius(G)
    names = [iv.intervention_id for iv in case.candidates]
    eids = [b.eid for b in case.scene.entities]
    sources = [{"pair": [names[s["p"]], names[s["q"]]], "beta": s["beta"],
                "entities": [{**sh, "entity": eids[sh["entity"]]} for sh in s["entities"]]}
               for s in e.pair_sources(G, C, ORDER_TOL)]
    kinds = [{sh["kind"] for sh in s["entities"]} for s in sources]
    order = e.interaction_order(G, ORDER_TOL)
    report = {
        "name": case.name, "family": case.family, "P": len(case.candidates),
        "minimal_repairs": _sets(s_star), "minimal_cardinality": sorted({len(s) for s in s_star}),
        "order": order, "degree": e.degree(coeffs, ORDER_TOL),
        "G0": float(coeffs[0]),
        "alpha": coefficient_dict(coeffs, names, 1), "beta": coefficient_dict(coeffs, names, 2),
        "max_abs_gamma": float(np.max(np.abs(coeffs[e.popcount(len(coeffs)) == 3]), initial=0.0)),
        "approximations": approximation_report(case, G, F, K, s_star),
        "pair_sources": sources,
        "physical_pairs": sum(e.PHYSICAL in k for k in kinds),
        "substitutable_only_pairs": sum(k == {e.SUBSTITUTABLE} for k in kinds),
    }
    report["unary_repairs_match"] = report["approximations"]["order_1"]["minimal_repair_match"]
    if order <= 2:
        report["qubo"] = qubo_report(case, coeffs, G, F, K, s_star)
    return report


def coefficient_dict(coeffs, names, size: int) -> dict:
    """Nonzero Möbius coefficients of one order, keyed by intervention ids."""
    out = {}
    for x in np.flatnonzero(e.popcount(len(coeffs)) == size):
        key = tuple(n for p, n in enumerate(names) if x >> p & 1)
        if abs(coeffs[x]) > ORDER_TOL:
            out["*".join(key)] = float(coeffs[x])
    return out


def additivity_evidence(reports: list[dict], cases: dict) -> list[dict]:
    """High cardinality vs interaction order: five-blocker cases."""
    rows = []
    for r in reports:
        if r["family"] != "five_independent":
            continue
        case = cases[r["name"]]
        c0 = dict(zip([b.eid for b in case.scene.entities], t.conflict(case.scene)))
        alpha_dev = max(abs(r["alpha"].get(iv.intervention_id, 0.0) + c0[iv.entity_id]) for iv in case.candidates)
        rows.append({"name": r["name"], "minimal_cardinality": r["minimal_cardinality"], "order": r["order"],
                     "alpha_equals_minus_c_empty_max_dev": alpha_dev, "n_nonzero_beta": len(r["beta"]),
                     "max_abs_gamma": r["max_abs_gamma"],
                     "unary_max_abs_error": r["approximations"]["order_1"]["max_abs"]})
    return rows


def substitution_orders() -> dict:
    """Synthetic: one conflict removable by any of m alternatives, G = d * prod_p (1 - x_p)."""
    out = {}
    for m in (2, 3, 4):
        X = e.binary_matrix(m)
        G = SUBSTITUTION_DEPTH * np.prod(1 - X, axis=1)
        out[f"m={m}"] = {"order": e.interaction_order(G, ORDER_TOL),
                         "top_coefficient": e.coefficient(e.mobius(G), *range(m))}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {c.name: c for c in t.all_cases()}
    t0 = time.perf_counter()
    reports = [analyse(c) for c in cases.values()]
    label, reason = e.classify(reports)
    qubos = [r for r in reports if "qubo" in r]
    summary = {
        "classification": label, "reason": reason,
        "orders": {r["name"]: r["order"] for r in reports},
        "n_cases_by_order": {k: sum(r["order"] == k for r in reports) for k in range(4)},
        "cases_with_physical_pairs": [r["name"] for r in reports if r["physical_pairs"]],
        "cases_with_substitutable_only_pairs": [r["name"] for r in reports if r["substitutable_only_pairs"]],
        "unary_minimal_repair_match": sum(r["unary_repairs_match"] for r in reports),
        "pairwise_minimal_repair_match": sum(r["approximations"]["order_2"]["minimal_repair_match"] for r in reports),
        "n_cases": len(reports), "n_pairwise_representable": len(qubos),
        "qubo_argmin_matches_oracle": sum(r["qubo"]["argmin_matches_oracle"] for r in qubos),
        "qubo_ordering_verified": all(r["qubo"]["feasible_below_infeasible"] for r in qubos),
        "runtime_s": round(time.perf_counter() - t0, 3),
    }
    metrics = {
        "stage": 2, "seed": SEED,
        "constants": {"ORDER_TOL": ORDER_TOL, "ENERGY_TIE_TOL": e.ENERGY_TIE_TOL, "G_FORM": t.G_FORM,
                      "kappa_rule": "B / CONTACT_TOL", "lambda_rule": "1 (energy unit = one repair action)"},
        "summary": summary,
        "five_blocker_additivity": additivity_evidence(reports, cases),
        "synthetic_substitution_orders": substitution_orders(),
        "cases": reports,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("summary", "five_blocker_additivity",
                                              "synthetic_substitution_orders")}, indent=2))


if __name__ == "__main__":
    main()
