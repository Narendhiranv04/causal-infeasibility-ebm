"""PoC-2 Stage 6 contract tests: exact constraint encoding, QUBO, branch-on-macro, Hopfield, verdicts."""

import numpy as np
import pytest

from poc import energy as en
from poc2 import optimize as op
from poc2 import oracle as orc
from poc2 import scenes
from poc2 import structure as st

CASES = {c.name: c for c in scenes.stage1_cases()}


@pytest.fixture(scope="module")
def built():
    out = {}
    for name, case in CASES.items():
        T = orc.repair_table(case.scene, case.options)
        qb = op.build_qubo(T, case.options, op.validity_encoding(case.scene, case.options))
        out[name] = (case, T, qb, en.qubo_energy(qb["Q"], qb["q"], qb["c"]))
    return out


def test_preregistered_thresholds():
    assert (op.TRIVIAL_MAX, op.HIGHER_ORDER_MAX, op.R_PAIR_MIN) == (0.5, 0.05, 0.05)
    assert (op.HOPFIELD_EXACT_MIN, op.HOPFIELD_FEASIBLE_MIN, op.BUDGETS) == (0.95, 0.99, (1, 4, 16))


@pytest.mark.parametrize("name", sorted(CASES))
def test_quadratic_constraint_penalty_is_zero_exactly_when_M_and_V_hold(name, built):
    case, T, qb, _ = built[name]
    chk = op.constraint_check(T, qb)
    assert chk["zero_iff_M_and_V"] and chk["vpen_counts_violations"] and chk["min_penalty_when_violated"] >= qb["B"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_qubo_matrix_and_polynomial_energies_agree(name, built):
    case, T, qb, H = built[name]
    K = en.binary_matrix(len(case.options)) @ np.array([o.cost for o in case.options], dtype=float)
    poly = qb["kappa"] * qb["G2"] + qb["lambda_V"] * qb["vpen"] + qb["lambda_M"] * qb["cpen"] + K
    assert np.allclose(H, poly, rtol=1e-12, atol=1e-6)


@pytest.mark.parametrize("name", sorted(CASES))
def test_explicit_bounds_make_the_qubo_argmin_the_exact_repair(name, built):
    case, T, qb, H = built[name]
    assert en.argmin_sets(H) == st.exact_decision(T)
    invalid = ~((T.M == 1) & (T.V == 1) & (T.F == 0))
    assert H[invalid].min() >= qb["B"] > H[~invalid].max()


def test_pairwise_geometry_is_exact_on_the_consistent_domain(built):
    for case, T, qb, _ in built.values():
        assert np.max(np.abs(qb["G2"][T.M == 1] - T.G[T.M == 1])) <= 1e-12


@pytest.mark.parametrize("name", sorted(CASES))
def test_branch_on_macro_recovers_the_exact_repair(name, built):
    case, T, qb, _ = built[name]
    br = op.branch_on_macro(T, case.options)
    assert br["decision"] == st.exact_decision(T) and br["conditional_unary_exact"]
    assert br["subsets_examined"] <= br["of_2P"]


def test_hopfield_is_checked_only_against_the_exact_qubo_optimum(built):
    case, T, qb, H = built["M4_envelope_coupling"]
    masks, energies, _ = op.hopfield_runs(qb, [7, 0])
    best = masks[int(np.argmin(energies))]
    assert best in en.argmin_sets(H)


def test_validity_levels_distinguish_binding_from_critical(built):
    assert op.validity_levels(built["M2_placement_competition"][1]) == {"binding": True, "critical": True,
                                                                        "cost_raising": True}
    assert op.validity_levels(built["M5_distractors"][1]) == {"binding": True, "critical": False, "cost_raising": False}
    assert op.validity_levels(built["M3_substitutable"][1]) == {"binding": False, "critical": False, "cost_raising": False}


def test_verdict_rules():
    ev = {"C_star_stable": True, "nontrivial_causal_sets": False, "S_star_stable": True, "all_S_star_valid_feasible": True,
          "trivial_repair_fraction": 0.4, "pairwise_failure_rate": 0.0, "families_with_R_pair_min": 3,
          "n_mechanism_kinds": 2, "hopfield_exact_R16": 1.0, "hopfield_feasible_R16": 1.0,
          "hopfield_faster_than_exact_and_branch": False}
    assert op.verdicts(ev) == {"A_causal_diagnosis": "WEAK", "B_corrective_recourse": "STRONG",
                               "C_pairwise_hypothesis": "SUPPORTED", "D_hopfield": "OPTIONAL"}
    assert op.verdicts({**ev, "pairwise_failure_rate": 0.2})["C_pairwise_hypothesis"] == "INSUFFICIENT"
    assert op.verdicts({**ev, "families_with_R_pair_min": 1})["C_pairwise_hypothesis"] == "NOT NEEDED"
    assert op.verdicts({**ev, "hopfield_exact_R16": 0.9})["D_hopfield"] == "DROP"
    assert op.verdicts({**ev, "hopfield_faster_than_exact_and_branch": True})["D_hopfield"] == "RETAIN"
