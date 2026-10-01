"""PoC-2 Stage 3 contract tests: compatible-domain structure analysis (plan2.md section 19)."""

import itertools

import numpy as np
import pytest

from poc import energy as en
from poc2 import oracle as orc
from poc2 import scenes
from poc2 import structure as st
from poc2.oracle import RepairTable

# Options 0 and 1 are mutually exclusive (one object, two destinations); 2 and 3 are free.
P, GROUP = 4, 0b0011


def _M() -> np.ndarray:
    return np.array([int((x & GROUP).bit_count() <= 1) for x in range(2 ** P)])


def _planted(terms: dict) -> np.ndarray:
    """G(x) = sum_T w_T prod_{p in T} x_p on the consistent domain, NaN elsewhere."""
    G = np.array([sum(w * all(x >> p & 1 for p in T) for T, w in terms.items()) for x in range(2 ** P)], dtype=float)
    return np.where(_M() == 1, G, np.nan)


def _table(G, V=None, F=None) -> RepairTable:
    M = _M()
    V = np.where(M == 1, 1, -1) if V is None else V
    F = np.where(M == 1, (np.nan_to_num(G) > st.G_FEAS_TOL).astype(int), -1) if F is None else F
    K = np.array([x.bit_count() for x in range(2 ** P)])
    return RepairTable(("e",), tuple(f"I{p}" for p in range(P)), M, V, F, G, np.zeros((2 ** P, 1)), K, None)


def test_preregistered_constants():
    assert (st.COEF_FLOOR, st.SIG_FACTOR, st.G_FEAS_TOL, st.REPR_FACTOR, st.ORDERS) == (1e-6, 10.0, 1e-6, 10.0, (1, 2, 3))


# ------------------------------------------------------------ compatible-domain Möbius

def test_planted_coefficients_are_recovered_exactly_on_the_consistent_domain():
    terms = {(): 0.03, (0,): -0.01, (1,): -0.02, (2,): 0.004, (3,): -0.007, (0, 2): 0.002, (1, 3): -0.005,
             (2, 3): 0.001, (0, 2, 3): -0.003}
    G = _planted(terms)
    a = st.compatible_mobius(G, _M())
    for T, w in terms.items():
        assert a[sum(1 << p for p in T)] == pytest.approx(w, abs=1e-15)
    assert np.isnan(a[0b0011]) and np.isnan(a[0b1111]), "contradictory selections get no coefficient"
    assert st.representation_errors(G, _M(), st.reconstruct(a, 3))["max_abs"] == pytest.approx(0.0, abs=1e-15)
    assert st.representation_errors(G, _M(), st.reconstruct(a, 2))["max_abs"] == pytest.approx(0.003, abs=1e-15)


def test_reconstruction_is_finite_exactly_on_the_consistent_domain():
    G = _planted({(): 0.02, (0,): -0.02, (2, 3): 0.01})
    G2 = st.reconstruct(st.compatible_mobius(G, _M()), 2)
    assert np.all(np.isfinite(G2[_M() == 1])) and np.all(np.isnan(G2[_M() == 0]))


def test_representation_order_detects_a_planted_triple():
    G = _planted({(): 0.02, (0,): -0.01, (0, 2, 3): -0.004})
    a = st.compatible_mobius(G, _M())
    assert st.representation_order(G, _M(), a, st.COEF_FLOOR) == 3
    unary = _planted({(): 0.02, (0,): -0.01, (3,): -0.01})
    assert st.representation_order(unary, _M(), st.compatible_mobius(unary, _M()), st.COEF_FLOOR) == 1


def test_significance_requires_stable_sign_and_resolution():
    base = np.array([np.nan, 0.02, 0.02, 5e-7, 1e-3, -0.01])
    fine = np.array([np.nan, 0.02, -0.02, 5e-7, 7e-4, -0.0101])
    assert st.significant(base, fine).tolist() == [False, True, False, False, False, True]


# ------------------------------------------------------------ decisions

def test_order_decisions_use_exact_admissibility_and_cost():
    # Option 3 alone does nothing, option 2 alone overshoots; only {2, 3} together clears the conflict.
    G = _planted({(): 0.02, (2,): -0.01, (3,): 0.0, (2, 3): -0.01})
    T = _table(G)
    exact = st.exact_decision(T)
    assert exact == {0b1100}
    assert st.order_decision(T, st.reconstruct(st.compatible_mobius(G, _M()), 1)) != exact  # unary misses
    assert st.order_decision(T, st.reconstruct(st.compatible_mobius(G, _M()), 2)) == exact


def test_static_invalidity_is_never_overridden_by_a_surrogate():
    G = _planted({(): 0.02, (0,): -0.02, (1,): -0.02})
    V = np.where(_M() == 1, 1, -1)
    V[0b0001] = 0  # option 0 would clear the conflict but leaves an invalid scene
    T = _table(G, V=V)
    for k in st.ORDERS:
        assert st.order_decision(T, st.reconstruct(st.compatible_mobius(G, _M()), k)) == {0b0010}


# ------------------------------------------------------------ interaction sources

def test_sources_stay_separate_on_the_stage1_cases():
    for case in scenes.stage1_cases():
        T = orc.repair_table(case.scene, case.options)
        a = st.compatible_mobius(T.G, T.M)
        edges = st.interaction_edges(T, a, st.significant(a, a))
        geo = {(p, q) for p, q, _, _ in edges["geometric"]}
        assert not geo & set(edges["choice"]), "a choice constraint is never a geometric beta"
        if case.mechanism == "placement_competition":
            p, q = (list(T.option_ids).index(i) for i in ("b1->rA", "b2->rA"))
            assert (p, q) in edges["validity"] and (p, q) not in geo
        if case.mechanism == "envelope_coupling":
            p, q = (list(T.option_ids).index(i) for i in ("shift", "nb->r1"))
            assert (p, q, pytest.approx(-0.0199, abs=1e-6), st.PHYSICAL) in edges["geometric"]
        if case.mechanism == "substitutable":
            assert [kind for *_, kind in edges["geometric"]] == [st.SUBSTITUTABLE]


def test_base_fine_significance_is_deterministic():
    case = scenes.envelope_coupling()
    runs = []
    for _ in range(2):
        b = orc.repair_table(case.scene, case.options)
        f = orc.repair_table(case.scene, case.options, 1e-3)
        runs.append(st.significant(st.compatible_mobius(b.G, b.M), st.compatible_mobius(f.G, f.M)))
    assert np.array_equal(*runs)


def test_density_of_a_complete_graph_is_one():
    assert st.density(len(list(itertools.combinations(range(5), 2))), 5) == 1.0 and st.density(0, 1) == 0.0
    assert en.popcount(8).tolist() == [0, 1, 1, 2, 1, 2, 2, 3]
