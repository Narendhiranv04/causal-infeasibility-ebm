"""Stage 2 contract tests: Möbius interaction order + exact QUBO."""

import itertools

import numpy as np
import pytest

from poc import energy as e
from poc import toy2d as t

SEED = 7
TOL = t.CONTACT_TOL


def _poly_table(P: int, terms: dict) -> np.ndarray:
    """G(x) = sum_T w_T prod_{p in T} x_p for planted terms {tuple(T): w_T}."""
    X = e.binary_matrix(P)
    return np.array([sum(w * all(X[x, p] for p in T) for T, w in terms.items()) for x in range(2 ** P)])


def _random_terms(rng, P: int, order: int) -> dict:
    return {T: float(rng.normal()) for k in range(order + 1) for T in itertools.combinations(range(P), k)}


# ------------------------------------------------------------- synthetic truth

def test_unary_coefficients_reconstruct_unary_truth_exactly():
    rng = np.random.default_rng(SEED)
    terms = _random_terms(rng, 6, 1)
    G = _poly_table(6, terms)
    a = e.mobius(G)
    assert e.coefficient(a) == pytest.approx(terms[()])
    assert all(e.coefficient(a, p) == pytest.approx(terms[(p,)]) for p in range(6))
    assert np.allclose(e.approximate(G, 1), G, atol=1e-12) and e.interaction_order(G, TOL) == 1


def test_pairwise_finite_difference_matches_planted_beta():
    rng = np.random.default_rng(SEED)
    terms = _random_terms(rng, 6, 2)
    G = _poly_table(6, terms)
    a = e.mobius(G)
    for p, q in itertools.combinations(range(6), 2):
        fd = G[(1 << p) | (1 << q)] - G[1 << p] - G[1 << q] + G[0]  # plan.md beta formula
        assert e.coefficient(a, p, q) == pytest.approx(fd) == pytest.approx(terms[(p, q)])
    assert np.allclose(e.approximate(G, 2), G, atol=1e-12)
    assert e.error_metrics(G, e.approximate(G, 1))["max_abs"] > 0.1
    assert e.interaction_order(G, TOL) == 2


def test_third_order_detects_a_known_triple_term():
    G = _poly_table(5, {(): 0.3, (0,): -0.1, (1, 3): 0.05, (0, 2, 4): -0.02})
    a = e.mobius(G)
    assert e.coefficient(a, 0, 2, 4) == pytest.approx(-0.02)
    assert e.degree(a, TOL) == 3 and e.interaction_order(G, TOL) == 3
    err = np.abs(e.approximate(G, 2) - G)
    assert err.max() == pytest.approx(0.02) and np.flatnonzero(err > TOL).tolist() == [0b10101, 0b10111, 0b11101, 0b11111]


@pytest.mark.parametrize("m", [2, 3, 4])
def test_m_way_substitutability_has_order_m(m):
    G = 0.01 * np.prod(1 - e.binary_matrix(m), axis=1)  # any one of m alternatives removes the conflict
    assert e.interaction_order(G, TOL) == m
    assert e.coefficient(e.mobius(G), *range(m)) == pytest.approx(0.01 * (-1) ** m)


def test_truncation_is_exact_on_small_subsets_and_invertible():
    G = np.random.default_rng(SEED).normal(size=2 ** 7)
    assert np.allclose(e.zeta(e.mobius(G)), G)
    size = e.popcount(len(G))
    for k in range(4):
        assert np.allclose(e.approximate(G, k)[size <= k], G[size <= k])


def test_qubo_matrix_form_equals_polynomial_energy():
    rng = np.random.default_rng(SEED)
    G = _poly_table(6, _random_terms(rng, 6, 2))
    lengths = [1, 2, 1, 3, 1, 1]
    Q, q, c = e.qubo(e.mobius(G), lengths, kappa=4.0, lam=0.5)
    assert np.array_equal(Q, Q.T) and np.all(np.diag(Q) == 0)
    H = e.qubo_energy(Q, q, c)
    assert np.allclose(H, e.polynomial_energy(e.mobius(G), lengths, 4.0, 0.5), atol=1e-12)
    assert np.allclose(H, 4.0 * G + 0.5 * e.binary_matrix(6) @ lengths, atol=1e-12)


def test_classification_rule():
    unary = {"order": 1, "physical_pairs": 0, "unary_repairs_match": True}
    subst = {"order": 2, "physical_pairs": 0, "unary_repairs_match": False}
    phys = {"order": 2, "physical_pairs": 1, "unary_repairs_match": False}
    assert e.classify([unary, subst])[0] == "A"  # substitutability alone does not justify QUBO
    assert e.classify([unary, phys])[0] == "B"
    assert e.classify([unary, phys, {"order": 3, "physical_pairs": 0, "unary_repairs_match": True}])[0] == "C"


# ---------------------------------------------------------- Stage 1 tables

@pytest.fixture(scope="module")
def stage1():
    out = {}
    for case in t.all_cases():
        results = t.enumerate_subsets(case)
        G, F, K = (np.array([getattr(r, f) for r in results], dtype=float) for f in ("G", "F", "K"))
        out[case.name] = (case, results, G, np.array([r.conflict for r in results]), F, K)
    return out


def _ids(case, xs):
    return frozenset(frozenset(iv.intervention_id for iv in t.subset(case.candidates, x)) for x in xs)


@pytest.mark.parametrize("name", ["T4_five", "H4_five"])
def test_five_blockers_are_unary_with_cardinality_five(name, stage1):
    case, results, G, *_ = stage1[name]
    assert {len(s) for s in t.minimal_repairs(results)} == {5}
    assert e.interaction_order(G, TOL) == 1
    c0 = dict(zip([b.eid for b in case.scene.entities], results[0].conflict))
    a = e.mobius(G)
    for p, iv in enumerate(case.candidates):
        assert e.coefficient(a, p) == pytest.approx(-c0[iv.entity_id], abs=1e-12)


def test_per_entity_decomposition_is_additive(stage1):
    _, _, G, C, _, _ = stage1["T6_coupled"]
    assert np.allclose(sum(e.mobius(C[:, i]) for i in range(C.shape[1])), e.mobius(G), atol=1e-15)


def test_pair_sources_distinguish_physical_from_substitutable(stage1):
    (t6,) = e.pair_sources(stage1["T6_coupled"][2], stage1["T6_coupled"][3], TOL)
    (t5,) = e.pair_sources(stage1["T5_redundant"][2], stage1["T5_redundant"][3], TOL)
    assert [s["kind"] for s in t6["entities"]] == [e.PHYSICAL] and t6["beta"] < 0
    assert [s["kind"] for s in t5["entities"]] == [e.SUBSTITUTABLE] and t5["beta"] > 0


def test_unary_fails_only_where_physical_coupling_exists(stage1):
    case, results, G, _, _, K = stage1["T6_coupled"]
    F_hat = (e.approximate(G, 1) > TOL).astype(int)
    J1 = [t.oracle_repair_cost(int(f), int(k), case.k_max) for f, k in zip(F_hat, K)]
    assert _ids(case, e.argmin_sets(J1)) != t.minimal_repairs(results)


def test_exact_qubo_argmin_matches_oracle_including_ties(stage1):
    for name, (case, results, G, _, F, K) in stage1.items():
        if e.interaction_order(G, TOL) > 2:
            continue
        kappa, lam = e.qubo_weights(t.oracle_repair_cost(1, 0, case.k_max), TOL)
        a, lengths = e.mobius(G), [iv.length for iv in case.candidates]
        H = e.qubo_energy(*e.qubo(a, lengths, kappa, lam))
        assert np.allclose(H, e.polynomial_energy(a, lengths, kappa, lam), rtol=1e-12, atol=1e-9)
        assert _ids(case, e.argmin_sets(H)) == t.minimal_repairs(results), name
        assert H[F == 0].max() < H[F == 1].min(initial=np.inf), "feasible repairs below infeasible"
        assert np.allclose(H[F == 0], K[F == 0], atol=e.ENERGY_TIE_TOL), "feasible energy = repair length"
    assert len(t.minimal_repairs(stage1["T5_redundant"][1])) == 2
