"""Stage 3 contract tests: QUBO -> Hopfield mapping and asynchronous descent."""

import numpy as np
import pytest

from poc import energy as e
from poc import hopfield as hf
from poc import toy2d as t

SEED = 7


def _random_qubo(rng, P: int, raw: bool = False):
    """Mixed-sign QUBO; raw=True gives a non-symmetric Q with a nonzero diagonal."""
    Q = rng.normal(size=(P, P))
    if not raw:
        Q = np.triu(Q, 1) + np.triu(Q, 1).T
    return Q, rng.normal(size=P), float(rng.normal())


def _qubo_all(Q, q, c):
    X = e.binary_matrix(len(q)).astype(float)
    return np.einsum("xp,pr,xr->x", X, Q, X) + X @ q + c  # direct definition, any Q


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("P", [1, 3, 6])
def test_hopfield_energy_equals_qubo_plus_constant_on_every_state(P, raw):
    Q, q, c = _random_qubo(np.random.default_rng(SEED + P), P, raw)
    net = hf.to_hopfield(Q, q, c)
    S = hf.spins(e.binary_matrix(P))
    diff = hf.energy(net, S) - _qubo_all(Q, q, c)
    assert np.allclose(diff, net.C, atol=1e-12), "E_H(s) = H_Q(x) + C on all 2^P states"


def test_weights_symmetric_zero_diagonal():
    net = hf.to_hopfield(*_random_qubo(np.random.default_rng(SEED), 7, raw=True))
    assert np.array_equal(net.W, net.W.T) and np.all(np.diag(net.W) == 0.0)


@pytest.mark.parametrize("q1, best_spin", [(+2.0, -1), (-2.0, +1)])
def test_bias_sign_convention(q1, best_spin):
    net = hf.to_hopfield(np.zeros((1, 1)), np.array([q1]), 0.0)  # H = q1 x
    assert np.sign(net.b[0]) == best_spin  # b = -q/2: positive cost pushes s to -1 (x = 0)
    assert hf.descend(net, np.array([-best_spin]), order="fixed").s[0] == best_spin


def test_local_field_rule_equals_exact_one_bit_energy_difference():
    rng = np.random.default_rng(SEED)
    Q, q, c = _random_qubo(rng, 8, raw=True)
    net = hf.to_hopfield(Q, q, c)
    for _ in range(50):
        s = rng.choice([-1, 1], size=8)
        for i in range(8):
            flipped = s.copy()
            flipped[i] = -flipped[i]
            exact = hf.energy(net, flipped) - hf.energy(net, s)
            H_diff = _qubo_all(Q, q, c)[int(hf.bits(flipped) @ (1 << np.arange(8)))] \
                - _qubo_all(Q, q, c)[int(hf.bits(s) @ (1 << np.arange(8)))]
            assert hf.flip_delta(net, s, i) == pytest.approx(exact, abs=1e-12) == pytest.approx(H_diff, abs=1e-12)


@pytest.mark.parametrize("order", ["random", "fixed"])
def test_energy_never_increases_and_converged_state_is_a_fixed_point(order):
    rng = np.random.default_rng(SEED)
    for _ in range(40):
        net = hf.to_hopfield(*_random_qubo(rng, 10))
        s0 = rng.choice([-1, 1], size=10)
        d = hf.descend(net, s0, rng, order, record=True)
        trace = np.array((hf.energy(net, s0),) + d.trace)
        assert np.all(np.diff(trace) <= 1e-12)
        assert d.converged and hf.is_fixed_point(net, d.s)
        assert d.energy == pytest.approx(trace[-1], abs=1e-9)


def test_zero_field_keeps_current_state():
    net = hf.to_hopfield(np.zeros((4, 4)), np.zeros(4), 0.0)  # every h_i = 0, every flip dE = 0
    s0 = np.array([1, -1, -1, 1])
    d = hf.descend(net, s0, order="fixed")
    assert np.array_equal(d.s, s0) and d.flips == 0 and d.converged and d.sweeps == 1


def test_max_sweeps_is_a_hard_bound():
    rng = np.random.default_rng(SEED)
    results = [hf.descend(hf.to_hopfield(*_random_qubo(rng, 12)), rng.choice([-1, 1], size=12), rng, max_sweeps=1)
               for _ in range(50)]
    assert all(d.sweeps <= 1 for d in results)
    assert any(not d.converged for d in results)  # one sweep is not always enough


def test_seeded_runs_are_deterministic():
    net = hf.to_hopfield(*_random_qubo(np.random.default_rng(SEED), 10))
    a = hf.restarts(net, 8, np.random.default_rng(SEED))
    b = hf.restarts(net, 8, np.random.default_rng(SEED))
    assert all(np.array_equal(x.s, y.s) and x.energy == y.energy for x, y in zip(a, b))


@pytest.mark.parametrize("eta", [1e-6, 1e-3, 1.0, 1e3, 1e6])
def test_global_positive_rescaling_keeps_argmin_and_hopfield_decisions(eta):
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        Q, q, c = _random_qubo(rng, 8)
        H, H_eta = _qubo_all(Q, q, c), _qubo_all(*hf.rescale(Q, q, c, eta))
        assert e.argmin_sets(H, 1e-9) == e.argmin_sets(H_eta, 1e-9 * eta)
        base = hf.restarts(hf.to_hopfield(Q, q, c), 6, np.random.default_rng(SEED))
        scaled = hf.restarts(hf.to_hopfield(*hf.rescale(Q, q, c, eta)), 6, np.random.default_rng(SEED))
        assert all(np.array_equal(x.s, y.s) for x, y in zip(base, scaled))


def test_normalising_factor_sets_max_coefficient_to_one():
    Q, q, c = _random_qubo(np.random.default_rng(SEED), 5)
    Qn, qn, _ = hf.rescale(Q, q, c, hf.normalising_factor(Q, q))
    assert max(np.abs(Qn).max(), np.abs(qn).max()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        hf.rescale(Q, q, c, 0.0)


def test_fixed_point_is_not_global_optimality_on_the_coupled_scene():
    """T6: 'do nothing' (x = 0) is a converged fixed point but not the QUBO optimum."""
    case = next(c for c in t.all_cases() if c.name == "T6_coupled")
    G = np.array([r.G for r in t.enumerate_subsets(case)])
    kappa, lam = e.qubo_weights(t.oracle_repair_cost(1, 0, case.k_max), t.CONTACT_TOL)
    Q, q, c = e.qubo(e.mobius(G), [iv.length for iv in case.candidates], kappa, lam)
    net = hf.to_hopfield(*hf.rescale(Q, q, c, hf.normalising_factor(Q, q)))
    d = hf.descend(net, -np.ones(3), order="fixed")
    assert d.converged and np.array_equal(hf.bits(d.s), [0, 0, 0])
    assert 0 not in e.argmin_sets(e.qubo_energy(Q, q, c))
