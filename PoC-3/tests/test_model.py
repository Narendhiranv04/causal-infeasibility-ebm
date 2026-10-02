"""PoC-3 Stage 1 model tests: invariance, symmetry, exact energy over all bitmasks, size discipline."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from poc import energy as en
from poc2 import scenes, tasks
from poc3 import TRAINING_SEEDS
from poc3 import features as ft
from poc3 import model as md

SEED = TRAINING_SEEDS[0]
STAGE1 = {c.name: (c.scene, c.regions, c.options) for c in scenes.stage1_cases()}
TASKS = {k: tasks.build(spec) for k, (spec, _, _) in tasks.regression_cases().items()}
ALL = {**STAGE1, **TASKS}
TOL = dict(rtol=1e-5, atol=1e-6)  # float32 pooling / GEMM reordering only


def seeded(pairwise: bool = True) -> md.EnergyModel:
    torch.manual_seed(SEED)
    return md.EnergyModel(pairwise=pairwise).eval()


@pytest.fixture(scope="module")
def model() -> md.EnergyModel:
    return seeded()


def coeffs(m, f):
    with torch.no_grad():
        return m.coefficients(f)


def remap(x: int, perm) -> int:
    """Bitmask x over the original options -> bitmask over options reordered as options[perm[k]]."""
    return sum(1 << k for k, p in enumerate(perm) if x >> p & 1)


# ------------------------------------------------------------ size and determinism

def test_model_size_discipline():
    n = md.n_parameters(seeded())
    assert n < md.MAX_PARAMS and md.n_parameters(seeded(pairwise=False)) < n
    assert all(len(m) == 3 and isinstance(m[1], torch.nn.GELU) for m in seeded().children())  # 2-layer MLPs


def test_seeded_initialisation_is_deterministic():
    f = ft.extract(*ALL["M4_envelope_coupling"])
    (q1, Q1), (q2, Q2) = coeffs(seeded(), f), coeffs(seeded(), f)
    assert torch.equal(q1, q2) and torch.equal(Q1, Q2)


# ------------------------------------------------------------ symmetry and invariance

@pytest.mark.parametrize("name", sorted(ALL))
def test_pair_symmetry_and_zero_diagonal(model, name):
    q, Q = coeffs(model, ft.extract(*ALL[name]))
    assert torch.equal(Q, Q.T) and torch.all(torch.diagonal(Q) == 0) and Q.abs().sum() > 0


@pytest.mark.parametrize("name", sorted(ALL))
def test_entity_permutation_invariance(model, name):
    scene, regions, options = ALL[name]
    q, Q = coeffs(model, ft.extract(scene, regions, options))
    rev = replace(scene, entities=scene.entities[::-1])        # scene-level reorder (affected remapped)
    q2, Q2 = coeffs(model, ft.extract(rev, regions, options))
    assert torch.allclose(q, q2, **TOL) and torch.allclose(Q, Q2, **TOL)
    f = ft.extract(scene, regions, options)                    # tensor-level reorder including the fixture row
    perm = np.random.default_rng(SEED).permutation(len(f.entities))
    inv = np.argsort(perm)
    g = replace(f, entities=f.entities[perm], affected=np.where(f.affected < 0, -1, inv[f.affected]))
    q3, Q3 = coeffs(model, g)
    assert torch.allclose(q, q3, **TOL) and torch.allclose(Q, Q3, **TOL)


@pytest.mark.parametrize("name", sorted(ALL))
def test_candidate_permutation_only_permutes_q_and_Q(model, name):
    scene, regions, options = ALL[name]
    perm = np.random.default_rng(SEED).permutation(len(options))
    q, Q = coeffs(model, ft.extract(scene, regions, options))
    q2, Q2 = coeffs(model, ft.extract(scene, regions, tuple(options[p] for p in perm)))
    t = torch.as_tensor(perm)
    assert torch.allclose(q2, q[t], **TOL) and torch.allclose(Q2, Q[t][:, t], **TOL)
    E, E2 = md.state_energies(q.double(), Q.double()), md.state_energies(q2.double(), Q2.double())
    assert torch.allclose(E2[[remap(x, perm) for x in range(len(E))]], E, **TOL)


def test_action_changes_the_energy(model):
    scene, regions, options = STAGE1["M1_independent"]
    other = replace(scene, motion=replace(scene.motion, goal=(0.25, 0.01, scene.motion.goal[2])))
    q, _ = coeffs(model, ft.extract(scene, regions, options))
    q2, _ = coeffs(model, ft.extract(other, regions, options))
    assert not torch.allclose(q, q2)


# ------------------------------------------------------------ exact energy over all bitmasks

@pytest.mark.parametrize("name", sorted(ALL))
def test_state_energies_equal_direct_polynomial(model, name):
    scene, regions, options = ALL[name]
    q, Q = (t.double() for t in coeffs(model, ft.extract(scene, regions, options)))
    cost = torch.tensor([float(o.cost) for o in options], dtype=torch.float64)
    E, EK = md.state_energies(q, Q), md.state_energies(q, Q, cost)
    P = len(q)
    for x, bits in enumerate(product((0, 1), repeat=P)):
        b = bits[::-1]  # bit p of x
        x_check = sum(v << p for p, v in enumerate(b))
        direct = sum(q[p] * b[p] for p in range(P)) + sum(Q[p, r] * b[p] * b[r] for p in range(P) for r in range(p + 1, P))
        assert x_check == x and abs(float(E[x] - direct)) <= 1e-12
        assert abs(float(EK[x] - E[x]) - sum(o.cost * v for o, v in zip(options, b))) <= 1e-12
    assert E[0] == 0 and len(E) == 2 ** P
    assert np.allclose(E.numpy(), en.qubo_energy(Q.numpy() / 2, q.numpy(), 0.0), rtol=0, atol=1e-12)  # PoC-1 form


def test_unary_model_and_zero_Q_reduce_exactly_to_unary(model):
    f = ft.extract(*ALL["storage_extraction"])
    q, Q = coeffs(seeded(pairwise=False), f)
    X = md.bit_matrix(len(q))
    assert torch.equal(Q, torch.zeros_like(Q))
    assert torch.equal(md.state_energies(q.double(), Q.double()), X @ q.double())
    q2, _ = coeffs(model, f)
    assert torch.equal(md.state_energies(q2.double(), torch.zeros(len(q2), len(q2), dtype=torch.float64)),
                       X @ q2.double())


def test_energy_is_differentiable_for_later_training():
    m = seeded()
    q, Q = m.coefficients(ft.extract(*ALL["articulated_opening"]))
    md.state_energies(q, Q).logsumexp(dim=0).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_no_entity_candidates_use_the_zero_embedding(model):
    f = ft.extract(*ALL["storage_insertion"])
    assert (f.affected == ft.NO_ENTITY).any()
    t = md.as_tensors(f)
    with torch.no_grad():
        h_e, h_s = model.scene_embedding(t["entities"], t["action"], t["moving"])
        h = model.candidate_embeddings(h_e, h_s, t["candidates"], t["affected"])
        p = int(np.flatnonzero(f.affected == ft.NO_ENTITY)[0])
        ref = model.phi_i(torch.cat([h_s, torch.zeros(md.HIDDEN), t["candidates"][p]]))
    assert torch.allclose(h[p], ref, **TOL)
