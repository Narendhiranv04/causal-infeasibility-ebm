"""PoC-3 Stage 4.6C feasibility-model tests: frozen base reuse, bias, polynomial, threshold, cost, loss, decision."""

import functools
import inspect
import itertools
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2 import structure as st
from poc3 import feasibility as fz
from poc3 import features as ft
from poc3 import model as md
from poc3 import relational as rl
from poc3 import train as tr

TOL = dict(rtol=1e-5, atol=1e-6)
S46B = Path(__file__).resolve().parents[1] / "out" / "s46b" / "checkpoints"
CASES = {c.name: (c.scene, c.regions, c.options) for c in scenes.stage1_cases()}
CASES |= {k: tasks.build(spec) for k, (spec, _, _) in tasks.regression_cases().items()}


def sample(name: str) -> fz.DenseSample:
    sc, rg, op = CASES[name]
    t = orc.repair_table(sc, op)
    states = np.flatnonzero((t.M == 1) & (t.V == 1))
    stable = np.ones(len(states), bool)
    return fz.make_dense_sample(sc, rg, op, states, t.F[states], stable, st.exact_decision(t), [o.cost for o in op])


@pytest.fixture(scope="module")
def samples():
    return [sample(n) for n in CASES]


def seeded() -> fz.FeasibilityModel:
    torch.manual_seed(7)
    return fz.FeasibilityModel().eval()


@pytest.mark.skipif(not (S46B / "rel_pair_shift_seed7.pt").exists(), reason="Stage-4.6B checkpoints not present")
def test_stage46b_checkpoints_still_load_strictly():
    for seed in (7, 17, 27):
        ck = torch.load(S46B / f"rel_pair_shift_seed{seed}.pt")
        rl.load(ck["state_dict"], "cf", "shift")
        fz.FeasibilityModel().base.load_state_dict(ck["state_dict"], strict=True)


def test_base_is_the_unchanged_rel_cf_pair_shift_and_terms_match_it(samples):
    m = seeded()
    assert isinstance(m.base, rl.RelationalEnergy) and (m.base.variant, m.base.pair) == ("cf", "shift")
    assert md.n_parameters(m) - md.n_parameters(m.base) == md.n_parameters(md.mlp(rl.CONTEXT_DIM, 1, md.HIDDEN))
    with torch.no_grad():
        for s in samples:
            _, q, Q = m.terms(**s.inputs)
            q0, Q0 = m.base(**s.inputs)
            assert torch.equal(q, q0) and torch.equal(Q, Q0)
            assert torch.equal(Q, Q.T) and torch.all(torch.diagonal(Q) == 0)


def test_scene_bias_depends_only_on_g0():
    m, s = seeded(), sample("M1_independent")
    with torch.no_grad():
        b = m.terms(**s.inputs)[0]
        cand = s.inputs | {"candidates": s.inputs["candidates"] + 0.1}
        assert torch.equal(m.terms(**cand)[0], b)                              # candidates do not enter b
        g0 = m.base.contexts(**s.inputs)[0]
        assert torch.equal(m.bias(g0).squeeze(-1), b)
        moved = s.inputs["entities"].clone()
        moved[0, 0] += 0.2
        assert not torch.equal(m.terms(**(s.inputs | {"entities": moved}))[0], b)


def test_logit_is_the_polynomial_counting_each_pair_once_without_cost(samples):
    m = seeded()
    for s in samples[:4]:
        with torch.no_grad():
            b, q, Q = m.terms(**s.inputs)
            lg = fz.feasibility_logits(m, s).numpy()
        X = s.bits.numpy()
        manual = float(b) + X @ q.double().numpy()
        manual += sum(Q[p, r].item() * X[:, p] * X[:, r] for p, r in itertools.combinations(range(X.shape[1]), 2))
        assert np.allclose(lg, manual, atol=1e-9)


def test_shift_mask_is_exact():
    m, s = seeded(), sample("storage_insertion")
    shift = (s.inputs["candidates"][:, rl.SHIFT] == 1).numpy()
    with torch.no_grad():
        Q = m.terms(**s.inputs)[2]
    for p, r in itertools.combinations(range(len(shift)), 2):
        assert (Q[p, r] != 0) == bool(shift[p] or shift[r])


def test_candidate_permutation_equivariance_and_entity_permutation_invariance():
    m, s = seeded(), sample("M4_envelope_coupling")
    perm = np.random.default_rng(17).permutation(len(s.cost))
    t = s.inputs
    with torch.no_grad():
        b, q, Q = m.terms(**t)
        b2, q2, Q2 = m.terms(**(t | {"candidates": t["candidates"][perm], "affected": t["affected"][perm]}))
        ep = torch.as_tensor(np.random.default_rng(7).permutation(len(t["entities"])))
        inv = torch.argsort(ep)
        b3, q3, Q3 = m.terms(**(t | {"entities": t["entities"][ep],
                                     "affected": torch.where(t["affected"] < 0, -1, inv[t["affected"].clamp(min=0)])}))
    pt = torch.as_tensor(perm)
    assert torch.allclose(b2, b, **TOL) and torch.allclose(q2, q[pt], **TOL) and torch.allclose(Q2, Q[pt][:, pt], **TOL)
    assert torch.allclose(b3, b, **TOL) and torch.allclose(q3, q, **TOL) and torch.allclose(Q3, Q, **TOL)


def test_threshold_is_fixed_at_logit_zero():
    assert fz.THRESHOLD_LOGIT == 0.0
    assert list(fz.predicted_infeasible([-1e-9, 0.0, 1e-9])) == [False, True, True]


def test_cost_changes_only_the_selected_repair_never_F_hat():
    s = sample("M2_placement_competition")
    lg = np.where(np.arange(len(s.states)) % 2 == 0, -1.0, 1.0)              # every other state predicted feasible
    lg[0] = 1.0                                                              # empty state predicted infeasible
    feas = np.flatnonzero(lg < 0)
    a = fz.derive_repair(lg, s)
    K = s.bits.numpy() @ s.cost
    assert a == int(s.states[feas[np.argmin(K[feas])]])                    # min cost, lowest index on ties
    target = int(s.states[feas[-1]])                                          # make another feasible state cheapest
    assert target != a
    cost = np.where((target >> np.arange(len(s.cost))) & 1, 0.0, 5.0)
    s2 = replace(s, cost=cost)
    m = seeded()
    with torch.no_grad():
        assert torch.equal(fz.feasibility_logits(m, s), fz.feasibility_logits(m, s2))   # l(x) and F_hat unchanged
    b = fz.derive_repair(lg, s2)
    K2 = s2.bits.numpy() @ s2.cost
    assert b != a and K2[list(s.states).index(b)] == 0.0 and b in set(s.states[feas].tolist())


def test_no_predicted_feasible_state_is_exact():
    s = sample("M1_independent")
    assert fz.derive_repair(np.zeros(len(s.states)), s) is fz.NO_PREDICTED_FEASIBLE_STATE is None


def test_dense_loss_is_scene_balanced_and_excludes_unstable_states(samples):
    m = seeded()
    batch = samples[:3]
    per = [torch.nn.functional.binary_cross_entropy_with_logits(fz.feasibility_logits(m, s), s.F) for s in batch]
    assert torch.allclose(fz.dense_loss(m, batch), torch.stack(per).mean())
    s = batch[0]
    unstable = s.stable.clone()
    unstable[0] = False
    flipped = s.F.clone()
    flipped[0] = 1 - flipped[0]
    a = fz.dense_loss(m, [replace(s, stable=unstable)])
    b = fz.dense_loss(m, [replace(s, stable=unstable, F=flipped)])
    assert torch.equal(a, b)                                                  # unstable state's label is ignored
    st0 = fz.state_metrics(np.zeros(len(s.states)), replace(s, stable=unstable))
    assert st0["n_states"] == len(s.states) - 1


def test_state_metrics_on_a_known_prediction():
    s = sample("M1_independent")
    y = s.F.numpy()
    perfect = np.where(y == 1, 2.0, -2.0)
    r = fz.state_metrics(perfect, s)
    assert r["recall_F0"] == r["recall_F1"] == r["balanced_accuracy"] == 1.0 and r["false_feasible"] == 0.0
    allf = fz.state_metrics(np.full(len(y), -1.0), s)
    assert allf["recall_F0"] == 1.0 and allf["recall_F1"] == 0.0 and allf["balanced_accuracy"] == 0.5


def test_model_forward_takes_only_the_feature_contract_and_imports_no_oracle():
    assert list(inspect.signature(fz.FeasibilityModel.terms).parameters)[1:] == \
        ["entities", "action", "moving", "candidates", "affected"]
    src = inspect.getsource(fz).replace(fz.__doc__, "")
    assert re.search(r"^\s*(import|from)\s+(mujoco|poc\.oracle|poc2|poc\.envelope|poc\.mj_scene)", src, re.M) is None
    for banned in ("repair_table", "sweep(", "signed_distance", "conflict_3d", "category", "family", "scene_id"):
        assert banned not in src, banned
    assert "F" not in inspect.signature(fz.FeasibilityModel.forward).parameters


def test_tiny_dense_training_is_deterministic(samples):
    kw = dict(max_epochs=3, patience=3, batch=4, pairwise=True, loss_fn=fz.dense_loss, name="dense_f_bce",
              make_model=fz.FeasibilityModel)
    a, b = tr.train(samples, samples[:3], 7, **kw), tr.train(samples, samples[:3], 7, **kw)
    assert a["history"] == b["history"] and all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    assert a["history"]["train_dense_f_bce"][-1] < a["history"]["train_dense_f_bce"][0]
    ma, mb = fz.load(a["state_dict"]), fz.load(b["state_dict"])
    with torch.no_grad():
        assert all(torch.equal(fz.feasibility_logits(ma, s), fz.feasibility_logits(mb, s)) for s in samples)


def test_feature_contract_is_unchanged():
    f = ft.extract(*CASES["M2_placement_competition"])
    assert f.entities.shape[1] == 9 and f.action.shape == (8, 9) and f.candidates.shape[1] == 14
    assert functools.reduce(lambda a, b: a * b, f.action.shape) == 72
