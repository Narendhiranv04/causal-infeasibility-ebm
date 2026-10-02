"""PoC-3 Stage 4.5 relational-diagnostic tests: invariance, counterfactual context, leakage, reuse, determinism."""

import functools
import inspect
import re
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch

from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2 import structure as st
from poc3 import features as ft
from poc3 import model as md
from poc3 import relational as rl
from poc3 import train as tr

TOL = dict(rtol=1e-5, atol=1e-6)
CASES = {c.name: (c.scene, c.regions, c.options) for c in scenes.stage1_cases()}
CASES |= {k: tasks.build(spec) for k, (spec, _, _) in tasks.regression_cases().items()}


def seeded(variant: str) -> rl.RelationalEnergy:
    torch.manual_seed(7)
    return rl.RelationalEnergy(variant).eval()


def inputs(name: str) -> dict:
    return md.as_tensors(ft.extract(*CASES[name]))


def ctx(model, t):
    with torch.no_grad():
        return model.contexts(**t)


@pytest.fixture(scope="module")
def structured():
    out = []
    for sc, rg, op in CASES.values():
        t = orc.repair_table(sc, op)
        adm = {int(x) for x in np.flatnonzero((t.M == 1) & (t.V == 1))}
        out.append(tr.make_structured_sample(sc, rg, op, adm, st.exact_decision(t), [o.cost for o in op]))
    return out


def test_rotation_from_6d_matches_the_quaternion_rotation():
    rng = np.random.default_rng(107)
    for _ in range(50):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        got = rl.rotation_from_6d(torch.as_tensor(ft.rot6d(quat)))
        assert np.allclose(got.numpy(), R.reshape(3, 3), atol=1e-12)


def test_parameter_counts_identical_and_small():
    n1, n2 = md.n_parameters(rl.RelationalEnergy("static")), md.n_parameters(rl.RelationalEnergy("cf"))
    assert n1 == n2 < 500_000 and not rl.RelationalEnergy("cf").pairwise


@pytest.mark.parametrize("variant", rl.VARIANTS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_entity_permutation_invariance(variant, name):
    m, t = seeded(variant), inputs(name)
    perm = torch.as_tensor(np.random.default_rng(7).permutation(len(t["entities"])))
    inv = torch.argsort(perm)
    t2 = t | {"entities": t["entities"][perm], "affected": torch.where(t["affected"] < 0, -1, inv[t["affected"].clamp(min=0)])}
    (g0, g, _, _), (g0b, gb, _, _) = ctx(m, t), ctx(m, t2)
    with torch.no_grad():
        q, q2 = m(**t)[0], m(**t2)[0]
    assert torch.allclose(g0, g0b, **TOL) and torch.allclose(g, gb, **TOL) and torch.allclose(q, q2, **TOL)


@pytest.mark.parametrize("variant", rl.VARIANTS)
def test_candidate_permutation_only_permutes_scores(variant):
    m, t = seeded(variant), inputs("M4_envelope_coupling")
    perm = torch.as_tensor(np.random.default_rng(17).permutation(len(t["candidates"])))
    with torch.no_grad():
        q, q2 = m(**t)[0], m(**(t | {"candidates": t["candidates"][perm], "affected": t["affected"][perm]}))[0]
    assert torch.allclose(q2, q[perm], **TOL)


def test_rel_static_uses_g0_for_every_candidate():
    g0, g, _, _ = ctx(seeded("static"), inputs("storage_insertion"))
    assert torch.equal(g, g0.expand_as(g))


def test_zero_shift_gives_g0_and_nonzero_shift_changes_context():
    m, t = seeded("cf"), inputs("storage_insertion")
    shift = torch.nonzero(t["candidates"][:, rl.SHIFT] == 1).flatten()
    assert len(shift) and torch.any(t["candidates"][shift][:, rl.DELTA] != 0)
    g0, g, _, _ = ctx(m, t)
    assert not torch.allclose(g[shift], g0.expand(len(shift), -1), **TOL)   # asymmetric scene
    zero = t["candidates"].clone()
    zero[shift[:, None], torch.arange(rl.DELTA.start, rl.DELTA.stop)] = 0.0
    g0z, gz, _, _ = ctx(m, t | {"candidates": zero})
    assert torch.allclose(gz[shift], g0z.expand(len(shift), -1), **TOL)


def test_relocation_changes_context_only_through_that_entity_pose():
    m, t = seeded("cf"), inputs("M1_independent")
    g0, g, e0, aff = ctx(m, t)
    p = int(torch.nonzero(aff >= 0).flatten()[0])
    i = int(aff[p])
    ent = t["entities"][t["entities"][:, rl.FIXTURE] == 0]
    pos, R = t["action"][:, :3], rl.rotation_from_6d(t["action"][:, 3:])
    with torch.no_grad():
        moved = m.embed(t["candidates"][p, rl.PROPOSED][None], ent[i:i + 1, 3:6], ent[i:i + 1, 6:9], pos, R, t["moving"])
        manual = m.pool(torch.cat([e0[:i], moved, e0[i + 1:]]))
    assert torch.allclose(g[p], manual, **TOL) and not torch.allclose(g[p], g0, **TOL)
    same = t["candidates"].clone()
    same[p, rl.PROPOSED] = ent[i, 0:3]                                      # "relocate" to where it already is
    assert torch.allclose(ctx(m, t | {"candidates": same})[1][p], g0, **TOL)


@pytest.mark.parametrize("variant", rl.VARIANTS)
def test_target_fixture_is_not_in_the_relation_pool(variant):
    name = "articulated_opening"
    m, t = seeded(variant), inputs(name)
    fix = torch.nonzero(t["entities"][:, rl.FIXTURE] == 1).flatten()
    assert len(fix) == 1
    moved = t["entities"].clone()
    moved[fix, 0:6] += 0.3
    with torch.no_grad():
        assert torch.equal(m(**t)[0], m(**(t | {"entities": moved}))[0])
        assert len(m.contexts(**t)[2]) == len(t["entities"]) - 1


def test_model_inputs_are_only_the_feature_contract_and_no_oracle_is_touched():
    assert list(inspect.signature(rl.RelationalEnergy.forward).parameters)[1:] == \
        ["entities", "action", "moving", "candidates", "affected"]
    src = inspect.getsource(rl)
    assert re.search(r"^\s*(import|from)\s+(mujoco|poc\.oracle|poc2|poc\.envelope|poc\.mj_scene)", src, re.M) is None
    for banned in ("repair_table", "sweep(", "signed_distance", "conflict_3d", "assess(", "logsumexp", "family",
                   "scene_id", "status", "B0", "alpha", "beta", "seed"):
        assert banned not in src.replace(rl.__doc__, ""), banned


@pytest.mark.parametrize("variant", rl.VARIANTS)
def test_structured_loss_and_inference_are_the_stage4_implementations(variant, structured):
    m = seeded(variant)
    with torch.no_grad():
        manual = torch.stack([tr.set_nll(tr.admissible_energies(m(**s.inputs)[0], torch.zeros(len(s.cost), len(s.cost)), s),
                                         s.target) for s in structured]).mean()
        assert torch.equal(tr.structured_loss(m, structured), manual)
    E = tr.energies(m, structured[0])
    assert tr.exact_inference(E, structured[0])[0] in set(structured[0].states.tolist())


def test_model_hook_keeps_the_default_training_bit_identical(structured):
    kw = dict(max_epochs=3, patience=3, batch=4, pairwise=True, loss_fn=tr.structured_loss, name="set_nll")
    a = tr.train(structured, structured[:3], 7, **kw)
    b = tr.train(structured, structured[:3], 7, make_model=functools.partial(md.EnergyModel, pairwise=True), **kw)
    assert a["history"] == b["history"] and all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])


@pytest.mark.parametrize("variant", rl.VARIANTS)
def test_tiny_relational_training_is_deterministic_and_learns(variant, structured):
    kw = dict(max_epochs=4, patience=4, batch=4, loss_fn=tr.structured_loss, name="set_nll",
              make_model=functools.partial(rl.RelationalEnergy, variant))
    a, b = tr.train(structured, structured, 7, **kw), tr.train(structured, structured, 7, **kw)
    assert a["history"] == b["history"] and all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    assert a["history"]["train_set_nll"][-1] < a["history"]["train_set_nll"][0]
    ma, mb = rl.load(a["state_dict"], variant), rl.load(b["state_dict"], variant)
    assert all(np.array_equal(tr.energies(ma, s), tr.energies(mb, s)) for s in structured)


def test_stage1_contract_unchanged_for_relational_inputs():
    f = ft.extract(*CASES["M2_placement_competition"])
    assert f.entities.shape[1] == 9 and f.action.shape == (8, 9) and f.candidates.shape[1] == 14
    assert replace(f, affected=f.affected).affected.dtype == np.int64
