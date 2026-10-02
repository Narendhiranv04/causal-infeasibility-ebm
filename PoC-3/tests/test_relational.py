"""PoC-3 Stage 4.5 relational-diagnostic tests: invariance, counterfactual context, leakage, reuse, determinism."""

import functools
import inspect
import itertools
import re
from dataclasses import replace
from pathlib import Path

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


# ------------------------------------------------------------ Stage 4.6B relational pair head

S45 = Path(__file__).resolve().parents[1] / "out" / "s45" / "checkpoints"


def paired_model(pair: str, zero_head: bool = False) -> rl.RelationalEnergy:
    torch.manual_seed(7)
    m = rl.RelationalEnergy("cf", pair)
    if zero_head:
        with torch.no_grad():
            m.f_2[2].weight.zero_()
            m.f_2[2].bias.zero_()
    return m.eval()


def unary_twin(pair_model) -> rl.RelationalEnergy:
    u = rl.RelationalEnergy("cf")
    u.load_state_dict({k: v for k, v in pair_model.state_dict().items() if k.split(".")[0] in ("phi_rel", "score")})
    return u.eval()


@pytest.mark.parametrize("pair", ["all", "shift"])
def test_unary_q_path_is_unchanged_by_the_pair_head(pair):
    m = paired_model(pair)
    t = inputs("storage_insertion")
    with torch.no_grad():
        assert torch.equal(m(**t)[0], unary_twin(m)(**t)[0])


@pytest.mark.skipif(not (S45 / "rel_cf_seed7.pt").exists(), reason="Stage-4.5 checkpoints not present")
def test_frozen_stage45_rel_cf_checkpoints_still_load_strictly():
    for seed in (7, 17, 27):
        rl.load(torch.load(S45 / f"rel_cf_seed{seed}.pt")["state_dict"], "cf")


@pytest.mark.parametrize("pair", ["all", "shift"])
@pytest.mark.parametrize("name", ["storage_insertion", "M4_envelope_coupling", "articulated_opening"])
def test_pair_Q_symmetric_zero_diagonal_and_permutation_equivariant(pair, name):
    m, t = paired_model(pair), inputs(name)
    with torch.no_grad():
        q, Q = m(**t)
        perm = torch.as_tensor(np.random.default_rng(17).permutation(len(t["candidates"])))
        q2, Q2 = m(**(t | {"candidates": t["candidates"][perm], "affected": t["affected"][perm]}))
    assert torch.equal(Q, Q.T) and torch.all(torch.diagonal(Q) == 0)
    assert torch.allclose(q2, q[perm], **TOL) and torch.allclose(Q2, Q[perm][:, perm], **TOL)


def test_pair_descriptor_is_symmetric():
    h, g0 = torch.randn(5, rl.PAIR_EMBED), torch.randn(rl.CONTEXT_DIM)
    p, r = torch.tensor([0, 1, 3]), torch.tensor([2, 4, 4])
    assert torch.equal(rl.RelationalEnergy.pair_descriptor(h, g0, p, r), rl.RelationalEnergy.pair_descriptor(h, g0, r, p))


def test_shift_mask_keeps_shift_pairs_and_zeros_relocation_pairs():
    t = inputs("storage_insertion")  # two relocations of one object, one other relocation, two shifts
    shift = (t["candidates"][:, rl.SHIFT] == 1).numpy()
    assert shift.sum() >= 2 and (~shift).sum() >= 2
    with torch.no_grad():
        Q_all, Q_shift = paired_model("all")(**t)[1], paired_model("shift")(**t)[1]
    for p, r in itertools.combinations(range(len(shift)), 2):
        if shift[p] or shift[r]:
            assert Q_shift[p, r] == Q_all[p, r] != 0          # SHIFT-relocation and SHIFT-SHIFT kept
        else:
            assert Q_shift[p, r] == 0 and Q_all[p, r] != 0    # relocation-relocation masked only in P2


def test_pair_variants_have_identical_parameter_counts_below_500k():
    n_all, n_shift = md.n_parameters(rl.RelationalEnergy("cf", "all")), md.n_parameters(rl.RelationalEnergy("cf", "shift"))
    assert n_all == n_shift < 500_000 and rl.RelationalEnergy("cf", "all").pairwise
    assert not rl.RelationalEnergy("cf").pairwise


@pytest.mark.parametrize("pair", ["all", "shift"])
def test_zero_pair_head_reduces_exactly_to_rel_cf_unary_energy(pair, structured):
    m = paired_model(pair, zero_head=True)
    u = unary_twin(m)
    for s in structured:
        assert np.array_equal(tr.energies(m, s), tr.energies(u, s))


def test_pair_terms_are_counted_once_in_stage4_energies(structured):
    m = paired_model("all")
    s = structured[2]
    with torch.no_grad():
        q, Q = m(**s.inputs)
    X = s.bits.numpy()
    manual = X @ q.double().numpy() + X @ s.cost.numpy()
    manual += sum(Q[p, r].item() * X[:, p] * X[:, r] for p in range(len(q)) for r in range(p + 1, len(q)))
    assert np.allclose(tr.energies(m, s), manual, atol=1e-9)


def test_gradients_reach_unary_and_pair_heads(structured):
    m = paired_model("shift").train()
    tr.structured_loss(m, structured).backward()
    for head in (m.score, m.phi_pair, m.f_2, m.phi_rel):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_tiny_pairwise_relational_training_is_deterministic(structured):
    kw = dict(max_epochs=3, patience=3, batch=4, pairwise=True, loss_fn=tr.structured_loss, name="set_nll",
              make_model=functools.partial(rl.RelationalEnergy, "cf", "shift"))
    a, b = tr.train(structured, structured[:3], 7, **kw), tr.train(structured, structured[:3], 7, **kw)
    assert a["history"] == b["history"] and all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    ma, mb = rl.load(a["state_dict"], "cf", "shift"), rl.load(b["state_dict"], "cf", "shift")
    assert all(np.array_equal(tr.energies(ma, s), tr.energies(mb, s)) for s in structured)
