"""PoC-3 Stage 3 training tests: protocol, scene-balanced soft BCE, raw thresholding, determinism."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as fnn

from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2 import structure as st
from poc3 import dataset as ds
from poc3 import train as tr

OUT = Path(__file__).resolve().parents[1] / "out" / "s2"


@pytest.fixture(scope="module")
def tiny() -> list[tr.Sample]:
    cases = [(c.scene, c.regions, c.options) for c in scenes.stage1_cases()]
    cases += [tasks.build(spec) for spec, _, _ in tasks.regression_cases().values()]
    return [tr.make_sample(sc, rg, op, st.exact_decision(orc.repair_table(sc, op)), {"k": k})
            for k, (sc, rg, op) in enumerate(cases)]


def test_protocol_is_fixed():
    assert (tr.LR, tr.WEIGHT_DECAY, tr.MAX_EPOCHS, tr.PATIENCE, tr.SEEDS) == (1e-3, 1e-5, 200, 20, (7, 17, 27))
    assert tr.THRESHOLD == 0.5 and tr.BATCH_SCENES == 32
    src = inspect.getsource(tr).replace(tr.__doc__, "")  # code only, not the protocol prose
    for banned in ("pos_weight", "focal", "state_energies", "lambda_M", "test_set"):
        assert banned not in src, banned
    assert "test" not in inspect.signature(tr.train).parameters


def test_samples_use_only_the_feature_contract(tiny):
    for s in tiny:
        assert set(s.inputs) == {"entities", "action", "moving", "candidates", "affected"}
        assert s.y.shape == (len(s.inputs["candidates"]),) and float(s.y.min()) >= 0 and float(s.y.max()) <= 1


def test_loss_is_mean_over_own_P_then_equal_over_scenes(tiny):
    torch.manual_seed(7)
    model = tr.load_model(tr.md.EnergyModel(pairwise=False).state_dict())
    batch = tiny[:4]
    per_scene = [fnn.binary_cross_entropy_with_logits(tr.logits(model, s), s.y) for s in batch]
    loss = tr.scene_balanced_bce(model, batch)
    assert torch.allclose(loss, torch.stack(per_scene).mean())
    pooled = fnn.binary_cross_entropy_with_logits(torch.cat([tr.logits(model, s) for s in batch]),
                                                  torch.cat([s.y for s in batch]))
    assert len({len(s.y) for s in batch}) > 1 and not torch.allclose(loss, pooled)  # not candidate-weighted


def test_early_stopping_selects_lowest_validation_loss():
    es, best = tr.EarlyStopping(patience=2), []
    for epoch, v in enumerate([1.0, 0.8, 0.9, 0.8, 0.85]):
        best.append(es.update(epoch, v))
        if es.stop:
            break
    assert best == [True, True, False, False] and es.best_epoch == 1 and es.best == 0.8 and es.stop


def test_prediction_is_the_raw_threshold_without_repair(tiny):
    s = tiny[1]  # M2: choice groups exist, so the all-ones state violates M
    torch.manual_seed(7)
    model = tr.md.EnergyModel(pairwise=False)
    with torch.no_grad():
        model.f_u[2].bias.fill_(10.0)  # every pi_p ~ 1
    pi, x = tr.predict(model.eval(), s)
    assert x == 2 ** len(pi) - 1  # returned as predicted, even though M(x) = 0
    t = orc.repair_table(scenes.stage1_cases()[1].scene, scenes.stage1_cases()[1].options)
    assert t.M[x] == 0
    torch.manual_seed(17)
    pi, x = tr.predict(tr.md.EnergyModel(pairwise=False).eval(), s)
    assert x == sum(1 << p for p in range(len(pi)) if pi[p] >= 0.5)


def test_tiny_training_is_deterministic(tiny):
    kw = dict(max_epochs=3, patience=2, batch=4)
    a, b = tr.train(tiny, tiny[:4], 7, **kw), tr.train(tiny, tiny[:4], 7, **kw)
    c = tr.train(tiny, tiny[:4], 17, **kw)
    assert a["history"] == b["history"] and a["best_epoch"] == b["best_epoch"]
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    assert not all(torch.equal(a["state_dict"][k], c["state_dict"][k]) for k in a["state_dict"])
    ma, mb = tr.load_model(a["state_dict"]), tr.load_model(b["state_dict"])
    for s in tiny:
        pa, xa = tr.predict(ma, s)
        pb, xb = tr.predict(mb, s)
        assert np.array_equal(pa, pb) and xa == xb
    assert torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_training_reduces_the_training_loss(tiny):
    run = tr.train(tiny, tiny, 7, max_epochs=40, patience=40, batch=4)
    y = [s.y.double().clamp(1e-12, 1 - 1e-12) for s in tiny]   # soft-target BCE floor = target entropy
    floor = float(np.mean([float(-(t * t.log() + (1 - t) * (1 - t).log()).mean()) for t in y]))
    first, last = run["history"]["train_bce"][0], run["history"]["train_bce"][-1]
    assert last - floor < 0.75 * (first - floor)


@pytest.mark.skipif(not (OUT / "labels.npz").exists(), reason="Stage-2 dataset not generated")
def test_dataset_samples_decode_soft_targets_exactly():
    records, arrays = ds.load(OUT)
    rows = tr.split_rows(records, "val")[:8] + tr.split_rows(records, "test")[-4:]
    for j, s in zip(rows, tr.samples_from_dataset(records, arrays, rows)):
        opt = ds.unpack_states(arrays["optimal"][j], int(arrays["P"][j]))
        assert s.optimal == opt and np.allclose(s.y.numpy(), tr.soft_target(opt, int(arrays["P"][j])))
        assert s.meta["scene_id"] == records[j]["scene_id"] and "scene_id" not in s.inputs
    assert len(tr.split_rows(records, "train")) == 2240 and len(tr.split_rows(records, "test")) == 480


# ------------------------------------------------------------ Stage 4: exact structured set likelihood

def manual(E_bits, cost=None, target=(0,), P=2):
    states = np.arange(2 ** P)
    bits = torch.as_tensor((states[:, None] >> np.arange(P)) & 1, dtype=torch.float64)
    t = torch.as_tensor(np.isin(states, target))
    c = torch.zeros(P, dtype=torch.float64) if cost is None else torch.as_tensor(cost, dtype=torch.float64)
    return tr.StructuredSample({}, c, states, bits, t, frozenset(target), {})


@pytest.fixture(scope="module")
def tiny_structured():
    cases = [(c.scene, c.regions, c.options) for c in scenes.stage1_cases()]
    cases += [tasks.build(spec) for spec, _, _ in tasks.regression_cases().values()]
    out = []
    for sc, rg, op in cases:
        t = orc.repair_table(sc, op)
        adm = {int(x) for x in np.flatnonzero((t.M == 1) & (t.V == 1))}
        out.append(tr.make_structured_sample(sc, rg, op, adm, st.exact_decision(t), [o.cost for o in op]))
    return out


def test_structured_nll_matches_a_manual_calculation():
    s = manual(None, target=(0,))
    q, Q = torch.tensor([1.0, 2.0]), torch.tensor([[0.0, -0.5], [-0.5, 0.0]])
    E = tr.admissible_energies(q, Q, s)
    assert torch.allclose(E, torch.tensor([0.0, 1.0, 2.0, 2.5], dtype=torch.float64))  # pair counted once
    expected = 0.0 + np.log(1 + np.exp(-1) + np.exp(-2) + np.exp(-2.5))
    assert abs(float(tr.set_nll(E, s.target)) - expected) < 1e-12


def test_tied_optima_get_uniform_target_weight():
    s = manual(None, target=(1, 2))
    E = torch.tensor([0.3, 1.0, 2.0, 0.0], dtype=torch.float64)
    logp = -E - torch.logsumexp(-E, 0)
    assert abs(float(tr.set_nll(E, s.target)) + float(logp[[1, 2]].mean())) < 1e-12


def test_cost_is_added_exactly_and_non_unit():
    s = manual(None, cost=[2.0, 5.0], target=(0,))
    E = tr.admissible_energies(torch.zeros(2), torch.zeros(2, 2), s)
    assert torch.equal(E, torch.tensor([0.0, 2.0, 5.0, 7.0], dtype=torch.float64))


def test_only_admissible_states_enter_partition_and_inference(tiny_structured):
    s = tiny_structured[1]  # M2: choice and static constraints exclude states
    full = 2 ** len(s.cost)
    assert len(s.states) < full and list(s.states) == sorted(s.states)
    torch.manual_seed(7)
    model = tr.md.EnergyModel(pairwise=True)
    with torch.no_grad():
        q, Q = model(**s.inputs)
        E_all = tr.md.state_energies(q.double(), Q.double(), s.cost)
    E = tr.admissible_energies(q, Q, s)
    assert torch.allclose(E, E_all[torch.as_tensor(s.states)], atol=1e-12)
    inadm = np.setdiff1d(np.arange(full), s.states)
    E_low = E_all.clone()
    E_low[torch.as_tensor(inadm)] = -1e6  # very attractive inadmissible states change nothing
    assert torch.equal(E_low[torch.as_tensor(s.states)], E)
    top, learned = tr.exact_inference(E.numpy(), s)
    assert top in set(s.states) and learned <= set(s.states.tolist())


def test_pairwise_loss_reduces_exactly_to_unary_when_Q_is_zero(tiny_structured):
    torch.manual_seed(7)
    pair = tr.md.EnergyModel(pairwise=True)
    with torch.no_grad():
        pair.f_2[2].weight.zero_()
        pair.f_2[2].bias.zero_()
    unary = tr.md.EnergyModel(pairwise=False)
    unary.load_state_dict({k: v for k, v in pair.state_dict().items() if not k.startswith("f_2")})
    assert torch.equal(tr.structured_loss(pair, tiny_structured), tr.structured_loss(unary, tiny_structured))


def test_gradients_reach_q_and_Q(tiny_structured):
    torch.manual_seed(7)
    model = tr.md.EnergyModel(pairwise=True)
    tr.structured_loss(model, tiny_structured).backward()
    for head in (model.f_u, model.f_2):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_exact_inference_ties_break_to_the_lowest_state_index():
    s = manual(None, target=(0,))
    top, learned = tr.exact_inference(np.array([1.0, 0.0, 0.0, 0.0 + 5e-7]), s)
    assert top == 1 and learned == {1, 2, 3}
    top, learned = tr.exact_inference(np.array([1.0, 0.0, 2e-6, 0.0]), s)
    assert top == 1 and learned == {1, 3}


def test_temperature_calibration_uses_only_the_given_validation_energies():
    E = [np.array([0.0, 2.0, 3.0]), np.array([1.0, 0.0, 4.0])]
    t = [np.array([True, False, False]), np.array([False, True, False])]
    cal = tr.calibrate_temperature(E, t)
    curve = np.mean([tr.nll_curve(e, m, tr.T_GRID) for e, m in zip(E, t)], axis=0)
    assert cal["index"] == int(np.argmin(curve)) and cal["T"] == tr.T_GRID[cal["index"]]
    assert len(tr.T_GRID) == 241 and np.isclose(tr.T_GRID[0], 0.05) and np.isclose(tr.T_GRID[-1], 20.0)
    flat = tr.calibrate_temperature([np.zeros(3)], [np.array([True, False, False])])  # every T ties
    assert flat["index"] == 0 and flat["at_boundary"]
    assert list(inspect.signature(tr.calibrate_temperature).parameters) == ["val_energies", "val_targets"]


def test_checkpoint_is_the_lowest_validation_loss_epoch(tiny_structured):
    run = tr.train(tiny_structured, tiny_structured[:3], 7, max_epochs=6, patience=6, batch=4, pairwise=True,
                   loss_fn=tr.structured_loss, name="set_nll")
    val = run["history"]["val_set_nll"]
    assert run["best_epoch"] == int(np.argmin(val)) and run["best_val_set_nll"] == min(val)


def test_tiny_structured_training_is_deterministic(tiny_structured):
    kw = dict(max_epochs=3, patience=3, batch=4, pairwise=True, loss_fn=tr.structured_loss, name="set_nll")
    a, b = tr.train(tiny_structured, tiny_structured[:3], 7, **kw), tr.train(tiny_structured, tiny_structured[:3], 7, **kw)
    assert a["history"] == b["history"]
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    ma, mb = tr.load_model(a["state_dict"], True), tr.load_model(b["state_dict"], True)
    assert all(np.array_equal(tr.energies(ma, s), tr.energies(mb, s)) for s in tiny_structured)


def test_planted_quadratic_needs_and_is_fit_by_the_pairwise_model(tiny_structured):
    planted = []  # optimum = {empty, {0, 1}} over the full domain: no unary energy can tie both
    for s in tiny_structured[:4]:
        P = len(s.cost)
        planted.append(tr.StructuredSample(s.inputs, torch.zeros(P, dtype=torch.float64), np.arange(2 ** P),
                                           torch.as_tensor((np.arange(2 ** P)[:, None] >> np.arange(P)) & 1,
                                                           dtype=torch.float64),
                                           torch.as_tensor(np.isin(np.arange(2 ** P), [0, 3])), frozenset({0, 3}), {}))
    kw = dict(max_epochs=150, patience=150, batch=4, loss_fn=tr.structured_loss, name="set_nll")
    pair, unary = tr.train(planted, planted, 7, pairwise=True, **kw), tr.train(planted, planted, 7, pairwise=False, **kw)
    assert unary["best_val_set_nll"] >= np.log(4) - 1e-6        # factorized bound
    assert pair["best_val_set_nll"] < 1.0                        # approaches log 2


def test_overfit_gate_selection_and_rule():
    recs = [{"split": sp, "family": f, "intent": i} for sp in ("val", "train") for f in tr.ds.FAMILIES
            for i in ("repairable", "negative", "repairable", "repairable", "repairable", "repairable", "repairable")]
    rows = tr.gate_rows(recs)
    assert len(rows) == 20 and all(recs[j]["split"] == "train" and recs[j]["intent"] == "repairable" for j in rows)
    assert [recs[j]["family"] for j in rows] == [f for f in tr.ds.FAMILIES for _ in range(5)]
    assert rows == sorted(rows)
    assert (tr.GATE_SCENES_PER_FAMILY, tr.GATE_EPOCHS, tr.GATE_BATCH, tr.GATE_MIN_HITS) == (5, 400, 20, 19)
    assert tr.ENERGY_TIE_TOL == 1e-6
