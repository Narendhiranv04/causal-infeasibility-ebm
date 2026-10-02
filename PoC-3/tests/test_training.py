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
