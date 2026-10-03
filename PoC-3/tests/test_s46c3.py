"""PoC-3 Stage 4.6C.3 tests: train-only guards, structural audit reuse, frozen run settings, fidelity, outputs."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2 import structure as st
from poc3 import feasibility as fz
from poc3 import train as tr

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "s46c3_full_train_fit.py"
_spec = importlib.util.spec_from_file_location("s46c3_full_train_fit", SCRIPT)
c3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c3)


@pytest.fixture(scope="module")
def tiny():
    out = []
    for k, case in enumerate([(c.scene, c.regions, c.options) for c in scenes.stage1_cases()] +
                             [tasks.build(s) for s, _, _ in tasks.regression_cases().values()]):
        t = orc.repair_table(case[0], case[2])
        states = np.flatnonzero((t.M == 1) & (t.V == 1))
        out.append(fz.make_dense_sample(*case, states, t.F[states], np.ones(len(states), bool), st.exact_decision(t),
                                        [o.cost for o in case[2]], {"scene_id": f"t{k}", "family": "make_space",
                                                                    "intent": "repairable"}))
    return out


def _records(n_train=2240, n_rep=1680):
    recs = [{"split": "train", "intent": "repairable" if i < n_rep else "negative"} for i in range(n_train)]
    return recs + [{"split": "val", "intent": "repairable"}, {"split": "test", "intent": "repairable"}]


def test_only_train_rows_and_exact_composition():
    recs = _records()
    rows = c3.train_rows_checked(recs)
    assert len(rows) == 2240 and all(recs[j]["split"] == "train" for j in rows)
    for bad in (2240, 2241):                                                 # validation row, test row
        with pytest.raises(RuntimeError):
            c3.guard_train(recs, [0, bad])
    with pytest.raises(RuntimeError):
        c3.train_rows_checked(_records(n_rep=1679))


def test_structural_audit_reuses_c1_design_and_dense_F_only(tiny):
    s = tiny[6]
    cand = s.inputs["candidates"].numpy()
    args = (s.bits.numpy(), cand, s.F.numpy(), s.stable.numpy(), True)
    assert c3._separable(args) == c3.gd.lp_separable(
        c3.gd.design(s.bits.numpy(), c3.gd.pair_list(cand, True)), s.F.numpy())
    assert c3.gd.pair_list(cand, False) != c3.gd.pair_list(cand, True) or not (cand[:, 1] == 0).any()
    src = SCRIPT.read_text()
    assert "args[i][:4] + (False,) for i in failed" in src                  # unrestricted only on masked failures
    for banned in ("compatible_mobius", ".G[", "alpha", "beta"):
        assert banned not in src.replace(c3.__doc__, ""), banned


def test_frozen_run_settings_and_no_third_setting():
    assert c3.RUN_A == {"seed": 7, "lr": 1e-3, "weight_decay": 1e-5, "batch": 32, "epochs": 715}
    assert c3.RUN_B == c3.RUN_A | {"lr": 3e-3}
    src = SCRIPT.read_text().replace(c3.__doc__, "")
    assert src.count("run(samples, RUN_") == 2 and "B = run(samples, RUN_B) if not A[\"full_train_fit\"] else None" in src
    for banned in ("3e-4", "1e-2", "patience", "EarlyStopping", "val_", "split_rows(records, \"val\")"):
        assert banned not in src, banned
    assert 715 * 70 == 50_050


def test_run_loop_reproduces_tr_train_and_records_without_early_stopping(tiny):
    cfg = {"seed": 7, "lr": 1e-3, "weight_decay": 1e-5, "batch": 4, "epochs": 4}
    r = c3.run(tiny[:6], cfg, record_every=2)
    ref = tr.train(tiny[:6], tiny[:6], 7, max_epochs=4, patience=4, batch=4, pairwise=True, loss_fn=fz.dense_loss,
                   name="dense_f_bce", make_model=fz.FeasibilityModel)
    assert r["train_bce_per_epoch"] == ref["history"]["train_dense_f_bce"]
    assert [x["epoch"] for x in r["records"]] == [2, 4] and r["total_steps"] == 4 * 2
    src = inspect.getsource(c3.run)
    assert all(tok not in src for tok in ("val_set", "val_", '"val"', "EarlyStopping", "patience"))


def test_plateau_rule():
    recs = [{"step": s, "epoch": s // 70, "balanced_accuracy": ba, "bce": b}
            for s, ba, b in ((30_000, 0.9, 0.2), (40_250, 0.92, 0.18), (50_050, 0.923, 0.175))]
    p = c3.plateau(recs, 50_050)
    assert p["tail_epochs"] == [575, 715] and p["plateau"] and p["label"] == "PLATEAU"
    recs[-1]["bce"] = 0.15
    assert not c3.plateau(recs, 50_050)["plateau"]


def test_outcome_rule():
    yes, no = {"full_train_fit": True}, {"full_train_fit": False}
    assert c3.outcome(0.98, no, None, {}) == "SF"
    assert c3.outcome(1.0, yes, None, {}) == "O"
    assert c3.outcome(1.0, no, yes, {"A": {"plateau": True}}) == "H"
    assert c3.outcome(1.0, no, no, {"A": {"plateau": True}, "B": {"plateau": True}}) == "J"
    assert c3.outcome(1.0, no, no, {"A": {"plateau": True}, "B": {"plateau": False}}) == "U"


def test_threshold_loss_and_outputs_are_frozen(tiny):
    assert fz.THRESHOLD_LOGIT == 0.0
    torch.manual_seed(7)
    m = fz.FeasibilityModel()
    per = [torch.nn.functional.binary_cross_entropy_with_logits(fz.feasibility_logits(m, s), s.F) for s in tiny[:3]]
    assert torch.allclose(fz.dense_loss(m, tiny[:3]), torch.stack(per).mean())
    src = SCRIPT.read_text().replace(c3.__doc__, "")
    assert "torch.save" not in src and "generate_stream" not in src and "sample_spec" not in src
    writes = [ln for ln in src.splitlines() if "write_text" in ln]
    assert len(writes) == 2 and all("metrics.json" in ln for ln in writes)
    assert '"validation_evaluated": False' in src and '"future_audit_seed_61_generated": False' in src


def test_scene_eval_and_aggregate(tiny):
    torch.manual_seed(7)
    per = c3.evaluate(fz.FeasibilityModel(), tiny[:3])
    agg = c3.aggregate(per, tiny[:3])
    assert agg["n"] == 3 and 0 <= agg["balanced_accuracy"] <= 1 and "repairable_hit" in agg
    assert set(c3.card_masks(tiny[0])) == {"x0", "x1", "x2plus"}
