"""PoC-3 Stage 4.6C.1 tests: free-polynomial design, masks, convex fit, probe fidelity, guards, outputs."""

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch

from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2 import structure as st
from poc3 import feasibility as fz
from poc3 import relational as rl
from poc3 import train as tr

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "s46c1_gate_diagnosis.py"
_spec = importlib.util.spec_from_file_location("s46c1_gate_diagnosis", SCRIPT)
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)


@pytest.fixture(scope="module")
def tiny():
    out = []
    for case in [(c.scene, c.regions, c.options) for c in scenes.stage1_cases()] + \
                [tasks.build(s) for s, _, _ in tasks.regression_cases().values()]:
        t = orc.repair_table(case[0], case[2])
        states = np.flatnonzero((t.M == 1) & (t.V == 1))
        out.append(fz.make_dense_sample(*case, states, t.F[states], np.ones(len(states), bool), st.exact_decision(t),
                                        [o.cost for o in case[2]], {"scene_id": f"tiny{len(out)}", "family": "x",
                                                                    "intent": "repairable"}))
    return out


def test_design_matrix_is_one_linear_and_pair_products():
    bits = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=float)
    Phi = gd.design(bits, [(0, 2), (1, 2)])
    assert np.array_equal(Phi, [[1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1]])


def test_shift_mask_excludes_exactly_non_shift_pairs_and_unrestricted_has_all():
    cand = np.zeros((5, 14))
    cand[[1, 3], rl.SHIFT] = 1
    masked, full = gd.pair_list(cand, True), gd.pair_list(cand, False)
    assert full == list(combinations(range(5), 2))
    assert masked == [(p, r) for p, r in full if p in (1, 3) or r in (1, 3)]
    assert (0, 2) not in masked and (0, 4) not in masked and (2, 4) not in masked


def test_free_logits_equal_manual_polynomial_and_fit_is_deterministic(tiny):
    s = tiny[3]
    pairs = gd.pair_list(s.inputs["candidates"].numpy(), True)
    bits = s.bits.numpy()
    Phi = gd.design(bits, pairs)
    w = np.random.default_rng(0).normal(size=Phi.shape[1])
    manual = w[0] + bits @ w[1:1 + bits.shape[1]] + sum(w[1 + bits.shape[1] + k] * bits[:, p] * bits[:, r]
                                                         for k, (p, r) in enumerate(pairs))
    assert np.allclose(Phi @ w, manual)
    y = s.F.numpy()
    assert np.array_equal(gd.fit_free(Phi, y), gd.fit_free(Phi, y))


def test_lp_separability_certificate():
    Phi = gd.design(np.array([[0.0], [1.0]]), [])
    assert gd.lp_separable(Phi, np.array([1.0, 0.0])) and not gd.lp_separable(
        gd.design(np.array([[0.0], [0.0]]), []), np.array([1.0, 0.0]))


def test_probe_train_reproduces_tr_train_trajectory(tiny):
    probe = gd.probe_train(tiny[:4], 1e-3, 1e-5, epochs=4, batch=2, record_every=2)
    ref = tr.train(tiny[:4], tiny[:4], 7, max_epochs=4, patience=4, batch=2, pairwise=True, loss_fn=fz.dense_loss,
                   name="dense_f_bce", make_model=fz.FeasibilityModel)
    assert probe["train_bce"] == ref["history"]["train_dense_f_bce"]
    assert [r["epoch"] for r in probe["records"]] == [2, 4] and probe["best_state"] is not None


def test_guard_admits_only_the_twenty_train_gate_scenes(monkeypatch):
    recs = [{"split": sp, "family": f, "intent": i} for sp in ("val", "test", "train") for f in tr.ds.FAMILIES
            for i in ("repairable",) * 6]
    rows = gd.gate_rows_checked(recs)
    assert rows == tr.gate_rows(recs) and len(rows) == 20 and all(recs[j]["split"] == "train" for j in rows)
    for leaked in (0, 30):                                                  # a validation row, then a test row
        monkeypatch.setattr(gd.tr, "gate_rows", lambda r, leaked=leaked: rows[:19] + [leaked])
        with pytest.raises(RuntimeError):
            gd.gate_rows_checked(recs)


def test_margins_and_threshold():
    m = gd.margins(np.array([-1.0, 0.5, 0.05]), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(m, [1.0, 0.5, -0.05]) and fz.THRESHOLD_LOGIT == 0.0
    s = gd.margin_summary(m)
    assert s["frac_negative"] == 1 / 3 and s["frac_abs_below_0.1"] == 1 / 3


def test_outcome_rules():
    assert gd.outcome({"realizable": False}, None, None) == "Q"
    assert gd.outcome({"realizable": True}, 17, True) == "L"
    assert gd.outcome({"realizable": True}, 18, True) == "O"
    assert gd.outcome({"realizable": True}, 20, False) == "J"


def test_script_sources_no_oracle_coefficients_no_checkpoint_no_validation():
    src = SCRIPT.read_text().replace(gd.__doc__, "")
    for banned in ("compatible_mobius", "reconstruct(", ".G[", "alpha", "beta", "torch.save", '"val")', '"test")',
                   "THRESHOLD_LOGIT =", "split_rows("):
        assert banned not in src, banned
    writes = [ln for ln in src.splitlines() if "write_text" in ln]
    assert len(writes) == 1 and "metrics.json" in writes[0]
