"""PoC-3 Stage 4.6C.2 tests: only the gate budget changed; full training, loss, threshold, guards unchanged."""

import importlib.util
import inspect
import shutil
import subprocess
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

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "PoC-3" / "scripts" / "s46c2_dense_f_full.py"
_spec = importlib.util.spec_from_file_location("s46c2_dense_f_full", SCRIPT)
c2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c2)
S46B = REPO / "PoC-3" / "out" / "s46b" / "checkpoints"
FROZEN_SRC = ("feasibility.py", "relational.py", "train.py", "metrics.py", "features.py", "model.py", "dataset.py")


@pytest.fixture(scope="module")
def tiny():
    out = []
    for case in [(c.scene, c.regions, c.options) for c in scenes.stage1_cases()] + \
                [tasks.build(s) for s, _, _ in tasks.regression_cases().values()]:
        t = orc.repair_table(case[0], case[2])
        states = np.flatnonzero((t.M == 1) & (t.V == 1))
        out.append(fz.make_dense_sample(*case, states, t.F[states], np.ones(len(states), bool), st.exact_decision(t),
                                        [o.cost for o in case[2]]))
    return out


def test_corrected_gate_is_1000_steps_at_lr_1e3_on_the_original_20_scenes():
    src = SCRIPT.read_text()
    assert c2.GATE_EPOCHS == 1000 and tr.GATE_BATCH == 20 and tr.LR == 1e-3 and tr.WEIGHT_DECAY == 1e-5
    assert "sc.gate(gate_set, epochs=GATE_EPOCHS)" in src and "gd.gate_rows_checked(records)" in src
    assert "probe_train(gate_set, tr.LR, tr.WEIGHT_DECAY, GATE_EPOCHS, tr.GATE_BATCH" in src
    assert (tr.GATE_SCENES_PER_FAMILY, tr.GATE_MIN_HITS, c2.sc.GATE_BA) == (5, 19, 0.98)


def test_full_training_is_the_unchanged_stage46c_protocol():
    assert (tr.MAX_EPOCHS, tr.PATIENCE, tr.LR, tr.WEIGHT_DECAY, tr.BATCH_SCENES) == (200, 20, 1e-3, 1e-5, 32)
    train_src = inspect.getsource(c2.sc._train)
    assert "max_epochs" not in train_src and "lr" not in train_src.replace("make_model", "")
    assert "pool.map(sc._train" in SCRIPT.read_text()
    src = SCRIPT.read_text().replace(c2.__doc__, "")
    for leak in ("3e-3", "3e-4", "SETTINGS", "2000"):
        assert leak not in src, leak


def test_dense_loss_threshold_and_repair_rule_are_the_frozen_ones(tiny):
    torch.manual_seed(7)
    m = fz.FeasibilityModel()
    per = [torch.nn.functional.binary_cross_entropy_with_logits(fz.feasibility_logits(m, s), s.F) for s in tiny[:3]]
    assert torch.allclose(fz.dense_loss(m, tiny[:3]), torch.stack(per).mean()) and fz.THRESHOLD_LOGIT == 0.0
    assert c2.sc.repair_row(fz.NO_PREDICTED_FEASIBLE_STATE, {})["no_feasible"]


@pytest.mark.skipif(not (S46B / "rel_pair_shift_seed7.pt").exists(), reason="Stage-4.6B checkpoints not present")
def test_reference_checkpoints_load_strictly():
    for seed in (7, 17, 27):
        ck = torch.load(S46B / f"rel_pair_shift_seed{seed}.pt")
        assert ck["architecture"] == c2.sc.REF_ARCH
        rl.load(ck["state_dict"], "cf", "shift")


def test_categories_guards_and_no_seed_61():
    assert c2.sc.EXPECTED_COUNTS == {"U-EXACT": 288, "U-OVERLAP": 27, "P-TOP1": 45}
    with pytest.raises(RuntimeError):
        c2.sc.guard([{"split": "test"}], [0])
    with pytest.raises(RuntimeError):
        c2.au.guard([{"split": "test", "intent": "repairable"}], [0])
    src = SCRIPT.read_text().replace(c2.__doc__, "")
    for banned in ('"test")', "generate_stream", "sample_spec", "ds.sample(", "calibrate_temperature"):
        assert banned not in src, banned
    assert '"future_audit_seed_61_generated": False' in src and '"stage2_test_split_evaluated": False' in src


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_no_model_source_file_changed_since_stage46c1():
    try:
        diff = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", "d0c0895", "--"]
                              + [f"PoC-3/src/poc3/{f}" for f in FROZEN_SRC], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        pytest.skip("commit d0c0895 not present")
    assert diff.stdout == ""


def test_tiny_full_training_path_is_deterministic(tiny):
    kw = dict(max_epochs=3, patience=3, batch=4, pairwise=True, loss_fn=fz.dense_loss, name="dense_f_bce",
              make_model=fz.FeasibilityModel)
    a, b = tr.train(tiny, tiny[:3], 7, **kw), tr.train(tiny, tiny[:3], 7, **kw)
    assert a["history"] == b["history"] and all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])


def test_helpers():
    s = type("S", (), {"bits": torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.float64)})()
    m = c2.card_masks(s)
    assert [list(m[k]) for k in c2.CARD] == [[True, False, False], [False, True, False], [False, False, True]]
    rows = {sd: [{"x_hat": 0}, {"x_hat": 5}, {"x_hat": None}, {"x_hat": 0}] for sd in (7, 17, 27)}
    r = c2.negative_rates(rows, [0, 1, 2, 3])
    assert (r["correct_empty"]["mean"], r["false_repair"]["mean"], r["no_predicted_feasible"]["mean"]) == (0.5, 0.25, 0.25)
    recs = [{"epoch": 1, "ba": 0.5, "hits": 3}, {"epoch": 2, "ba": 0.99, "hits": 19}]
    assert c2.first_epoch(recs, lambda r: r["ba"] >= 0.98) == 2 and c2.first_epoch(recs, lambda r: r["ba"] > 1) is None
