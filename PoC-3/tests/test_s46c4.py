"""PoC-3 Stage 4.6C.4 tests: frozen protocol, train-only FIT selection, guards, references, outputs."""

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
SCRIPT = REPO / "PoC-3" / "scripts" / "s46c4_fitted_generalization.py"
_spec = importlib.util.spec_from_file_location("s46c4_fitted_generalization", SCRIPT)
c4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c4)
OUT = REPO / "PoC-3" / "out"


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


def test_frozen_training_protocol():
    assert c4.SEEDS == (7, 17, 27) and c4.CFG == {"lr": 1e-3, "weight_decay": 1e-5, "batch": 32, "epochs": 700}
    assert 700 * 70 == 49_000 and (c4.FIT_BA, c4.FIT_RECALL, c4.RECORD_EVERY) == (0.98, 0.97, 25)
    loop = inspect.getsource(c4.run_fit_final)
    for banned in ("EarlyStopping", "patience", "val_", '"val"', "split_rows"):
        assert banned not in loop, banned


def test_fit_selection_uses_train_records_only():
    recs = [{"balanced_accuracy": .97, "recall_F0": .99, "recall_F1": .99}, {"balanced_accuracy": .99, "recall_F0": .96,
            "recall_F1": .99}, {"balanced_accuracy": .985, "recall_F0": .975, "recall_F1": .98}]
    assert [c4.is_fit(r) for r in recs] == [False, False, True]
    worker = inspect.getsource(c4._train_seed)
    assert "train_rows_checked" in worker and "val" not in worker.replace("eval", "")
    src = SCRIPT.read_text().replace(c4.__doc__, "")
    assert "# train labels only" in src and src.index("pool.map(_train_seed") < src.index("val = sc.dense_samples")
    assert src.index("torch.save(") < src.index("evaluate_dense(fz.load")         # states frozen before validation


def test_run_loop_matches_tr_train_and_freezes_fit_and_final(tiny):
    cfg = {"lr": 1e-3, "weight_decay": 1e-5, "batch": 4, "epochs": 4}
    r = c4.run_fit_final(tiny[:6], 7, cfg, record_every=2)
    ref = tr.train(tiny[:6], tiny[:6], 7, max_epochs=4, patience=4, batch=4, pairwise=True, loss_fn=fz.dense_loss,
                   name="dense_f_bce", make_model=fz.FeasibilityModel)
    assert r["train_bce_per_epoch"] == ref["history"]["train_dense_f_bce"]
    assert r["final"]["epoch"] == 4 and [x["epoch"] for x in r["records"]] == [2, 4]
    assert r["fit"] is None or r["fit"]["epoch"] in (2, 4)


def test_guards_categories_and_no_seed_61():
    with pytest.raises(RuntimeError):
        c4.c3.guard_train([{"split": "test"}], [0])
    with pytest.raises(RuntimeError):
        c4.sc.guard([{"split": "test"}], [0])
    assert c4.sc.EXPECTED_COUNTS == {"U-EXACT": 288, "U-OVERLAP": 27, "P-TOP1": 45}
    src = SCRIPT.read_text().replace(c4.__doc__, "")
    for banned in ('"test")', "generate_stream", "sample_spec", "ds.sample(", "calibrate_temperature", "threshold_logit ="):
        assert banned not in src, banned
    assert '"future_audit_seed_61_generated": False' in src and '"stage2_test_split_evaluated": False' in src


def test_dense_loss_and_threshold_unchanged(tiny):
    assert fz.THRESHOLD_LOGIT == 0.0
    torch.manual_seed(7)
    m = fz.FeasibilityModel()
    per = [torch.nn.functional.binary_cross_entropy_with_logits(fz.feasibility_logits(m, s), s.F) for s in tiny[:3]]
    assert torch.allclose(fz.dense_loss(m, tiny[:3]), torch.stack(per).mean())


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_model_source_unchanged_since_c3():
    try:
        diff = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", "091dcdc", "--", "PoC-3/src"],
                              check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        pytest.skip("commit 091dcdc not present")
    assert diff.stdout == ""


@pytest.mark.skipif(not (OUT / "s46c2" / "checkpoints" / "dense_f_seed7.pt").exists(), reason="references absent")
def test_references_load_strictly():
    models = c4.load_c2()
    assert set(models) == {7, 17, 27} and all(isinstance(m, fz.FeasibilityModel) for m in models.values())
    for seed in (7, 17, 27):
        rl.load(torch.load(OUT / "s46b" / "checkpoints" / f"rel_pair_shift_seed{seed}.pt")["state_dict"], "cf", "shift")


def test_outcome_rules_and_bootstrap_helper():
    crit_s, crit_d = {"outcome": "S"}, {"outcome": "D"}
    assert c4.outcome(False, crit_s, {"ba": .99, "vf": .99}, True) == "INSTABILITY"
    assert c4.outcome(True, crit_s, {"ba": .99, "vf": .99}, True) == "T"
    assert c4.outcome(True, crit_d, {"ba": .89, "vf": .95}, True) == "G"
    assert c4.outcome(True, crit_d, {"ba": .93, "vf": .80}, True) == "G"
    assert c4.outcome(True, crit_d, {"ba": .93, "vf": .88}, True) == "M"
    assert c4.outcome(True, crit_d, {"ba": .93, "vf": .88}, False) == "N"
    a = {s: [{"hit": h} for h in np.random.default_rng(s).integers(0, 2, 40)] for s in (7, 17, 27)}
    b = {s: [{"hit": h} for h in np.random.default_rng(s + 1).integers(0, 2, 40)] for s in (7, 17)}
    p = c4.paired(a, b, list(range(40)), "hit")
    assert p == c4.paired(a, b, list(range(40)), "hit") and not p["low_sample_exploratory"]
