"""PoC-3 Stage 4.6B tests: guards, implementation gate, frozen selection / interpretation rules, checkpoints."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2 import structure as st
from poc3 import dataset as ds
from poc3 import relational as rl
from poc3 import train as tr

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "s46b_relational_pair.py"
_spec = importlib.util.spec_from_file_location("s46b_relational_pair", SCRIPT)
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)


@pytest.fixture(scope="module")
def structured():
    cases = [(c.scene, c.regions, c.options) for c in scenes.stage1_cases()]
    cases += [tasks.build(spec) for spec, _, _ in tasks.regression_cases().values()]
    out = []
    for sc, rg, op in cases:
        t = orc.repair_table(sc, op)
        adm = {int(x) for x in np.flatnonzero((t.M == 1) & (t.V == 1))}
        out.append(tr.make_structured_sample(sc, rg, op, adm, st.exact_decision(t), [o.cost for o in op]))
    return out


def test_training_guard_rejects_test_rows():
    recs = [{"split": "train"}, {"split": "val"}, {"split": "test"}]
    assert sb.train_guard(recs, [0, 1]) == [0, 1]
    with pytest.raises(RuntimeError, match="test split"):
        sb.train_guard(recs, [0, 2])
    with pytest.raises(RuntimeError, match="test split"):
        sb.au.guard([{"split": "test", "intent": "repairable"}], [0])


def test_implementation_gate_machinery(structured):
    g = sb.gate(structured, epochs=3)
    assert set(g) == {"hits", "n", "passed", "best_epoch", "runtime_s"} and g["n"] == len(structured)
    assert g["passed"] == (g["hits"] * 20 >= 19 * len(structured))
    assert (tr.GATE_EPOCHS, tr.GATE_BATCH, tr.GATE_MIN_HITS) == (400, 20, 19)


def test_frozen_selection_rule():
    d = lambda u: {"U-EXACT": u}  # noqa: E731
    both = {"REL-CF-PAIR-ALL": d(0.0), "REL-CF-PAIR-SHIFT": d(-0.01)}
    assert sb.select(both, {"REL-CF-PAIR-ALL": 0.70, "REL-CF-PAIR-SHIFT": 0.60}) == "REL-CF-PAIR-ALL"
    assert sb.select(both, {"REL-CF-PAIR-ALL": 0.60, "REL-CF-PAIR-SHIFT": 0.70}) == "REL-CF-PAIR-SHIFT"
    assert sb.select(both, {"REL-CF-PAIR-ALL": 0.705, "REL-CF-PAIR-SHIFT": 0.70}) == "REL-CF-PAIR-SHIFT"  # < 0.01
    one = {"REL-CF-PAIR-ALL": d(-0.03), "REL-CF-PAIR-SHIFT": d(0.0)}
    assert sb.select(one, {"REL-CF-PAIR-ALL": 0.99, "REL-CF-PAIR-SHIFT": 0.5}) == "REL-CF-PAIR-SHIFT"
    assert sb.select({n: d(-0.05) for n in sb.VARIANTS}, {n: 0.9 for n in sb.VARIANTS}) is None


def test_frozen_interpretation_rules():
    r = sb.rules(0.12, [0.03, 0.2], -0.01, 0.04, [0.01, 0.07])
    assert r == {"RELATIONAL_PAIRWISE_HELPS_P_TOP1": True, "unary_strength_preserved": True, "overall_improvement": True}
    assert not sb.rules(0.12, [-0.01, 0.2], 0.0, 0.0, [-0.1, 0.1])["RELATIONAL_PAIRWISE_HELPS_P_TOP1"]
    assert not sb.rules(0.09, [0.01, 0.2], 0.0, 0.0, [-0.1, 0.1])["RELATIONAL_PAIRWISE_HELPS_P_TOP1"]
    assert not sb.rules(0.2, [0.1, 0.3], -0.021, 0.0, [-0.1, 0.1])["unary_strength_preserved"]
    assert not sb.rules(0.2, [0.1, 0.3], 0.0, 0.05, [-0.01, 0.1])["overall_improvement"]


def test_checkpoint_payload_has_required_fields():
    run = {"state_dict": rl.RelationalEnergy("cf", "shift").state_dict(), "name": "REL-CF-PAIR-SHIFT", "seed": 7,
           "best_val_set_nll": 1.0, "calibration": {"T": 1.1}}
    ck = sb.checkpoint_payload(run)
    assert {"state_dict", "variant", "seed", "dataset_digest", "best_val_set_nll", "calibrated_temperature",
            "architecture", "hyperparameters"} <= set(ck)
    assert ck["dataset_digest"] == ds.STAGE2_DIGEST and rl.PAIR_ARCHITECTURE in ck["architecture"]
    rl.load(ck["state_dict"], "cf", ck["pair"])


def test_script_evaluates_validation_only_and_writes_only_declared_outputs():
    src = SCRIPT.read_text().replace(sb.__doc__, "")
    assert '"test")' not in src and "rows_evaluated_per_split" in src
    saves = [ln for ln in src.splitlines() if "torch.save" in ln or "write_text" in ln]
    assert all("rel_pair_" in ln or "metrics.json" in ln for ln in saves) and len(saves) == 3
    assert sb.EXPECTED_COUNTS == {"U-EXACT": 288, "U-OVERLAP": 27, "P-TOP1": 45}
    assert "make_model=make(" in inspect.getsource(sb._train)
