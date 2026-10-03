"""PoC-3 Stage 4.6C script tests: guards, frozen rules, gate machinery, repair rows, outputs, no seed 61."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2 import structure as st
from poc3 import feasibility as fz

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "s46c_feasibility_supervision.py"
_spec = importlib.util.spec_from_file_location("s46c_feasibility_supervision", SCRIPT)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


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


def test_guard_rejects_test_rows():
    recs = [{"split": "train"}, {"split": "val"}, {"split": "test"}]
    assert sc.guard(recs, [0, 1]) == [0, 1]
    with pytest.raises(RuntimeError, match="test split"):
        sc.guard(recs, [2])


def test_frozen_constants():
    assert sc.EXPECTED_COUNTS == {"U-EXACT": 288, "U-OVERLAP": 27, "P-TOP1": 45}
    assert (sc.MAX_DISAGREEMENT, sc.GATE_BA, sc.LOW_SAMPLE) == (0.01, 0.98, 30)


def test_outcome_rules_and_precedence():
    good = {"ba": 0.97, "r0": 0.95, "r1": 0.93, "vf": 0.93, "hit": 0.86, "dhit": 0.0, "dhit_ci_lo": -0.1}
    assert sc.outcome(good)["outcome"] == "S"
    assert sc.outcome(good | {"hit": 0.80, "dhit": 0.09, "dhit_ci_lo": 0.01})["outcome"] == "S"
    assert sc.outcome(good | {"vf": 0.80, "hit": 0.70})["outcome"] == "D"           # D precedes R
    assert sc.outcome(good | {"ba": 0.85})["outcome"] == "R"
    assert sc.outcome(good | {"ba": 0.93, "vf": 0.88})["outcome"] == "M"
    assert sc.outcome(good | {"ba": 0.85})["failed"] == ["S1"]


def test_no_predicted_feasible_state_repair_row():
    r = sc.repair_row(fz.NO_PREDICTED_FEASIBLE_STATE, {})
    assert r["no_feasible"] and not r["hit"] and not r["valid_feasible"] and r["excess_cost"] is None


def test_gate_machinery(tiny):
    g = sc.gate(tiny, epochs=3)
    assert set(g) == {"state_balanced_accuracy", "hits", "n", "passed", "runtime_s"} and g["n"] == len(tiny)
    assert g["passed"] == (g["state_balanced_accuracy"] >= 0.98 and g["hits"] * 20 >= 19 * len(tiny))


def test_paired_bootstrap_is_deterministic():
    a = {s: [{"hit": h} for h in np.random.default_rng(s).integers(0, 2, 40)] for s in (7, 17, 27)}
    b = {s: [{"hit": h} for h in np.random.default_rng(s + 3).integers(0, 2, 40)] for s in (7, 17, 27)}
    assert sc.paired(a, b, list(range(40)), "hit") == sc.paired(a, b, list(range(40)), "hit")
    assert sc.paired(a, b, list(range(10)), "hit")["low_sample_exploratory"]


def test_script_is_validation_only_writes_declared_outputs_and_never_generates_seed_61():
    src = SCRIPT.read_text().replace(sc.__doc__, "")
    assert '"test")' not in src and "generate_stream" not in src and "sample_spec" not in src and "ds.sample(" not in src
    saves = [ln for ln in src.splitlines() if "torch.save" in ln or "write_text" in ln]
    assert len(saves) == 4 and all("metrics.json" in ln or "torch.save" in ln for ln in saves)
    assert 'f"dense_f_seed{seed}.pt"' in src and '"future_audit_seed_61_generated": False' in src
    assert "make_model=fz.FeasibilityModel" in src and ".backward(" not in src and "AdamW(" not in src
