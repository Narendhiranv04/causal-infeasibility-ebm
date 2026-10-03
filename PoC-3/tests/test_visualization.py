"""Tests for the read-only scene / infeasibility visualizer: oracle-checked labels, frozen geometry reuse, CLI parsing."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from poc2 import oracle as orc
from poc2 import scenes, tasks

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import viz_utils as vz  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex, cli = _load("make_visual_explainer"), _load("visualize_scene")
M4 = scenes.envelope_coupling()


def test_m4_counterfactual_states_match_the_designed_mechanism():
    F, blk = {}, {}
    for name, x in ex.M4_STATES.items():
        info = vz.analyse(vz.apply_state(M4.scene, M4.options, x))
        F[x], blk[x] = info["a"].F, info["a"].blockers
    assert F == ex.M4_EXPECT == {0: 1, 0b10: 1, 0b01: 1, 0b11: 0}
    assert blk[0] == ("jamb",) and blk[0b10] == ("jamb",) and blk[0b01] == ("nb",) and blk[0b11] == ()
    assert [M4.options[p].option_id for p in vz.bits(0b11, len(M4.options))] == ["shift", "nb->r1"]


@pytest.mark.parametrize("build", [lambda: (M4.scene, M4.options)] +
                         [lambda k=k: tasks.build(tasks.regression_cases()[k][0])[0::2] for k in tasks.regression_cases()])
def test_rendered_poses_are_the_oracle_sweep_samples(build):
    scene, _ = build()
    info = vz.analyse(scene)
    assert len(info["poses"]) == len(info["sweep"].taus)
    assert info["a"].F == orc.evaluate(scene).F
    for eid, k in info["tau_idx"].items():
        j = info["a"].ids.index(eid)
        assert info["a"].c[j] > 0 and k == int(np.argmin(info["sweep"].distances[:, j]))


def test_highlight_and_label_checks_refuse_misleading_panels():
    info = vz.analyse(M4.scene)
    vz.verify(info, 1, "ok")
    with pytest.raises(AssertionError):
        vz.verify(info, 0, "mislabelled")
    assert set(info["a"].blockers) == {i for i, c in zip(info["a"].ids, info["a"].c) if c > 0}


def test_relocation_and_shift_rendering_follow_apply_option():
    after = vz.apply_state(M4.scene, M4.options, 0b11)
    ref = orc.apply_option(orc.apply_option(M4.scene, M4.options[0].intervention), M4.options[1].intervention)
    assert after == ref
    r1 = next(r for r in M4.regions if r.region_id == "r1")
    c = vz.centre(after, "nb")
    assert abs(c[0] - r1.center[0]) < 1e-9 and abs(c[1] - r1.center[1]) < 1e-9
    p0, p1 = vz.analyse(M4.scene)["poses"][0][0], vz.analyse(after)["poses"][0][0]
    assert np.allclose(np.asarray(p1) - np.asarray(p0), M4.options[0].intervention.params)


def test_cli_state_parsing_and_option_listing():
    assert cli.parse_state(None, "0,3", 5) == 9 and cli.parse_state(5, None, 5) == 5 and cli.parse_state(None, None, 5) is None
    for bad in ((32, None), (None, "5")):
        with pytest.raises(SystemExit):
            cli.parse_state(*bad, 5)
    with pytest.raises(SystemExit):
        cli.parse_state(1, "0", 5)
    assert vz.option_text(M4.options[0]).startswith("SHIFT") and vz.option_text(M4.options[1]) == "move nb -> region r1"


@pytest.mark.skipif(not (SCRIPTS.parent / "out" / "s2" / "labels.npz").exists(), reason="Stage-2 dataset absent")
def test_cli_refuses_test_split_scenes():
    with pytest.raises(SystemExit, match="test-split"):
        cli.load_scene("test-ms-000")


def test_before_repair_after_smoke(tmp_path):
    case = scenes.stage1_cases()[0]
    t = orc.repair_table(case.scene, case.options)
    x = int(np.flatnonzero((t.M == 1) & (t.V == 1) & (t.F == 0))[0])
    meta = ex.before_repair_after(tmp_path / "m1.png", "smoke", case.scene, case.regions, case.options, x, ["line"], side=False)
    assert (tmp_path / "m1.png").stat().st_size > 10_000 and meta["before"]["F"] == 1 and meta["after"]["F"] == 0


def test_tooling_is_read_only():
    for f in ("viz_utils.py", "make_visual_explainer.py", "visualize_scene.py"):
        src = (SCRIPTS / f).read_text()
        for banned in ("torch", "tr.train", "generate_stream", "sample_spec", 'split_rows(records, "test")'):
            assert banned not in src, (f, banned)
