"""Fixture, asset and gripper sanity."""
import numpy as np
import pytest

from artrecourse.assets import analyze_object, load_manifest, selected_assets
from artrecourse.fixture import fixture_report


@pytest.fixture(scope="module")
def fx():
    return fixture_report()


def test_fixture_loads(fx):
    assert fx["fixture_id"] == "Dishwasher054"
    assert fx["checks"]["passed"]


def test_door_joint_genuinely_articulates(fx):
    assert fx["checks"]["door_rotation_deg_at_max"] > 30
    assert fx["checks"]["door_centre_displacement_m"] > 0.1


def test_rack_joint_genuinely_translates(fx):
    assert abs(fx["checks"]["upper_rack_translation_m"] - 0.40) < 1e-6
    assert fx["checks"]["clearance_open_door_vs_pulled_upper_rack_m"] > 0


def test_every_selected_mesh_loads():
    assets = selected_assets()
    assert len(assets) >= 30
    for a in assets.values():
        info = analyze_object(a)
        assert np.isfinite(info["mass"]) and info["mass"] > 0
        assert min(info["inertia_diag"]) > 0
        assert info["n_collision_geoms"] >= 1


def test_visual_collision_scales_agree():
    for rec in load_manifest()["objects"]:
        if rec["accepted"]:
            assert all(abs(r - 1) <= 0.08 for r in rec["visual_collision_extent_ratio"]), rec["asset_id"]
            assert rec["drop_test"]["passed"], rec["asset_id"]


def test_floating_gripper_matches_arm_hand():
    import mujoco

    from artrecourse.primitives import tcp_rotation
    from artrecourse.scene import Scene

    sc = Scene([], with_robot=True)
    m, d = sc.model, sc.data
    p, R = np.array([0.1, -0.5, 0.8]), tcp_rotation(0.3)
    sc.set_station("dishwasher_right")
    r = sc.kin.solve(p, R)
    assert r.ok
    sc.kin.fk(r.q, 0.04)
    sc.set_gripper(p, R=R, opening=0.04)
    mujoco.mj_kinematics(m, d)
    for nm in ("hand", "left_finger", "right_finger"):
        a, b = m.body("panda_" + nm).id, m.body("grip_" + nm).id
        assert np.abs(d.xpos[a] - d.xpos[b]).max() < 2e-3
