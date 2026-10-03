"""Primitives, placements, sweep determinism and the exact distance check."""
import numpy as np

from artrecourse.exactgeom import exact_signed_distance, hull_normals
from artrecourse.placements import check_placement, footprint
from artrecourse.primitives import COARSE_SPACING, relocate


def _box(c, h):
    return np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * h + c


def test_exact_distance_known_cases():
    a = _box([0, 0, 0], [0.05, 0.05, 0.05])
    b = _box([0.13, 0, 0], [0.05, 0.05, 0.05])          # 3 cm gap
    c = _box([0.08, 0.01, 0], [0.05, 0.05, 0.05])       # 2 cm overlap along x
    na, nb, nc = hull_normals(a), hull_normals(b), hull_normals(c)
    assert abs(exact_signed_distance(a, b, na, nb) - 0.03) < 1e-4
    assert abs(exact_signed_distance(a, c, na, nc) + 0.02) < 1e-4


def test_deterministic_trajectories_reproduce(problems):
    pb, _ = problems["V3"]
    iv = pb.ivs[0]
    meta = pb.scene.objects[iv.obj].asset.meta
    cat = pb.scene.objects[iv.obj].asset.category
    args = (iv.obj, meta, cat, iv.src.pos, iv.src.yaw, iv.placement.pos, iv.placement.yaw,
            pb.sup[iv.src.support].z, pb.sup[iv.placement.support].z, iv.grasp)
    t1, t2 = relocate(*args, spacing=COARSE_SPACING), relocate(*args, spacing=COARSE_SPACING)
    assert np.array_equal(t1.grip_pos, t2.grip_pos) and np.array_equal(t1.tau, t2.tau)
    assert np.array_equal(t1.carried[iv.obj][0], t2.carried[iv.obj][0])
    assert t1.tau[0] == 0 and abs(t1.tau[-1] - 1) < 1e-9 and np.all(np.diff(t1.tau) >= 0)


def test_placements_stay_inside_support(problems):
    for pb, _ in problems.values():
        for o in pb.obj_keys:
            ok, why, _ = check_placement(pb.scene, pb.sup, o, pb.poses[o][0])
            assert ok, (o, why)
        for iv in pb.ivs:
            if iv.admissible:
                ok, why, _ = check_placement(pb.scene, pb.sup, iv.obj, iv.placement)
                assert ok, (iv.id, why)
                lo, hi = footprint(pb.scene.objects[iv.obj].asset.meta, iv.placement.pos, iv.placement.yaw)
                s = pb.sup[iv.placement.support]
                assert s.x[0] - 0.002 <= lo[0] and hi[0] <= s.x[1] + 0.002


def test_original_and_destination_pose_differ(problems):
    for pb, _ in problems.values():
        for iv in pb.ivs:
            if iv.admissible:
                assert np.linalg.norm(iv.placement.pos - iv.src.pos) > 0.01 or abs(iv.placement.yaw - iv.src.yaw) > 0.05


def test_swept_collision_and_blocker_identity_deterministic(problems):
    pb, res = problems["V3"]
    iv = pb.iv_by_id["move_bottle_to_counter_buffer_far_right"]
    ms = pb.eng.moving_set(iv.traj)
    g, p, R = pb.eng.object_geoms_at("box", *pb.pose_tuple("box", 0))
    a = pb.eng.profile(ms, g, p, R)
    b = pb.eng.profile(ms, g, p, R)
    assert a.min_dist == b.min_dist and a.tau_star == b.tau_star and a.part_star == b.part_star
    assert a.status == "conflict"
    ok, bl = pb.exec_info(iv, pb.s0)
    assert not ok and [o for o, _ in bl] == ["box"]


def test_apply_intervention_produces_exact_pose(problems):
    import mujoco

    from artrecourse.assets import body_pose_from_canonical

    pb, _ = problems["V3"]
    for iv in pb.ivs:
        s = pb.apply(pb.s0, iv)
        assert s[pb.obj_index(iv.obj)] == iv.pose_idx
        pb.place_state(s)
        mujoco.mj_kinematics(pb.scene.model, pb.scene.data)
        bid = pb.scene.idx.obj_body[iv.obj]
        bpos, _ = body_pose_from_canonical(pb.scene.objects[iv.obj].asset.meta, iv.placement.pos, iv.placement.yaw)
        assert np.allclose(pb.scene.data.xpos[bid], bpos, atol=1e-9)
