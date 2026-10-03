"""RoboCasa dishwasher fixtures: loading, naming and articulation checks.

The fixture MJCF is loaded unmodified from the RoboCasa lightwheel asset archive (same
file RoboCasa's ``Dishwasher`` fixture class consumes). We only (a) make file paths
absolute, (b) give its unnamed collision boxes stable names, (c) move visual geoms to the
render group and (d) record the region helper boxes before removing them.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import CACHE, ROOT
from .assets import COLLISION_GROUP, VISUAL_GROUP, _absolutize, load_asset_config

FIXTURE_ID = load_asset_config()["fixture"]["id"]
PRIMITIVES: dict = {}   # fixture id -> {geom name: (body, type, pos, quat, size)} before convexification


def fixture_xml(fid: str | None = None) -> Path:
    fx = load_asset_config()["fixture"]
    fid = fid or fx["id"]
    return CACHE / "robocasa" / fx["archive"] / "fixtures" / "dishwashers" / fid / "model.xml"


def load_fixture_spec(fid: str | None = None) -> tuple[mujoco.MjSpec, dict]:
    """Returns (spec, regions) where regions maps region name -> (body, pos, size)."""
    fid = fid or FIXTURE_ID
    path = fixture_xml(fid)
    spec = mujoco.MjSpec.from_file(str(path))
    _absolutize(spec, path.parent)
    regions = {}
    counters: dict[str, int] = {}
    for g in list(spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)):
        body = g.parent.name or "root"
        if g.classname.name == "region" or (g.name or "").startswith("reg_"):
            regions[g.name] = (body, np.array(g.pos, float), np.array(g.size, float))
            spec.delete(g)
            continue
        if g.contype == 0 and g.conaffinity == 0:
            g.group = VISUAL_GROUP
            g.density = 0.0
            continue
        k = counters.get(body, 0)
        counters[body] = k + 1
        if not g.name:
            g.name = f"{body}_c{k}"
        g.group = COLLISION_GROUP
    for a in list(spec.actuators):
        spec.delete(a)
    PRIMITIVES[fid] = {g.name: (g.parent.name, int(g.type), np.array(g.pos, float), np.array(g.quat, float),
                                np.array(g.size, float))
                       for g in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM) if g.group == COLLISION_GROUP}
    convexify_primitives(spec, "fx")
    return spec, regions


def convexify_primitives(spec: mujoco.MjSpec, prefix: str, group: int = COLLISION_GROUP):
    """Replace box/cylinder collision geoms by exact convex meshes of the same shape.

    MuJoCo's analytic box-box routine returns spurious 0.0 distances for some thin-box
    configurations (observed for the door vs. upper-rack tines); convex meshes go through
    the general GJK/EPA path, which gives consistent signed distances for every pair.
    """
    k = 0
    for g in list(spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)):
        if g.group != group or g.type not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
            continue
        s = np.array(g.size, float)
        if g.type == mujoco.mjtGeom.mjGEOM_BOX:
            v = np.array([[i, j, l] for i in (-1, 1) for j in (-1, 1) for l in (-1, 1)], float) * s[:3]
        else:
            a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            ring = np.stack([np.cos(a) * s[0], np.sin(a) * s[0], np.zeros_like(a)], 1)
            v = np.concatenate([ring + [0, 0, s[1]], ring - [0, 0, s[1]]])
        name = f"{prefix}_cvx{k}"
        k += 1
        spec.add_mesh(name=name, uservert=v.reshape(-1).tolist())
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = name


def _geom_distance(m, d, g1, g2, cap=0.5):
    ft = np.zeros(6)
    return mujoco.mj_geomDistance(m, d, g1, g2, cap, ft)


def body_min_distance(m, d, body_a: str, body_b: str, cap=0.5) -> float:
    ga = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body_a).id and m.geom_group[g] == COLLISION_GROUP]
    gb = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body_b).id and m.geom_group[g] == COLLISION_GROUP]
    return float(min(_geom_distance(m, d, a, b, cap) for a in ga for b in gb))


def fixture_report(render_dir: Path | None = None) -> dict:
    """Validation record for the manifest (loads, joints articulate, clean articulation)."""
    cfg = load_asset_config()
    fx = cfg["fixture"]
    spec, regions = load_fixture_spec()
    m = spec.compile()
    d = mujoco.MjData(m)

    def set_q(door=0.0, rack0=0.0, rack1=0.0):
        d.qpos[:] = 0
        d.qpos[m.joint(fx["door_joint"]).qposadr[0]] = door
        d.qpos[m.joint(fx["lower_rack_joint"]).qposadr[0]] = rack0
        d.qpos[m.joint(fx["upper_rack_joint"]).qposadr[0]] = rack1
        mujoco.mj_kinematics(m, d)

    door_max = float(m.jnt_range[m.joint(fx["door_joint"]).id][1])
    r1_max = float(m.jnt_range[m.joint(fx["upper_rack_joint"]).id][1])
    r0_max = float(m.jnt_range[m.joint(fx["lower_rack_joint"]).id][1])
    gdoor = m.geom("door_main").id
    set_q()
    door_c0, door_R0 = d.geom_xpos[gdoor].copy(), d.geom_xmat[gdoor].reshape(3, 3).copy()
    rack1_p0 = d.xpos[m.body("rack1").id].copy()
    set_q(door=door_max, rack1=r1_max)
    door_c1, door_R1 = d.geom_xpos[gdoor].copy(), d.geom_xmat[gdoor].reshape(3, 3).copy()
    rack1_p1 = d.xpos[m.body("rack1").id].copy()
    door_angle = float(np.degrees(np.arccos(np.clip((np.trace(door_R0.T @ door_R1) - 1) / 2, -1, 1))))
    clear_door_rack1 = body_min_distance(m, d, "door", "rack1")
    set_q(door=door_max, rack0=r0_max)
    clear_door_rack0 = body_min_distance(m, d, "door", "rack0")
    set_q()
    col = [g for g in range(m.ngeom) if m.geom_group[g] == COLLISION_GROUP]
    lo = np.min([d.geom_xpos[g] - np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g] for g in col], 0)
    hi = np.max([d.geom_xpos[g] + np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g] for g in col], 0)
    vis = [g for g in range(m.ngeom) if m.geom_group[g] == VISUAL_GROUP and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    rep = {
        "fixture_id": FIXTURE_ID,
        "category": "dishwasher",
        "source_repository": cfg["sources"]["robocasa"]["repo"],
        "upstream_asset": f"{fx['archive']}:{fx['member_prefix']}",
        "upstream_version": cfg["sources"]["robocasa"]["commit"],
        "license": cfg["sources"]["robocasa"]["license"],
        "local_cache_path": str(fixture_xml().parent.relative_to(ROOT)),
        "visual_meshes": sorted({Path(me.file).name for me in spec.meshes}),
        "collision": f"{len(col)} source box/cylinder collision geoms (RoboCasa class 'collision')",
        "n_visual_faces": int(sum(m.mesh_facenum[m.geom_dataid[g]] for g in vis)),
        "scale": 1.0,
        "bounding_box_dims": [round(float(x), 4) for x in (hi - lo)],
        "joints": {
            "door": {"name": fx["door_joint"], "type": "hinge", "range": [0.0, door_max]},
            "upper_rack": {"name": fx["upper_rack_joint"], "type": "slide", "range": [0.0, r1_max]},
            "lower_rack": {"name": fx["lower_rack_joint"], "type": "slide", "range": [0.0, r0_max]},
        },
        "checks": {
            "door_rotation_deg_at_max": round(door_angle, 2),
            "door_centre_displacement_m": round(float(np.linalg.norm(door_c1 - door_c0)), 4),
            "upper_rack_translation_m": round(float(np.linalg.norm(rack1_p1 - rack1_p0)), 4),
            "clearance_open_door_vs_pulled_upper_rack_m": round(clear_door_rack1, 4),
            "clearance_open_door_vs_pulled_lower_rack_m": round(clear_door_rack0, 4),
        },
        "regions": {k: {"body": v[0], "pos": v[1].round(5).tolist(), "size": v[2].round(5).tolist()}
                    for k, v in regions.items()},
    }
    rep["checks"]["passed"] = bool(door_angle > 20 and rep["checks"]["upper_rack_translation_m"] > 0.3
                                   and clear_door_rack1 > 0.0)
    rep["notes"] = ("Real joint ranges are used unmodified. At the door's real maximum (38.6 deg) the fully "
                    "pulled LOWER rack would penetrate the door (negative clearance above), so v0 scenes keep the "
                    "lower rack in and use the upper rack in its fully pulled loading position.")
    return rep


_BOUNDS: dict = {}


def fixture_bounds(fid: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Collision AABB (fixture frame) with door closed and racks in."""
    fid = fid or FIXTURE_ID
    if fid not in _BOUNDS:
        from .assets import geom_local_points

        spec, _ = load_fixture_spec(fid)
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        pts = [d.geom_xpos[g] + geom_local_points(m, g) @ d.geom_xmat[g].reshape(3, 3).T
               for g in range(m.ngeom) if m.geom_group[g] == COLLISION_GROUP]
        pts = np.concatenate(pts)
        _BOUNDS[fid] = (pts.min(0), pts.max(0))
    return _BOUNDS[fid]


def adapt_world(world: dict, fid: str | None = None) -> dict:
    """Fit the kitchen shell around the fixture: base on the floor, front face at
    y = FRONT_Y, cabinets flush with its sides, countertop just above its top."""
    import copy

    fid = fid or FIXTURE_ID
    w = copy.deepcopy(world)
    lo, hi = fixture_bounds(fid)
    front_y = -0.28
    pos = np.array([-(lo[0] + hi[0]) / 2, front_y - lo[1], -lo[2] + 0.0005])
    w["dishwasher_pos"] = pos.round(5).tolist()
    half_w = (hi[0] - lo[0]) / 2 + 0.004
    top = hi[2] + pos[2]
    back = hi[1] + pos[1]
    ct_bottom = max(0.87, top + 0.012)
    w["countertop"] = {"x": [-0.95, 1.20], "y": [front_y - 0.05, back + 0.03], "z": [ct_bottom, ct_bottom + 0.04]}
    w["cabinets"] = [{"x": [-0.95, -half_w], "y": [front_y - 0.02, back + 0.03], "z": [0.0, ct_bottom]},
                     {"x": [half_w, 1.20], "y": [front_y - 0.02, back + 0.03], "z": [0.0, ct_bottom]}]
    w["wall"] = {"x": [-3.0, 3.0], "y": [back + 0.03, back + 0.05], "z": [0.0, 2.4]}
    w["fixture_id"] = fid
    return w
