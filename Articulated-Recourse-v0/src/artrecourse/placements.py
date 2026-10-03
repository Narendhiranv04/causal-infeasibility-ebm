"""Support surfaces and exact placement poses.

Support regions are measured from the compiled scene's actual collision geometry (the
pulled-out upper rack's floor, tine plates and walls; the countertop). A placement is an
exact pose T = (x, y, z, qx, qy, qz, qw) of the object's canonical frame; z is the support
height, orientation is a yaw about world z (handled objects: handle direction).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .assets import geom_local_points
from .geom import pose7, quat_from_yaw, rot_z
from .scene import Scene, load_world_config

EPS_REST = 0.0025      # |gap| allowed between object bottom and its support at rest
PEN_ENV = 0.002        # penetration tolerated against environment (contact numerics)


@dataclass
class Support:
    name: str
    z: float
    x: tuple
    y: tuple
    kind: str            # "bay" | "tines" | "counter"
    tine_x: tuple = ()   # x centres of tine plates (tines only)

    def contains(self, lo, hi, margin=0.0) -> bool:
        return (lo[0] >= self.x[0] + margin and hi[0] <= self.x[1] - margin
                and lo[1] >= self.y[0] + margin and hi[1] <= self.y[1] - margin)


def _world_aabb(m, d, g):
    p = geom_local_points(m, g)
    w = d.geom_xpos[g] + p @ d.geom_xmat[g].reshape(3, 3).T
    return w.min(0), w.max(0)


def _rack_supports(scene, rack: str, prefix: str) -> tuple[dict, float]:
    m, d = scene.model, scene.data
    boxes = [_world_aabb(m, d, g) for g in scene.idx.articulated[rack]]
    floor = None
    tines, walls_x, walls_y = [], [], []
    for lo, hi in boxes:
        ext = hi - lo
        if ext[2] < 0.01 and ext[0] > 0.4:
            floor = (lo, hi)
        elif ext[0] < 0.01 and ext[1] > 0.4:
            tines.append((lo, hi))
        elif ext[0] < 0.03 and ext[2] > 0.08:
            walls_x.append((lo, hi))
        elif ext[1] < 0.02 and ext[0] > 0.4:
            walls_y.append((lo, hi))
    tines.sort(key=lambda b: b[0][0])
    walls_x.sort(key=lambda b: b[0][0])
    walls_y.sort(key=lambda b: b[0][1])
    y_lo, y_hi = walls_y[0][1][1], walls_y[-1][0][1]
    sup = {
        f"{prefix}_left_bay": Support(f"{prefix}_left_bay", float(floor[1][2]),
                                      (float(walls_x[0][1][0]), float(tines[0][0][0])), (float(y_lo), float(y_hi)), "bay"),
        f"{prefix}_right_bay": Support(f"{prefix}_right_bay", float(floor[1][2]),
                                       (float(tines[-1][1][0]), float(walls_x[-1][0][0])), (float(y_lo), float(y_hi)), "bay"),
        f"{prefix}_tines": Support(f"{prefix}_tines", float(max(t[1][2] for t in tines)),
                                   (float(walls_x[0][1][0]), float(walls_x[-1][0][0])), (float(y_lo), float(y_hi)),
                                   "tines", tuple(float((t[0][0] + t[1][0]) / 2) for t in tines)),
    }
    return sup, float(max(b[1][2] for b in walls_x + walls_y))


def measure_supports(scene: Scene) -> dict[str, Support]:
    scene.set_articulation()
    scene.forward()
    sup, walls_top = _rack_supports(scene, "rack1", "rack")
    try:
        lower, _ = _rack_supports(scene, "rack0", "lower")
        sup.update(lower)
    except Exception:  # noqa: BLE001  (fixtures whose lower rack is not a wire basket)
        pass
    ct = scene.world["countertop"]
    sup["counter"] = Support("counter", float(ct["z"][1]), (float(ct["x"][0]) + 0.02, float(ct["x"][1]) - 0.02),
                             (float(ct["y"][0]), float(ct["y"][1]) - 0.02), "counter")
    from .topology import TRAY

    rt = TRAY["rim_t"]
    sup["staging_tray"] = Support("staging_tray", float(ct["z"][1] + TRAY["base"]),
                                  (TRAY["x"][0] + rt, TRAY["x"][1] - rt), (TRAY["y"][0] + rt, TRAY["y"][1] - rt), "counter")
    sup["_rack_walls_top"] = walls_top
    return sup


@dataclass
class Placement:
    """An exact object pose at a named slot."""
    slot: str
    support: str
    pos: np.ndarray       # canonical-frame origin (anchor on the bottom plane), world
    yaw: float
    orient: str = ""      # named orientation template (v0.1)

    @property
    def quat(self):
        return quat_from_yaw(self.yaw)

    def as_pose7(self):
        return pose7(self.pos, self.quat)

    def key(self):
        return f"{self.slot}:{self.orient}" if self.orient else f"{self.slot}@{np.degrees(self.yaw):.0f}"


def footprint(meta: dict, pos, yaw) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.array(meta["canonical_xy_min"]), np.array(meta["canonical_xy_max"])
    c = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    w = c @ rot_z(yaw)[:2, :2].T + np.asarray(pos)[:2]
    return w.min(0), w.max(0)


def slot_catalogue() -> dict:
    return load_world_config()["slots"]


def make_placement(scene: Scene, supports: dict, obj_key: str, slot: str, yaw_deg: float,
                   dxy=(0.0, 0.0)) -> Placement:
    s = slot_catalogue()[slot]
    sup = supports[s["support"]]
    xy = np.asarray(s["xy"], float) + np.asarray(dxy, float)
    return Placement(slot, sup.name, np.array([xy[0], xy[1], sup.z]), float(np.radians(yaw_deg)))


def check_placement(scene: Scene, supports: dict, obj_key: str, pl: Placement) -> tuple[bool, list[str], dict]:
    """Static validity of an object at a placement: inside its support region, resting on
    the support (not floating), not penetrating the environment, statically stable."""
    meta = scene.objects[obj_key].asset.meta
    sup = supports[pl.support]
    reasons, info = [], {}
    lo, hi = footprint(meta, pl.pos, pl.yaw)
    if sup.kind == "counter":
        if not sup.contains(lo, hi, margin=0.005):
            reasons.append("footprint outside support region")
    elif sup.kind == "bay":
        if not sup.contains(lo, hi, margin=-0.001):
            reasons.append("footprint outside rack bay")
    else:  # tines: centre (anchor) over the tine span and resting on >= 2 tine plates
        if not (sup.tine_x[0] <= pl.pos[0] <= sup.tine_x[-1] and sup.y[0] < pl.pos[1] < sup.y[1]):
            reasons.append("anchor not over the tine span")
    # geometric checks with the actual collision geoms
    m, d = scene.model, scene.data
    for k in scene.objects:
        scene.hide_object(k)
    scene.set_object_canonical(obj_key, pl.pos, pl.yaw)
    scene.hide_gripper()
    scene.forward()
    ft = np.zeros(6)
    env = scene.idx.env_geoms + [g for v in scene.idx.articulated.values() for g in v]
    min_env, support_gap, n_tine_contacts = 1.0, 1.0, 0
    for g in scene.idx.obj_geoms[obj_key]:
        for e in env:
            if np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[e]) > m.geom_rbound[g] + m.geom_rbound[e] + 0.02:
                continue
            dist = mujoco.mj_geomDistance(m, d, g, e, 0.05, ft)
            min_env = min(min_env, dist)
    # resting: lowest collision point vs support height
    pts = []
    for g in scene.idx.obj_geoms[obj_key]:
        p = geom_local_points(m, g)
        pts.append(d.geom_xpos[g] + p @ d.geom_xmat[g].reshape(3, 3).T)
    pts = np.concatenate(pts)
    support_gap = float(pts[:, 2].min() - sup.z)
    if sup.kind == "tines":
        low = pts[pts[:, 2] < sup.z + 0.004]
        covered = [tx for tx in sup.tine_x if low.size and low[:, 0].min() - 0.002 <= tx <= low[:, 0].max() + 0.002]
        n_tine_contacts = len(covered)
        if n_tine_contacts < 2:
            reasons.append("rests on fewer than two tine plates")
        elif not (min(covered) <= pts[:, 0].mean() <= max(covered)):
            reasons.append("centre of mass outside tine support")
    info.update(min_env_distance=float(min_env), support_gap=support_gap, tine_contacts=n_tine_contacts)
    if min_env < -PEN_ENV:
        reasons.append(f"penetrates environment ({min_env:+.4f} m)")
    if abs(support_gap) > EPS_REST:
        reasons.append(f"not resting on support (gap {support_gap:+.4f} m)")
    return not reasons, reasons, info
