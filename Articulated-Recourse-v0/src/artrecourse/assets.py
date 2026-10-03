"""Asset loading, canonicalisation and validation.

Visual mesh != collision mesh: RoboCasa objects ship textured visual OBJs plus a convex
decomposition (CoACD pieces). We keep the visual meshes for rendering only (contype=0,
massless) and use the source convex pieces as the ONLY collision geometry. Region helper
geoms (reg_bbox, reg_int, liquid, ...) are removed.

Canonical object frame (used by placements): origin at the object's anchor on its bottom
plane (z=0 at the lowest collision vertex); for handled objects (pan, mug, cup) the
handle points along +x and the anchor is the centre of the main body (not the handle).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from . import CACHE, CONFIGS, ROOT
from .geom import rot_z

MANIFEST = ROOT / "assets" / "asset_manifest.json"
HANDLED = {"pan", "mug", "cup"}
VISUAL_GROUP, COLLISION_GROUP = 2, 3


def load_asset_config() -> dict:
    return yaml.safe_load((CONFIGS / "assets.yaml").read_text())


@dataclass
class ObjectAsset:
    asset_id: str              # e.g. "pan/pan_1"
    category: str              # dataset category (mug, cup, bowl, plate, pan, bottle, can, box, ...)
    robocasa_category: str
    archive: str
    upstream_id: str
    scale: float
    meta: dict = field(default_factory=dict)  # filled from the manifest (canonical frame, dims, ...)
    source: str = "robocasa"   # "robocasa" | "scanned_objects"

    @property
    def root_body(self) -> str:
        return "model" if self.source == "scanned_objects" else "object"

    @property
    def model_xml(self) -> Path:
        if self.source == "scanned_objects":
            return CACHE / "mujoco_scanned_objects" / "models" / self.upstream_id / "model.xml"
        top = "objaverse" if self.archive == "objaverse" else "lightwheel"
        return CACHE / "robocasa" / self.archive / top / self.robocasa_category / self.upstream_id / "model.xml"


def screening_pool() -> list[ObjectAsset]:
    cfg = load_asset_config()["objects"]
    out = []
    for cat, spec in cfg["categories"].items():
        src = spec.get("source", "robocasa")
        arch = spec.get("archive", cfg["archive"]) if src == "robocasa" else "git"
        for inst in spec["pool"]:
            out.append(ObjectAsset(f"{cat}/{inst}", cat, spec.get("robocasa_category", cat), arch, inst,
                                   float(spec["scale"]), source=src))
    return out


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def selected_assets(manifest: dict | None = None) -> dict[str, ObjectAsset]:
    """Accepted object assets from the manifest, keyed by asset id."""
    manifest = manifest or load_manifest()
    out = {}
    for a in manifest["objects"]:
        if not a["accepted"]:
            continue
        out[a["asset_id"]] = ObjectAsset(a["asset_id"], a["category"], a["robocasa_category"], a["archive"],
                                         a["upstream_id"], a["scale"], meta=a, source=a["source_key"])
    return out


# ----------------------------------------------------------------------------- loading

def _absolutize(spec: mujoco.MjSpec, base: Path):
    for me in spec.meshes:
        if me.file and not Path(me.file).is_absolute():
            me.file = str(base / me.file)
    for t in spec.textures:
        if t.file and not Path(t.file).is_absolute():
            t.file = str(base / t.file)
    spec.meshdir = ""
    spec.texturedir = ""


def load_object_spec(asset: ObjectAsset) -> mujoco.MjSpec:
    """Load a RoboCasa object MJCF, apply the registry scale and separate visual/collision."""
    spec = mujoco.MjSpec.from_file(str(asset.model_xml))
    _absolutize(spec, asset.model_xml.parent)
    s = asset.scale
    for me in spec.meshes:
        me.scale = np.asarray(me.scale) * s
    root = spec.body(asset.root_body)
    for b in root.find_all(mujoco.mjtObj.mjOBJ_BODY):
        b.pos = np.asarray(b.pos) * s
    ncol = 0
    for g in list(root.find_all(mujoco.mjtObj.mjOBJ_GEOM)):
        g.pos = np.asarray(g.pos) * s
        is_mesh = g.type == mujoco.mjtGeom.mjGEOM_MESH
        is_visual = g.contype == 0 and g.conaffinity == 0
        if asset.source == "scanned_objects":
            is_visual = g.group == 2
        if is_visual:
            if not is_mesh or (g.name or "").startswith("reg_"):
                spec.delete(g)              # region / helper geom
                continue
            g.group = VISUAL_GROUP          # visual mesh: rendering only, massless
            g.density = 0.0
            g.mass = 0.0
        else:
            if not is_mesh:
                g.size = np.asarray(g.size) * s
            g.group = COLLISION_GROUP
            g.rgba = [0.8, 0.3, 0.3, 0.0]
            g.name = f"col{ncol}"
            g.contype, g.conaffinity = 1, 1
            ncol += 1
    for site in list(root.find_all(mujoco.mjtObj.mjOBJ_SITE)):
        spec.delete(site)
    return spec


def object_collision_vertices(asset: ObjectAsset) -> tuple[np.ndarray, mujoco.MjModel]:
    """All collision-mesh vertices expressed in the RoboCasa 'object' body frame."""
    spec = load_object_spec(asset)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = m.body(asset.root_body).id
    R0, p0 = d.xmat[bid].reshape(3, 3), d.xpos[bid]
    pts = []
    for g in range(m.ngeom):
        if m.geom_group[g] != COLLISION_GROUP:
            continue
        v = geom_local_points(m, g)
        w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        pts.append((w - p0) @ R0)
    return np.concatenate(pts), m


def geom_local_points(m: mujoco.MjModel, g: int) -> np.ndarray:
    """Surface sample points of a collision geom in its own frame (exact for meshes/boxes)."""
    t, s = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        return m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        return c * s[:3]
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return np.concatenate([ring * s[0] + [0, 0, s[1]], ring * s[0] - [0, 0, s[1]]])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.concatenate([ring * s[0] + [0, 0, s[1]], ring * s[0] - [0, 0, s[1]],
                               [[0, 0, s[1] + s[0]], [0, 0, -s[1] - s[0]]]])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.concatenate([ring * s[0], [[0, 0, s[0]], [0, 0, -s[0]]]])
    raise ValueError(f"unsupported collision geom type {t}")


def canonical_frame(category: str, verts: np.ndarray) -> dict:
    """Canonical yaw (handle -> +x, long axis -> x) and anchor for placement."""
    xy = verts[:, :2]
    ext = xy.max(0) - xy.min(0)
    cyaw = 0.0
    if ext[1] > ext[0] * 1.05:
        cyaw = -np.pi / 2  # rotate long axis onto x
    R = rot_z(cyaw)[:2, :2]
    p = xy @ R.T
    if category in HANDLED:
        lo, hi = p[:, 0].min(), p[:, 0].max()
        L = hi - lo

        def width_near(x0):
            sel = np.abs(p[:, 0] - x0) < 0.12 * L
            return np.ptp(p[sel, 1]) if sel.sum() > 3 else 0.0

        if width_near(lo + 0.08 * L) < width_near(hi - 0.08 * L):  # handle is on the -x side
            cyaw += np.pi
            p = -p
    lo, hi = p.min(0), p.max(0)
    width_y = hi[1] - lo[1]
    if category in HANDLED:
        body_d = min(width_y, hi[0] - lo[0])
        anchor_xy = np.array([lo[0] + body_d / 2, (lo[1] + hi[1]) / 2])
    else:
        anchor_xy = (lo + hi) / 2
    z0 = float(verts[:, 2].min())
    return {
        "canonical_yaw": float(cyaw),
        "anchor": [float(anchor_xy[0]), float(anchor_xy[1]), z0],   # in yaw-rotated body frame
        "extent_canonical": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(verts[:, 2].max() - z0)],
        "canonical_xy_min": [float(lo[0] - anchor_xy[0]), float(lo[1] - anchor_xy[1])],
        "canonical_xy_max": [float(hi[0] - anchor_xy[0]), float(hi[1] - anchor_xy[1])],
        "height": float(verts[:, 2].max() - z0),
    }


def body_pose_from_canonical(meta: dict, pos, yaw) -> tuple[np.ndarray, np.ndarray]:
    """World pose of the RoboCasa 'object' body given the canonical-frame placement."""
    cyaw = meta["canonical_yaw"]
    ax, ay, az = meta["anchor"]
    total = yaw + cyaw
    # body point b maps to canonical c = Rz(cyaw) b - anchor; world = pos + Rz(yaw) c
    offset = rot_z(yaw) @ (-np.array([ax, ay, az]))
    bpos = np.asarray(pos, float) + offset
    bquat = np.array([np.cos(total / 2), 0, 0, np.sin(total / 2)])
    return bpos, bquat


# ----------------------------------------------------------------------------- validation

def analyze_object(asset: ObjectAsset) -> dict:
    verts, m = object_collision_vertices(asset)
    info = canonical_frame(asset.category, verts)
    bid = m.body(asset.root_body).id
    sub = [b for b in range(m.nbody) if b == bid or m.body_rootid[b] == m.body_rootid[bid]]
    mass = float(sum(m.body_mass[b] for b in sub))
    ncol = int(sum(m.geom_group[g] == COLLISION_GROUP for g in range(m.ngeom)))
    vis = [g for g in range(m.ngeom) if m.geom_group[g] == VISUAL_GROUP]
    vis_faces = int(sum(m.mesh_facenum[m.geom_dataid[g]] for g in vis))
    col_faces = int(sum(m.mesh_facenum[m.geom_dataid[g]] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else 12
                        for g in range(m.ngeom) if m.geom_group[g] == COLLISION_GROUP))
    inertia = [float(x) for b in sub if m.body_mass[b] > 0 for x in m.body_inertia[b]]
    bbox = (verts.max(0) - verts.min(0)).tolist()
    # visual-vs-collision scale agreement: compare visual AABB with collision AABB
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    vpts = []
    for g in vis:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        vpts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
    vpts = np.concatenate(vpts) - d.xpos[bid]
    vext = vpts.max(0) - vpts.min(0)
    info.update({
        "mass": mass,
        "inertia_diag": inertia,
        "bbox_dims": [round(float(x), 4) for x in bbox],
        "visual_bbox_dims": [round(float(x), 4) for x in vext],
        "visual_collision_extent_ratio": [round(float(a / b), 3) for a, b in zip(vext, bbox)],
        "n_collision_geoms": ncol,
        "n_visual_faces": vis_faces,
        "n_collision_faces": col_faces,
    })
    return info


def drop_test(asset: ObjectAsset, seconds: float = 1.5) -> dict:
    """Drop the object 1 cm above a plane in its canonical upright pose and let it settle."""
    child = load_object_spec(asset)
    verts, _ = object_collision_vertices(asset)
    z0 = verts[:, 2].min()
    w = mujoco.MjSpec()
    w.option.timestep = 0.002
    w.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    b = w.worldbody.add_body(name="drop", pos=[0, 0, 0.01 - z0])
    b.add_freejoint()
    b.add_frame().attach_body(child.body(asset.root_body), "o_", "")
    m = w.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p_start = d.qpos[:3].copy()
    max_pen = 0.0
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if d.ncon:
            max_pen = max(max_pen, float(-min(c.dist for c in d.contact[: d.ncon])))
    q = d.qpos[3:7]
    tilt = float(np.degrees(2 * np.arccos(np.clip(np.sqrt(q[0] ** 2 + q[3] ** 2), -1, 1))))
    res = {
        "finite": bool(np.all(np.isfinite(d.qpos))),
        "settled_speed": float(np.linalg.norm(d.qvel)),
        "tilt_deg": tilt,
        "xy_drift": float(np.linalg.norm(d.qpos[:2] - p_start[:2])),
        "max_penetration": max_pen,
        "rest_height_error": float(d.qpos[2] - (0 - z0)),
    }
    res["passed"] = bool(res["finite"] and res["settled_speed"] < 0.05 and tilt < 10.0
                         and res["xy_drift"] < 0.03 and max_pen < 0.01 and abs(res["rest_height_error"]) < 0.01)
    return res


CATEGORY_DIM_RANGES = {  # (min, max) of the largest horizontal dimension and of height, metres
    "mug": ((0.07, 0.16), (0.06, 0.14)),
    "cup": ((0.05, 0.13), (0.06, 0.16)),
    "bowl": ((0.10, 0.26), (0.04, 0.12)),
    "plate": ((0.15, 0.32), (0.01, 0.06)),
    "pan": ((0.25, 0.50), (0.03, 0.12)),
    "bottle": ((0.05, 0.12), (0.18, 0.36)),
    "can": ((0.05, 0.10), (0.08, 0.16)),
    "box": ((0.07, 0.30), (0.10, 0.34)),
    "utensil_holder": ((0.08, 0.18), (0.10, 0.25)),
}


def dims_reasonable(category: str, info: dict) -> bool:
    (lo, hi), (hlo, hhi) = CATEGORY_DIM_RANGES[category]
    horiz = max(info["bbox_dims"][:2])
    return lo <= horiz <= hi and hlo <= info["height"] <= hhi
