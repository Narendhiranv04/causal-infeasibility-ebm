"""MuJoCo world construction.

One compiled model per scene contains:
  * the kitchen shell (floor, wall, base cabinets, countertop; textured RoboCasa materials,
    collision as exact convex meshes),
  * the RoboCasa Dishwasher054 fixture with its real joints,
  * every movable object as a kinematic mocap body (visual meshes + source convex pieces),
  * a floating Panda hand ("gripper", mocap, TCP frame) used by the swept-volume oracle,
  * optionally the full Panda arm on a pedestal (IK / admissibility / rendering).

Render geom groups: 1 = robot arm visuals, 2 = scene visuals, 3 = collision (hidden),
4 = floating-gripper visuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import yaml

from . import CACHE, CONFIGS
from .assets import COLLISION_GROUP, VISUAL_GROUP, ObjectAsset, body_pose_from_canonical, load_asset_config, load_object_spec
from .fixture import convexify_primitives, load_fixture_spec
from .geom import quat_from_yaw
from .robot import TCP_OFFSET, load_panda_spec

ROBOT_GROUP, GRIPPER_GROUP = 1, 4
DW = "dw_"


def load_world_config() -> dict:
    return yaml.safe_load((CONFIGS / "placements.yaml").read_text())


@dataclass
class ObjectInstance:
    key: str            # unique within scene, e.g. "pan", "mug1"
    asset: ObjectAsset


@dataclass
class SceneIndex:
    env_groups: dict = field(default_factory=dict)     # name -> [geom ids] (static environment)
    articulated: dict = field(default_factory=dict)    # "door" / "rack1" / "rack0" -> [geom ids]
    obj_mocap: dict = field(default_factory=dict)      # key -> mocap id
    obj_body: dict = field(default_factory=dict)       # key -> body id (mocap body)
    obj_geoms: dict = field(default_factory=dict)      # key -> [collision geom ids]
    obj_vis_geoms: dict = field(default_factory=dict)  # key -> [visual geom ids]
    gripper_mocap: int = -1
    gripper_geoms: list = field(default_factory=list)
    gripper_vis_geoms: list = field(default_factory=list)
    gripper_finger_qadr: list = field(default_factory=list)
    joint_qadr: dict = field(default_factory=dict)     # door / rack0 / rack1
    joint_range: dict = field(default_factory=dict)

    @property
    def env_geoms(self) -> list[int]:
        return [g for v in self.env_groups.values() for g in v]


def _box_mesh(spec, name, half):
    v = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * np.asarray(half, float)
    spec.add_mesh(name=name, uservert=v.reshape(-1).tolist())


def lookat_quat(pos, target, up=(0, 0, 1)):
    """Camera quaternion (MuJoCo cameras look along -z with +y up)."""
    from .geom import mat_to_quat

    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return mat_to_quat(np.stack([x, y, -f], 1))


CAMERAS = {
    "front": ((-0.35, -2.35, 1.75), (0.05, -0.30, 0.68)),      # wide, from the front-left (robot is front-right)
    "oblique": ((1.75, -1.75, 1.85), (0.10, -0.32, 0.62)),
    "top": ((0.05, -0.62, 2.75), (0.05, -0.40, 0.60)),
    "closeup": ((-0.55, -1.55, 1.45), (0.02, -0.42, 0.60)),
}


class Scene:
    """Compiled MuJoCo model for one scene (fixed object set)."""

    def __init__(self, objects: list[ObjectInstance], with_robot: bool = True, world: dict | None = None):
        self.world = world or load_world_config()["world"]
        self.objects = {o.key: o for o in objects}
        self.with_robot = with_robot
        self.spec = mujoco.MjSpec()
        self.idx = SceneIndex()
        self._build()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._index()
        self.set_articulation()
        self.kin = None
        if with_robot:
            from .robot import PandaKinematics

            self.kin = PandaKinematics(self.model, "panda_", data=self.data)

    # ------------------------------------------------------------------ build
    def _build(self):
        s, w = self.spec, self.world
        s.compiler.degree = False
        s.option.timestep = 0.002
        s.visual.global_.offwidth = 1280
        s.visual.global_.offheight = 960
        s.visual.headlight.ambient = [0.42, 0.42, 0.42]
        s.visual.headlight.diffuse = [0.35, 0.35, 0.35]
        s.visual.headlight.specular = [0.05, 0.05, 0.05]
        s.visual.quality.shadowsize = 4096
        s.visual.rgba.fog = [0.86, 0.88, 0.9, 1]
        s.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=[0.93, 0.94, 0.96], rgb2=[0.78, 0.80, 0.84], width=256, height=256)
        tex = load_asset_config()["textures"]
        tdir = CACHE / "robocasa" / "textures"
        for name, rep in (("counter", [2, 1]), ("cabinet", [2, 2]), ("floor", [8, 8]), ("wall", [3, 2])):
            s.add_texture(name=f"tex_{name}", file=str(tdir / tex[name]), type=mujoco.mjtTexture.mjTEXTURE_2D)
            s.add_material(name=f"mat_{name}", textures=["", f"tex_{name}"], texrepeat=rep, texuniform=True,
                           specular=0.2 if name != "counter" else 0.5, shininess=0.3)
        wb = s.worldbody
        wb.add_light(name="key", pos=[0.8, -1.8, 2.6], dir=[-0.3, 0.6, -0.75], diffuse=[0.55, 0.55, 0.52],
                     castshadow=True, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        wb.add_light(name="fill", pos=[-1.2, -1.5, 2.0], dir=[0.5, 0.5, -0.7], diffuse=[0.25, 0.25, 0.27],
                     castshadow=False, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        wb.add_geom(name="floor_vis", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], material="mat_floor",
                    contype=0, conaffinity=0, group=VISUAL_GROUP)
        boxes = {"floor": {"x": [-3, 3], "y": [-3, 3], "z": [-0.05, 0.0]}, "countertop": w["countertop"],
                 "wall": w["wall"]}
        for i, c in enumerate(w["cabinets"]):
            boxes[f"cabinet{i}"] = c
        mats = {"floor": None, "countertop": "mat_counter", "wall": "mat_wall"}
        for name, b in boxes.items():
            lo = np.array([b["x"][0], b["y"][0], b["z"][0]])
            hi = np.array([b["x"][1], b["y"][1], b["z"][1]])
            c, half = (lo + hi) / 2, (hi - lo) / 2
            mat = mats.get(name, "mat_cabinet")
            if mat:
                wb.add_geom(name=f"{name}_vis", type=mujoco.mjtGeom.mjGEOM_BOX, pos=c, size=half, material=mat,
                            contype=0, conaffinity=0, group=VISUAL_GROUP)
            _box_mesh(s, f"env_{name}_mesh", half)
            wb.add_geom(name=f"env_{name}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"env_{name}_mesh", pos=c,
                        group=COLLISION_GROUP, rgba=[0.5, 0.5, 0.5, 0])
        # dishwasher
        fx_spec, self.fixture_regions = load_fixture_spec()
        frame = wb.add_frame(pos=w["dishwasher_pos"])
        frame.attach_body(fx_spec.body("object"), DW, "")
        # objects (mocap)
        for key, inst in self.objects.items():
            child = load_object_spec(inst.asset)
            b = wb.add_body(name=f"obj_{key}", mocap=True, pos=[0, 0, -5])
            b.add_frame().attach_body(child.body(inst.asset.root_body), f"{key}__", "")
        # floating gripper in TCP frame
        panda = load_panda_spec()
        hand = panda.body("hand")
        hand.pos, hand.quat = [0, 0, 0], [1, 0, 0, 0]   # drop the link7->hand offset (0.107 m, -45 deg yaw)
        g = wb.add_body(name="gripper", mocap=True, pos=[0, 0, -5])
        g.add_frame(pos=[0, 0, -TCP_OFFSET]).attach_body(hand, "grip_", "")
        # robot: mobile base = mocap body carrying the pedestal and the Panda
        if self.with_robot:
            rb = w["robot_base"]
            ped = w["pedestal"]
            hz = max(st["pos"][2] for st in w["robot_stations"].values()) / 2
            base = wb.add_body(name="robot_base", mocap=True, pos=rb["pos"],
                               quat=quat_from_yaw(np.radians(rb["yaw_deg"])))
            base.add_geom(name="pedestal_vis", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -hz],
                          size=[ped["half"][0], ped["half"][1], hz], rgba=[0.22, 0.23, 0.25, 1],
                          contype=0, conaffinity=0, group=ROBOT_GROUP)
            arm = load_panda_spec()
            base.add_frame().attach_body(arm.body("link0"), "panda_", "")
        for name, (pos, tgt) in CAMERAS.items():
            wb.add_camera(name=name, pos=pos, quat=lookat_quat(pos, tgt), fovy=50 if name == "front" else 45)

    # ------------------------------------------------------------------ index
    def _index(self):
        m, ix = self.model, self.idx
        col = [g for g in range(m.ngeom) if m.geom_group[g] == COLLISION_GROUP]

        def body_name(g):
            return m.body(m.geom_bodyid[g]).name

        for g in col:
            bn, gn = body_name(g), m.geom(g).name
            if gn.startswith("env_"):
                ix.env_groups.setdefault(gn[4:], []).append(g)
            elif bn.startswith(DW):
                part = bn[len(DW):]
                if part in ("door", "button_power"):
                    ix.articulated.setdefault("door", []).append(g)
                elif part in ("rack0", "rack1"):
                    ix.articulated.setdefault(part, []).append(g)
                else:
                    ix.env_groups.setdefault("dishwasher", []).append(g)
        for key in self.objects:
            bid = m.body(f"obj_{key}").id
            ix.obj_body[key] = bid
            ix.obj_mocap[key] = m.body_mocapid[bid]
            sub = [b for b in range(m.nbody) if self._is_desc(b, bid)]
            ix.obj_geoms[key] = [g for g in col if m.geom_bodyid[g] in sub]
            ix.obj_vis_geoms[key] = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub and m.geom_group[g] == VISUAL_GROUP]
        gb = m.body("gripper").id
        ix.gripper_mocap = m.body_mocapid[gb]
        gsub = [b for b in range(m.nbody) if self._is_desc(b, gb)]
        ix.gripper_geoms = [g for g in col if m.geom_bodyid[g] in gsub]
        ix.gripper_vis_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in gsub and m.geom_group[g] == VISUAL_GROUP]
        for g in ix.gripper_vis_geoms:
            m.geom_group[g] = GRIPPER_GROUP
        for jn in ("grip_finger_joint1", "grip_finger_joint2"):
            ix.gripper_finger_qadr.append(m.jnt_qposadr[m.joint(jn).id])
        for k, jn in (("door", "door_joint"), ("rack0", "rack0_joint"), ("rack1", "rack1_joint")):
            j = m.joint(DW + jn).id
            ix.joint_qadr[k] = m.jnt_qposadr[j]
            ix.joint_range[k] = m.jnt_range[j].copy()
        if self.with_robot:
            for g in range(m.ngeom):
                if body_name(g).startswith("panda_") and m.geom_group[g] == VISUAL_GROUP:
                    m.geom_group[g] = ROBOT_GROUP

    def _is_desc(self, b, root):
        m = self.model
        while b > 0:
            if b == root:
                return True
            b = m.body_parentid[b]
        return False

    # ------------------------------------------------------------------ state setters
    def articulation_targets(self) -> dict:
        a = self.world["articulation"]
        out = {}
        for k, jk in (("door", "door"), ("upper_rack", "rack1"), ("lower_rack", "rack0")):
            v = a[k]
            out[jk] = float(self.idx.joint_range[jk][1]) if v == "max" else float(v)
        return out

    def set_articulation(self, door=None, rack0=None, rack1=None):
        t = self.articulation_targets()
        d = self.data
        d.qpos[self.idx.joint_qadr["door"]] = t["door"] if door is None else door
        d.qpos[self.idx.joint_qadr["rack0"]] = t["rack0"] if rack0 is None else rack0
        d.qpos[self.idx.joint_qadr["rack1"]] = t["rack1"] if rack1 is None else rack1

    def set_object_canonical(self, key: str, pos, yaw: float):
        bpos, bquat = body_pose_from_canonical(self.objects[key].asset.meta, pos, yaw)
        self.set_object_body(key, bpos, bquat)

    def set_object_body(self, key, bpos, bquat):
        mid = self.idx.obj_mocap[key]
        self.data.mocap_pos[mid] = bpos
        self.data.mocap_quat[mid] = bquat

    def hide_object(self, key):
        self.set_object_body(key, [0, 0, -5], [1, 0, 0, 0])

    def set_gripper(self, pos, R=None, quat=None, opening=0.08):
        from .geom import mat_to_quat

        mid = self.idx.gripper_mocap
        self.data.mocap_pos[mid] = pos
        self.data.mocap_quat[mid] = quat if quat is not None else mat_to_quat(R)
        for a in self.idx.gripper_finger_qadr:
            self.data.qpos[a] = np.clip(opening / 2, 0, 0.04)

    def set_station(self, name: str):
        st = self.world["robot_stations"][name]
        mid = self.model.body_mocapid[self.model.body("robot_base").id]
        self.data.mocap_pos[mid] = st["pos"]
        self.data.mocap_quat[mid] = quat_from_yaw(np.radians(st["yaw_deg"]))

    def hide_gripper(self):
        self.set_gripper([0, 0, -5], quat=[1, 0, 0, 0])

    def forward(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
