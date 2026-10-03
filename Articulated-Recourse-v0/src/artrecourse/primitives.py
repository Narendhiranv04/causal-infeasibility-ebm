"""Deterministic manipulation primitives, densely sampled in normalised progress tau.

RELOCATE(object, exact destination pose, grasp template):
    approach (top-down, 10 cm) -> grasp (fingers close) -> lift vertically -> transfer
    (straight line at the transfer altitude, yaw interpolated) -> lower -> release
    (fingers open) -> retreat (10 cm up).
  Transfer altitude: the carried object's bottom is LIFT_CLEAR above the higher of the two
  support surfaces (13 cm: just clears the 11.7 cm upper-rack walls). The trajectory is a
  pure function of (source pose, destination pose, object category / canonical geometry,
  grasp template).

CLOSE_DISHWASHER (target articulation), tau in [0, 0.5]: PUSH_RACK — real rack1_joint
  0.40 -> 0, the hand grasping the top edge of the rack's front wall, everything resting on
  the rack carried along; tau in [0.5, 1]: CLOSE_DOOR — real door_joint 0.6736 -> 0 rad with the closed hand
  pushing the door's outer face 0.25 m below its top edge. The robot's transit between the
  two contacts is not part of the swept volume (it happens in free space in front of the
  machine).

Sampling: every segment is sampled so that consecutive TCP/body samples are at most
`spacing` apart (coarse 1 cm, fine 2.5 mm); tau = cumulative path length / total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import mat_to_quat, quat_from_yaw, rot_z, wrap_angle

APPROACH_H = 0.10
RETREAT_H = 0.10
LIFT_CLEAR = 0.13          # v0 fallback only; v0.1 derives the carry height (transfer_bottom)
TRANSFER_MARGIN = 0.03     # v0.1: safety margin above the highest lip crossed by the transfer
LIP_MAX = 0.25             # geometry more than this above the higher support is a ceiling (passed under)
COARSE_SPACING = 0.01
FINE_SPACING = 0.0025
DOOR_CONTACT_FRACTION = 0.6   # CLOSE_DOOR contact point, fraction of door length from the hinge


@dataclass
class GraspTemplate:
    name: str
    local_pos: np.ndarray      # TCP position in the object's canonical frame
    finger_axis_local: float   # yaw fy of the finger axis: TCP +y = Rz(fy) * (0, 1, 0) in the object frame
    open_width: float
    closed_width: float

    def to_dict(self):
        return {"name": self.name, "tcp_in_object_frame": [round(float(x), 4) for x in self.local_pos],
                "finger_axis_yaw_in_object_frame": round(float(self.finger_axis_local), 4),
                "open_width": self.open_width, "closed_width": self.closed_width,
                "approach": "top-down (TCP z = -world z)"}


def grasp_candidates(category: str, meta: dict, verts_canon: np.ndarray | None = None) -> list[GraspTemplate]:
    """Deterministic, ordered grasp candidates in the canonical object frame (anchor, z=0 at
    the bottom, +x = handle). The primitive uses the first candidate whose whole trajectory is
    clear of the FIXED environment and admissible for the robot, so the choice depends only on
    (source pose, destination pose, category, fixture) -- never on other objects."""
    lo, hi = np.array(meta["canonical_xy_min"]), np.array(meta["canonical_xy_max"])
    h = meta["height"]
    out = []
    if category == "pan":
        r = -lo[0]
        for f in (0.75, 0.6, 0.9, 0.45):
            xg = r + f * (hi[0] - r)
            zg = h - 0.012
            if verts_canon is not None:
                near = verts_canon[(np.abs(verts_canon[:, 0] - xg) < 0.012) & (np.abs(verts_canon[:, 1]) < 0.03)]
                if len(near):
                    zg = float(near[:, 2].max()) - 0.010
            # fingers close ACROSS the handle (object +y), i.e. fy = 0
            out.append(GraspTemplate(f"pan_handle_topdown_{int(f * 100)}", np.array([xg, 0.0, zg]), 0.0, 0.05, 0.02))
        return out
    if category in ("mug", "cup", "bowl", "utensil_holder", "plate"):
        r_body = (hi[1] - lo[1]) / 2           # body radius (handle is along +x)
        inset, depth = (0.012, 0.006) if category == "plate" else (0.007, 0.016)
        zg = max(h - depth, 0.0115) if category == "plate" else h - depth
        azimuths = (180, 90, 270, 0) if category != "mug" else (180, 90, 270)
        for az in azimuths:
            a = np.radians(az)
            pos = np.array([(r_body - inset) * np.cos(a), (r_body - inset) * np.sin(a), zg])
            # fingers pinch the wall RADIALLY: Rz(fy)(0,1,0) = (cos a, sin a)  =>  fy = a - pi/2
            out.append(GraspTemplate(f"{category}_rim_topdown_az{az}", pos, a - np.pi / 2,
                                     0.045 if category != "plate" else 0.04, 0.012))
        return out
    if category == "bottle":
        return [GraspTemplate(f"bottle_cap_topdown_{d}", np.array([0.0, 0.0, h - 0.022]), np.radians(d), 0.055, 0.03)
                for d in (90, 0)]
    if category == "can":
        return [GraspTemplate(f"can_body_topdown_{d}", np.array([0.0, 0.0, h - 0.03]), np.radians(d), 0.08,
                              float(hi[1] - lo[1])) for d in (90, 0)]
    if category == "box":
        # fingers close across the thin side (object +y), fy = 0
        return [GraspTemplate("box_thin_topdown", np.array([0.0, 0.0, h - 0.03]), 0.0, 0.065, float(hi[1] - lo[1]))]
    raise KeyError(category)


def transfer_bottom(src_xy, dst_xy, src_z, dst_z, radius, lips) -> tuple[float, list]:
    """Deterministic carry height (object bottom) for RELOCATE.

    lips: list of (name, xmin, xmax, ymin, ymax, ztop) static geometry boxes. The carried object
    (footprint radius `radius`) is lifted so that its bottom clears, by TRANSFER_MARGIN, every lip
    whose xy box comes within `radius` of the straight source->destination segment, among lips
    lower than (higher support + LIP_MAX). Rack walls, tine plates, tray rims and the countertop
    edge are therefore cleared; nothing else (other objects are what the oracle checks)."""
    zs = max(src_z, dst_z)
    a, b = np.asarray(src_xy, float), np.asarray(dst_xy, float)
    pts = a[None] + np.linspace(0, 1, 60)[:, None] * (b - a)[None]
    best, crossed = zs, []
    for name, x0, x1, y0, y1, zt in lips:
        if zt >= zs + LIP_MAX or zt <= zs:
            continue
        dx = np.maximum(np.maximum(x0 - pts[:, 0], 0), pts[:, 0] - x1)
        dy = np.maximum(np.maximum(y0 - pts[:, 1], 0), pts[:, 1] - y1)
        if np.min(np.hypot(dx, dy)) < radius:
            crossed.append(name)
            best = max(best, zt)
    return best + TRANSFER_MARGIN, sorted(set(crossed))


def grasp_template(category: str, meta: dict, verts_canon: np.ndarray | None = None) -> GraspTemplate:
    return grasp_candidates(category, meta, verts_canon)[0]


def tcp_rotation(finger_yaw: float) -> np.ndarray:
    """Top-down TCP frame: z = -world z (approach), y = finger axis at `finger_yaw`."""
    y = rot_z(finger_yaw) @ np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.stack([np.cross(y, z), y, z], 1)


@dataclass
class Trajectory:
    kind: str
    tau: np.ndarray
    phase: list
    grip_pos: np.ndarray        # (N,3) TCP
    grip_quat: np.ndarray       # (N,4)
    opening: np.ndarray         # (N,)
    carried: dict = field(default_factory=dict)   # obj -> (pos (N,3), yaw (N,)) canonical pose over tau
    carry_mask: dict = field(default_factory=dict)  # obj -> bool (N,) object moving with the gripper / rack
    joints: dict = field(default_factory=dict)    # "rack1" / "door" -> (N,)
    waypoints: dict = field(default_factory=dict)  # name -> (pos, R) TCP poses for IK checks
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.tau)


def _segments_to_samples(segs, spacing):
    """segs: list of dict(p0,p1,yaw0,yaw1,w0,w1,phase,carry). Linear interpolation."""
    lengths = []
    for s in segs:
        L = np.linalg.norm(np.asarray(s["p1"]) - np.asarray(s["p0"])) + 0.05 * abs(wrap_angle(s["yaw1"] - s["yaw0"]))
        L += 0.5 * abs(s["w1"] - s["w0"])
        lengths.append(max(L, 0.01))
    total = sum(lengths)
    P, Y, W, PH, C, T = [], [], [], [], [], []
    acc = 0.0
    for s, L in zip(segs, lengths):
        n = max(2, int(np.ceil(L / spacing)) + 1)
        for i, u in enumerate(np.linspace(0, 1, n)):
            if P and i == 0:
                continue  # shared endpoint with previous segment
            P.append((1 - u) * np.asarray(s["p0"]) + u * np.asarray(s["p1"]))
            Y.append(s["yaw0"] + u * wrap_angle(s["yaw1"] - s["yaw0"]))
            W.append((1 - u) * s["w0"] + u * s["w1"])
            PH.append(s["phase"])
            C.append(s["carry"])
            T.append((acc + u * L) / total)
        acc += L
    return np.array(P), np.array(Y), np.array(W), PH, np.array(C, bool), np.array(T)


def relocate(obj: str, meta: dict, category: str, src_pos, src_yaw, dst_pos, dst_yaw, src_support_z, dst_support_z,
             grasp: GraspTemplate, spacing=COARSE_SPACING, carry_bottom: float | None = None) -> Trajectory:
    src_pos, dst_pos = np.asarray(src_pos, float), np.asarray(dst_pos, float)
    g = grasp.local_pos
    tcp_src = src_pos + rot_z(src_yaw) @ g
    tcp_dst = dst_pos + rot_z(dst_yaw) @ g
    fy_src, fy_dst = src_yaw + grasp.finger_axis_local, dst_yaw + grasp.finger_axis_local
    if carry_bottom is None:
        carry_bottom = max(src_support_z, dst_support_z) + LIFT_CLEAR
    lift_dz = carry_bottom - src_pos[2]
    tcp_lift = tcp_src + [0, 0, lift_dz]
    tcp_above_dst = tcp_dst + [0, 0, carry_bottom - dst_pos[2]]
    wo, wc = grasp.open_width, grasp.closed_width
    segs = [
        dict(p0=tcp_src + [0, 0, APPROACH_H], p1=tcp_src, yaw0=fy_src, yaw1=fy_src, w0=wo, w1=wo, phase="approach", carry=False),
        dict(p0=tcp_src, p1=tcp_src, yaw0=fy_src, yaw1=fy_src, w0=wo, w1=wc, phase="grasp", carry=False),
        dict(p0=tcp_src, p1=tcp_lift, yaw0=fy_src, yaw1=fy_src, w0=wc, w1=wc, phase="lift", carry=True),
        dict(p0=tcp_lift, p1=tcp_above_dst, yaw0=fy_src, yaw1=fy_dst, w0=wc, w1=wc, phase="transfer", carry=True),
        dict(p0=tcp_above_dst, p1=tcp_dst, yaw0=fy_dst, yaw1=fy_dst, w0=wc, w1=wc, phase="lower", carry=True),
        dict(p0=tcp_dst, p1=tcp_dst, yaw0=fy_dst, yaw1=fy_dst, w0=wc, w1=wo, phase="release", carry=False),
        dict(p0=tcp_dst, p1=tcp_dst + [0, 0, RETREAT_H], yaw0=fy_dst, yaw1=fy_dst, w0=wo, w1=wo, phase="retreat", carry=False),
    ]
    P, Y, W, PH, C, T = _segments_to_samples(segs, spacing)
    # object canonical pose over tau: rigidly attached while carried, else at src / dst
    obj_pos = np.empty_like(P)
    obj_yaw = np.empty(len(P))
    released = False
    for i in range(len(P)):
        if PH[i] == "release":
            released = True
        if C[i]:
            yaw_o = Y[i] - grasp.finger_axis_local
            obj_pos[i] = P[i] - rot_z(yaw_o) @ g
            obj_yaw[i] = yaw_o
        elif released or PH[i] == "retreat":
            obj_pos[i], obj_yaw[i] = dst_pos, dst_yaw
        else:
            obj_pos[i], obj_yaw[i] = src_pos, src_yaw
    quats = np.array([mat_to_quat(tcp_rotation(y)) for y in Y])
    wp = {
        "pregrasp": (tcp_src + [0, 0, APPROACH_H], tcp_rotation(fy_src)),
        "grasp": (tcp_src, tcp_rotation(fy_src)),
        "lift": (tcp_lift, tcp_rotation(fy_src)),
        "transfer_mid": ((tcp_lift + tcp_above_dst) / 2, tcp_rotation((fy_src + fy_dst) / 2)),
        "above_place": (tcp_above_dst, tcp_rotation(fy_dst)),
        "place": (tcp_dst, tcp_rotation(fy_dst)),
        "retreat": (tcp_dst + [0, 0, RETREAT_H], tcp_rotation(fy_dst)),
    }
    return Trajectory("relocate", T, PH, P, quats, W, carried={obj: (obj_pos, obj_yaw)},
                      carry_mask={obj: np.array([c for c in C])}, waypoints=wp,
                      meta={"object": obj, "carry_bottom_z": carry_bottom, "spacing": spacing})


def _box_world(scene, gname):
    """World AABB of a fixture collision geom (frame independent)."""
    from .assets import geom_local_points

    m, d = scene.model, scene.data
    g = m.geom(gname).id
    p = d.geom_xpos[g] + geom_local_points(m, g) @ d.geom_xmat[g].reshape(3, 3).T
    return p.min(0), p.max(0)


def rack_front_wall(scene, rack: str):
    """World AABB of the rack's front wall at the current articulation (thin in y, wide in x)."""
    best = None
    for g in scene.idx.articulated[rack]:
        lo, hi = _box_world(scene, scene.model.geom(g).name)
        if hi[1] - lo[1] < 0.03 and hi[0] - lo[0] > 0.3 and (best is None or lo[1] < best[0][1]):
            best = (lo, hi)
    return best


def tub_front_y(scene) -> float:
    """Front plane of the tub opening (minimum y of the static fixture body, door excluded)."""
    m, d = scene.model, scene.data
    from .assets import geom_local_points

    ys = []
    for g in scene.idx.env_groups["dishwasher"]:
        p = d.geom_xpos[g] + geom_local_points(m, g) @ d.geom_xmat[g].reshape(3, 3).T
        ys.append(p[:, 1].min())
    return float(np.percentile(ys, 25))


def push_rack(scene, rack: str, rack_objects: dict, spacing=COARSE_SPACING) -> Trajectory:
    """ATOMIC target skill PUSH_RACK(rack): the real prismatic joint goes from its loading pull
    to 0. The hand first grasps the top edge of the rack's front wall (fingers straddle the
    wall; there is no room for a palm in front of a pulled rack under an open door) and, once
    the hand would reach the tub's front frame, re-contacts the wall's outer face and palm-pushes
    the rest. Objects resting on the rack translate with it."""
    import mujoco

    m, d = scene.model, scene.data
    st = scene.articulation_state()
    r0 = st[rack]
    scene.set_articulation()
    mujoco.mj_kinematics(m, d)
    lo, hi = rack_front_wall(scene, rack)
    front = tub_front_y(scene)
    wall_y0 = (lo[1] + hi[1]) / 2
    grasp_tcp0 = np.array([(lo[0] + hi[0]) / 2, wall_y0, hi[2] - 0.012])
    palm_tcp0 = np.array([(lo[0] + hi[0]) / 2, lo[1] - 0.004, (lo[2] + hi[2]) / 2])
    R_grasp = tcp_rotation(0.0)
    zax, yax = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    R_palm = np.stack([np.cross(yax, zax), yax, zax], 1)
    switch_y = front - 0.115          # hand (+-0.104 m along y) must stay in front of the tub frame
    r_switch = max(0.0, r0 - (switch_y - wall_y0))
    vals = np.linspace(r0, 0.0, max(2, int(np.ceil(r0 / spacing)) + 1))
    P, Q, W, PH = [], [], [], []
    for r in vals:
        if r >= r_switch:
            P.append(grasp_tcp0 + [0, r0 - r, 0])
            Q.append(mat_to_quat(R_grasp))
            W.append(0.014)
            PH.append("push_grasp")
        else:
            P.append(palm_tcp0 + [0, r0 - r, 0])
            Q.append(mat_to_quat(R_palm))
            W.append(0.0)
            PH.append("push_palm")
    n = len(vals)
    joints = {k: np.full(n, v) for k, v in st.items()}
    joints[rack] = vals
    carried, mask = {}, {}
    for k, (pos, yaw) in rack_objects.items():
        pos = np.asarray(pos, float)
        carried[k] = (np.array([pos + [0, r0 - r, 0] for r in vals]), np.full(n, yaw))
        mask[k] = np.ones(n, bool)
    from .geom import quat_to_mat

    wp = {f"wp_{i}": (np.array(P[i]), quat_to_mat(Q[i])) for i in np.unique(np.linspace(0, n - 1, 12).astype(int))}
    return Trajectory("PUSH_RACK", np.linspace(0, 1, n), PH, np.array(P), np.array(Q), np.array(W),
                      carried=carried, carry_mask=mask, joints=joints, waypoints=wp,
                      meta={"skill": "PUSH_RACK", "rack": rack, "from": r0, "to": 0.0, "spacing": spacing,
                            "switch_rack_value": r_switch, "manipulated": rack})


def close_door(scene, contact_fraction=DOOR_CONTACT_FRACTION, spacing=COARSE_SPACING) -> Trajectory:
    """ATOMIC target skill CLOSE_DOOR: the real hinge goes from its open value to 0 while the
    closed hand pushes the door's outer face at `contact_fraction` of the door length from the
    hinge (centred across the width), moving rigidly with the door."""
    import mujoco

    from .fixture import PRIMITIVES

    m, d = scene.model, scene.data
    st = scene.articulation_state()
    q0 = st["door"]
    prim = PRIMITIVES[scene.fixture_id]
    _, _, gpos, _, gsize = prim["door_main"]       # RoboCasa door_main box, door-body frame
    jpos = m.jnt_pos[m.joint("dw_door_joint").id]
    outer_y = gpos[1] - gsize[1]
    z_low, z_high = gpos[2] - gsize[2], gpos[2] + gsize[2]
    contact_local = np.array([gpos[0], outer_y - 0.004, z_low + contact_fraction * (z_high - z_low)])
    _ = jpos
    length = (contact_fraction * (z_high - z_low))
    vals = np.linspace(q0, 0.0, max(2, int(np.ceil(length * q0 / spacing)) + 1))
    bd = m.body("dw_door").id
    P, Q = [], []
    for q in vals:
        scene.set_articulation(door=q)
        mujoco.mj_kinematics(m, d)
        R = d.xmat[bd].reshape(3, 3)
        P.append(d.xpos[bd] + R @ contact_local)
        z, y = R[:, 1], R[:, 0]
        Q.append(mat_to_quat(np.stack([np.cross(y, z), y, z], 1)))
    scene.set_articulation()
    n = len(vals)
    joints = {k: np.full(n, v) for k, v in st.items()}
    joints["door"] = vals
    from .geom import quat_to_mat

    wp = {f"wp_{i}": (np.array(P[i]), quat_to_mat(Q[i])) for i in np.unique(np.linspace(0, n - 1, 12).astype(int))}
    return Trajectory("CLOSE_DOOR", np.linspace(0, 1, n), ["close_door"] * n, np.array(P), np.array(Q),
                      np.zeros(n), joints=joints, waypoints=wp,
                      meta={"skill": "CLOSE_DOOR", "from": q0, "to": 0.0, "contact_fraction": contact_fraction,
                            "spacing": spacing, "manipulated": "door"})


def target_trajectory(scene, target: dict, rack_objects: dict, spacing=COARSE_SPACING) -> Trajectory:
    if target["skill"] == "PUSH_RACK":
        return push_rack(scene, target["rack"], rack_objects, spacing)
    if target["skill"] == "CLOSE_DOOR":
        return close_door(scene, target.get("contact_fraction", DOOR_CONTACT_FRACTION), spacing)
    raise KeyError(target)


def yaw_quat(yaw):
    return quat_from_yaw(yaw)
