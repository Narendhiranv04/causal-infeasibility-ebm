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
LIFT_CLEAR = 0.13
COARSE_SPACING = 0.01
FINE_SPACING = 0.0025
DOOR_PUSH_BELOW_TOP = 0.25
PUSH_SWITCH_Y = -0.37   # world y of the rack front wall where the push switches grasp -> palm


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
             grasp: GraspTemplate, spacing=COARSE_SPACING) -> Trajectory:
    src_pos, dst_pos = np.asarray(src_pos, float), np.asarray(dst_pos, float)
    g = grasp.local_pos
    tcp_src = src_pos + rot_z(src_yaw) @ g
    tcp_dst = dst_pos + rot_z(dst_yaw) @ g
    fy_src, fy_dst = src_yaw + grasp.finger_axis_local, dst_yaw + grasp.finger_axis_local
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


def close_dishwasher(scene, rack_objects: dict, spacing=COARSE_SPACING) -> Trajectory:
    """Target action. rack_objects: obj -> (canonical pos, yaw) for objects resting on the
    upper rack (they translate with it during the push)."""
    import mujoco

    m, d = scene.model, scene.data
    t = scene.articulation_targets()
    r0, q0 = t["rack1"], t["door"]
    # push phase: palm on the rack front wall (outer face centre), approach +y, fingers closed
    scene.set_articulation()
    scene.forward()
    front = None
    for g in scene.idx.articulated["rack1"]:
        from .assets import geom_local_points

        p = d.geom_xpos[g] + geom_local_points(m, g) @ d.geom_xmat[g].reshape(3, 3).T
        lo, hi = p.min(0), p.max(0)
        if hi[1] - lo[1] < 0.02 and hi[0] - lo[0] > 0.4 and (front is None or lo[1] < front[0][1]):
            front = (lo, hi)
    lo, hi = front
    # Stage A1: top-down grasp on the front wall's top edge (fingers straddle the 12 mm wall)
    # while the rack is far out -- there is no room for a palm push there (only ~6 cm
    # separate the pulled rack from the open door plate). Stage A2: once the wall is
    # PUSH_SWITCH_Y from the tub ceiling the hand would hit the ceiling, so the robot
    # re-contacts the wall's outer face and palm-pushes the remaining distance.
    wall_y0 = (lo[1] + hi[1]) / 2
    grasp_tcp0 = np.array([(lo[0] + hi[0]) / 2, wall_y0, hi[2] - 0.012])
    palm_tcp0 = np.array([(lo[0] + hi[0]) / 2, lo[1] - 0.004, (lo[2] + hi[2]) / 2])
    R_grasp = tcp_rotation(0.0)
    zax, yax = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    R_palm = np.stack([np.cross(yax, zax), yax, zax], 1)
    r_switch = max(0.0, r0 - (PUSH_SWITCH_Y - wall_y0))
    n_push = max(2, int(np.ceil(r0 / spacing)) + 1)
    rack_vals = np.linspace(r0, 0.0, n_push)
    door_top_arc = 0.72 * q0
    n_door = max(2, int(np.ceil(door_top_arc / spacing)) + 1)
    door_vals = np.linspace(q0, 0.0, n_door)
    P, Q, W, PH, J1, JD = [], [], [], [], [], []
    for r in rack_vals:
        if r >= r_switch:
            P.append(grasp_tcp0 + [0, r0 - r, 0])
            Q.append(mat_to_quat(R_grasp))
            W.append(0.014)
        else:
            P.append(palm_tcp0 + [0, r0 - r, 0])
            Q.append(mat_to_quat(R_palm))
            W.append(0.0)
        PH.append("push_rack")
        J1.append(r)
        JD.append(q0)
    # Door contact point in the door BODY frame (the convexified door_main mesh geom is expressed
    # in its principal-inertia frame, whose axes MuJoCo may permute). Values are the RoboCasa
    # door_main box (local centre (-0.0008, -0.2744, -0.0438), half-size (0.316, 0.0035, 0.3594)).
    bd = m.body("dw_door").id
    door_outer_y = -0.27445 - 0.00351
    door_top_z = -0.04382 + 0.35936
    contact_local = np.array([-0.0008, door_outer_y - 0.004, door_top_z - DOOR_PUSH_BELOW_TOP])
    for q in door_vals:
        scene.set_articulation(door=q, rack1=0.0)
        mujoco.mj_kinematics(m, d)
        R = d.xmat[bd].reshape(3, 3)
        p = d.xpos[bd] + R @ contact_local
        z = R[:, 1]           # approach: into the door (door +y)
        y = R[:, 0]           # finger axis along the door width
        P.append(p)
        Q.append(mat_to_quat(np.stack([np.cross(y, z), y, z], 1)))
        W.append(0.0)
        PH.append("close_door")
        J1.append(0.0)
        JD.append(q)
    scene.set_articulation()
    P = np.array(P)
    n1 = len(rack_vals)
    tau = np.concatenate([np.linspace(0, 0.5, n1), np.linspace(0.5, 1.0, len(door_vals) + 1)[1:]])
    carried, mask = {}, {}
    for k, (pos, yaw) in rack_objects.items():
        pos = np.asarray(pos, float)
        cp = np.array([pos + [0, r0 - r, 0] for r in J1])
        carried[k] = (cp, np.full(len(J1), yaw))
        mask[k] = np.ones(len(J1), bool)
    from .geom import quat_to_mat

    wp = {f"push_{i}": (P[i], quat_to_mat(Q[i])) for i in np.linspace(0, n1 - 1, 6).astype(int)}
    for i in np.linspace(n1, len(P) - 1, 5).astype(int):
        wp[f"door_{i}"] = (P[i], quat_to_mat(Q[i]))
    return Trajectory("close_dishwasher", tau, PH, P, np.array(Q), np.array(W), carried=carried, carry_mask=mask,
                      joints={"rack1": np.array(J1), "door": np.array(JD)}, waypoints=wp,
                      meta={"spacing": spacing, "phases": {"push_rack": [0.0, 0.5], "close_door": [0.5, 1.0]},
                            "rack_objects": sorted(rack_objects)})


def yaw_quat(yaw):
    return quat_from_yaw(yaw)
