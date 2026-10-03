"""Swept-volume oracle on actual MuJoCo collision geometry.

For an action a (a sampled Trajectory) and a static entity e (an object at an exact pose,
or an environment part) we compute the signed distance profile

    d_e(tau) = min over (moving geom, entity geom) pairs of mj_geomDistance(...)

where the moving set at tau is {gripper hand + fingers at the commanded opening, carried
object(s), articulated fixture bodies at the interpolated real joint values}. Distances
are exact GJK/EPA results on convex collision pieces (penetration is negative); a
bounding-sphere prefilter only skips pairs that provably cannot be closer than CAP.

Labels derived here (F, blockers, minimum distances, tau*, conflict magnitude) are
SUPERVISION ONLY and are written to the `labels` block of the record schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .geom import quat_from_yaw, quat_to_mat
from .scene import Scene
from .assets import body_pose_from_canonical, geom_local_points
from .exactgeom import VERIFY_BELOW, exact_signed_distance, hull_normals

CAP = 0.05          # distances above CAP are reported as CAP (">= CAP"). MuJoCo's convex
                    # mesh distance occasionally returns a spurious exact 0.0 (zero witness
                    # points) for large caps (observed at 0.10, never at <= 0.05); _dist also
                    # detects that signature and retries with a smaller cap.
PEN_TOL = 0.004     # conflict  : min distance <= -PEN_TOL (>= 4 mm interpenetration)
CLEAR_TOL = 0.004   # clear     : min distance >= +CLEAR_TOL
ENV_TOL = 0.002     # environment contact tolerance (resting contact numerics)


def status_of(dist: float) -> str:
    if dist <= -PEN_TOL:
        return "conflict"
    if dist >= CLEAR_TOL:
        return "clear"
    return "ambiguous"


@dataclass
class Profile:
    """Distance profile of one (action, entity) pair."""
    min_dist: float
    tau_star: float
    phase_star: str
    part_star: str
    dist: np.ndarray                    # (N,) capped per-tau distance
    phases_in_conflict: list = field(default_factory=list)
    parts_in_conflict: list = field(default_factory=list)

    @property
    def status(self) -> str:
        return status_of(self.min_dist)

    def to_dict(self):
        return {"min_dist": round(float(self.min_dist), 5), "tau_star": round(float(self.tau_star), 4),
                "phase_star": self.phase_star, "part_star": self.part_star, "status": self.status,
                "conflict_magnitude": round(float(max(0.0, -self.min_dist)), 5),
                "phases_in_conflict": self.phases_in_conflict, "parts_in_conflict": self.parts_in_conflict}


@dataclass
class MovingSet:
    """Per-tau world poses of every moving collision geom of an action."""
    traj: object
    geoms: np.ndarray        # (G,) geom ids
    parts: list              # (G,) part label per geom
    pos: np.ndarray          # (N, G, 3)
    mat: np.ndarray          # (N, G, 3, 3)
    rbound: np.ndarray       # (G,)


class SweepEngine:
    def __init__(self, scene: Scene):
        self.sc = scene
        self.m, self.d = scene.model, scene.data
        self.ft = np.zeros(6)
        self.glitches = 0          # number of MuJoCo near-contact results corrected by the exact check
        self._local_cache = {}
        self._obj_local = {}
        self._calibrate_objects()

    # ------------------------------------------------------------------ object geom poses
    def _calibrate_objects(self):
        m, d = self.m, self.d
        for k in self.sc.objects:
            mid = self.sc.idx.obj_mocap[k]
            d.mocap_pos[mid] = [0, 0, 0]
            d.mocap_quat[mid] = [1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        for k, geoms in self.sc.idx.obj_geoms.items():
            g = np.array(geoms)
            self._obj_local[k] = (g, d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(-1, 3, 3).copy())
        for k in self.sc.objects:
            self.sc.hide_object(k)
        mujoco.mj_kinematics(m, d)

    def object_geoms_at(self, key: str, canon_pos, yaw):
        g, lp, lR = self._obj_local[key]
        bpos, bquat = body_pose_from_canonical(self.sc.objects[key].asset.meta, canon_pos, yaw)
        Rb = quat_to_mat(bquat)
        return g, bpos + lp @ Rb.T, np.einsum("ij,gjk->gik", Rb, lR)

    # ------------------------------------------------------------------ moving set
    def moving_set(self, traj, include_gripper=True, include_articulated=(), carried=None) -> MovingSet:
        """Run kinematics at every tau and record moving geom poses.

        carried: subset of traj.carried keys whose geoms are part of the moving set (all by default).
        Objects that are carried only for part of the trajectory are still recorded at every tau
        (at rest they sit at their source/destination pose) but masked out where not carried.
        """
        sc, m, d = self.sc, self.m, self.d
        carried = list(traj.carried) if carried is None else list(carried)
        geoms, parts = [], []
        if include_gripper:
            geoms += sc.idx.gripper_geoms
            parts += ["gripper"] * len(sc.idx.gripper_geoms)
        for a in include_articulated:
            geoms += sc.idx.articulated[a]
            parts += [a] * len(sc.idx.articulated[a])
        for k in carried:
            geoms += sc.idx.obj_geoms[k]
            parts += [f"carried:{k}"] * len(sc.idx.obj_geoms[k])
        geoms = np.array(geoms)
        N = traj.n
        P = np.zeros((N, len(geoms), 3))
        R = np.zeros((N, len(geoms), 3, 3))
        for i in range(N):
            sc.set_gripper(traj.grip_pos[i], quat=traj.grip_quat[i], opening=traj.opening[i])
            if traj.joints:
                sc.set_articulation(**{k: v[i] for k, v in traj.joints.items()})
            else:
                sc.set_articulation()
            for k in carried:
                pos, yaw = traj.carried[k]
                sc.set_object_canonical(k, pos[i], yaw[i])
            mujoco.mj_kinematics(m, d)
            P[i] = d.geom_xpos[geoms]
            R[i] = d.geom_xmat[geoms].reshape(-1, 3, 3)
        sc.set_articulation()
        sc.hide_gripper()
        for k in carried:
            sc.hide_object(k)
        mujoco.mj_kinematics(m, d)
        return MovingSet(traj, geoms, parts, P, R, m.geom_rbound[geoms].copy())

    # ------------------------------------------------------------------ distances
    def _dist(self, ga, pa, Ra, gb, pb, Rb, cap):
        d = self.d
        d.geom_xpos[ga] = pa
        d.geom_xmat[ga] = Ra.reshape(9)
        d.geom_xpos[gb] = pb
        d.geom_xmat[gb] = Rb.reshape(9)
        dist = mujoco.mj_geomDistance(self.m, d, int(ga), int(gb), cap, self.ft)
        if dist < VERIFY_BELOW:   # decision-relevant: recompute exactly (see exactgeom.py)
            va = pa + self._local(ga)[0] @ np.asarray(Ra).reshape(3, 3).T
            vb = pb + self._local(gb)[0] @ np.asarray(Rb).reshape(3, 3).T
            na = self._local(ga)[1] @ np.asarray(Ra).reshape(3, 3).T
            nb = self._local(gb)[1] @ np.asarray(Rb).reshape(3, 3).T
            exact = exact_signed_distance(va, vb, na, nb)
            if abs(exact - dist) > 1e-3:
                self.glitches += 1
            dist = min(exact, cap)
        return dist

    def _local(self, g):
        if g not in self._local_cache:
            pts = geom_local_points(self.m, g)
            self._local_cache[g] = (pts, hull_normals(pts))
        return self._local_cache[g]

    def profile(self, ms: MovingSet, s_geoms, s_pos, s_mat, part_mask=None, tau_mask=None, cap=CAP,
                s_parts=None) -> Profile:
        """Distance profile between a moving set and a geom set B.

        B poses are either fixed ((S,3)/(S,3,3)) or per-tau ((N,S,3)/(N,S,3,3)).
        part_mask: optional (N, G) bool — which moving geoms are active at each tau.
        """
        traj = ms.traj
        N, G = ms.pos.shape[:2]
        s_geoms = np.asarray(s_geoms)
        if s_pos.ndim == 2:
            s_pos = np.broadcast_to(s_pos, (N,) + s_pos.shape)
            s_mat = np.broadcast_to(s_mat, (N,) + s_mat.shape)
        s_r = self.m.geom_rbound[s_geoms]
        lb = np.linalg.norm(ms.pos[:, :, None, :] - s_pos[:, None, :, :], axis=-1) \
            - ms.rbound[None, :, None] - s_r[None, None, :]
        if part_mask is not None:
            lb = np.where(part_mask[:, :, None], lb, np.inf)
        if tau_mask is not None:
            lb = np.where(tau_mask[:, None, None], lb, np.inf)
        dist = np.full(N, cap)
        who = np.full(N, -1)
        whob = np.full(N, -1)
        for i in np.where(lb.min(axis=(1, 2)) < cap)[0]:
            cand = np.argwhere(lb[i] < cap)
            order = np.argsort(lb[i][cand[:, 0], cand[:, 1]])
            best = cap
            for j in order:
                a, b = cand[j]
                if lb[i, a, b] >= best:
                    break
                dd = self._dist(ms.geoms[a], ms.pos[i, a], ms.mat[i, a], s_geoms[b], s_pos[i, b], s_mat[i, b], cap)
                if dd < best:
                    best, who[i], whob[i] = dd, a, b
            dist[i] = min(best, cap)
        i_star = int(np.argmin(dist))
        md = float(dist[i_star])
        bad = np.where(dist <= -PEN_TOL)[0]

        def lab(i):
            a = ms.parts[who[i]] if who[i] >= 0 else ""
            if s_parts is not None and whob[i] >= 0:
                a += f"|{s_parts[whob[i]]}"
            return a

        return Profile(
            min_dist=md, tau_star=float(traj.tau[i_star]), phase_star=traj.phase[i_star], part_star=lab(i_star),
            dist=dist, phases_in_conflict=sorted({traj.phase[i] for i in bad}),
            parts_in_conflict=sorted({lab(i) for i in bad}),
        )

    def carried_object_set(self, traj, key, canon_pos, yaw, offsets) -> MovingSet:
        """Object `key` translated by per-tau offsets (N,3) from a canonical pose (rack-carried)."""
        g, p0, R0 = self.object_geoms_at(key, canon_pos, yaw)
        P = p0[None] + np.asarray(offsets)[:, None, :]
        R = np.broadcast_to(R0, (traj.n,) + R0.shape)
        return MovingSet(traj, g, [f"object:{key}"] * len(g), P, R, self.m.geom_rbound[g].copy())

    def static_geoms(self, geoms):
        """Current (scene) world poses of static geoms (environment)."""
        self.sc.set_articulation()
        mujoco.mj_kinematics(self.m, self.d)
        g = np.asarray(geoms)
        return g, self.d.geom_xpos[g].copy(), self.d.geom_xmat[g].reshape(-1, 3, 3).copy()

    def static_distance(self, key_a, pose_a, key_b, pose_b, cap=CAP) -> float:
        """Distance between two objects at fixed canonical poses (placement overlap tests)."""
        ga, pa, Ra = self.object_geoms_at(key_a, *pose_a)
        gb, pb, Rb = self.object_geoms_at(key_b, *pose_b)
        ra, rb = self.m.geom_rbound[ga], self.m.geom_rbound[gb]
        lb = np.linalg.norm(pa[:, None] - pb[None], axis=-1) - ra[:, None] - rb[None]
        best = cap
        for a, b in sorted(np.argwhere(lb < cap), key=lambda ab: lb[ab[0], ab[1]]):
            if lb[a, b] >= best:
                break
            best = min(best, self._dist(ga[a], pa[a], Ra[a], gb[b], pb[b], Rb[b], cap))
        return float(best)
