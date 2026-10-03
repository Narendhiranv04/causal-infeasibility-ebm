"""Franka Panda (mujoco_menagerie) — kinematic model, IK and admissibility checks.

The robot is ONLY an admissibility filter: an action whose required waypoints admit no
collision-free IK solution is rejected from the candidate catalogue / scene. IK failure
is never recorded as a recourse failure.

TCP convention (shared with the floating gripper used by the sweep oracle):
  origin between the fingertip pads, +z = approach direction (from hand towards the
  fingertips), +y = finger opening axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import CACHE
from .assets import COLLISION_GROUP, VISUAL_GROUP, _absolutize

PANDA_DIR = CACHE / "mujoco_menagerie" / "franka_emika_panda"
TCP_OFFSET = 0.1034           # hand origin -> fingertip pad centre along hand z
FINGER_BASE_Z = 0.0584        # finger body origin in hand frame
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])
Q_SEEDS = [Q_HOME,
           np.array([0.0, 0.3, 0.0, -1.9, 0.0, 2.2, 0.785]),      # forward / low reach, top-down
           np.array([0.0, 0.2, 0.0, -1.6, 0.0, 0.3, 0.785]),      # forward, horizontal approach
           np.array([0.0, -0.6, 0.0, -2.6, 0.0, 1.9, 0.785]),     # high, close
           np.array([0.0, 0.6, 0.0, -1.2, 0.0, 1.0, 0.785])]      # long horizontal reach


def load_panda_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "panda.xml"))
    _absolutize(spec, PANDA_DIR / "assets")
    for coll in (spec.actuators, spec.tendons, spec.equalities, spec.keys):
        for x in list(coll):
            spec.delete(x)
    for g in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
        if g.contype == 0 and g.conaffinity == 0:
            g.group = VISUAL_GROUP
        else:
            g.group = COLLISION_GROUP
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0.005, 0.005], rgba=[1, 0, 0, 0])
    return spec


@dataclass
class IKResult:
    ok: bool
    q: np.ndarray
    pos_err: float
    rot_err: float
    reason: str = ""


class PandaKinematics:
    """IK + collision checks for a Panda mounted at a fixed base pose inside a scene model.

    `model` must contain the Panda attached with prefix `prefix` (see scene.SceneBuilder).
    """

    def __init__(self, model: mujoco.MjModel, prefix: str = "panda_", data: mujoco.MjData | None = None):
        self.m = model
        # share the scene's MjData so the mobile-base station and the dishwasher articulation
        # used for arm-collision checks are the scene's actual state
        self.d = data if data is not None else mujoco.MjData(model)
        self.prefix = prefix
        self.jids = [model.joint(prefix + j).id for j in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = np.array([model.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jids])
        self.site = model.site(prefix + "tcp").id
        fj = [prefix + "finger_joint1", prefix + "finger_joint2"]
        self.fadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in fj])
        robot_bodies = {b for b in range(model.nbody) if model.body(b).name.startswith(prefix)}
        self.robot_geoms = [g for g in range(model.ngeom)
                            if model.geom_bodyid[g] in robot_bodies and model.geom_group[g] == COLLISION_GROUP]
        self.link_of = {g: model.geom_bodyid[g] for g in self.robot_geoms}
        # self-collision pairs: skip same/adjacent bodies and hand/finger internal pairs
        names = {b: model.body(b).name[len(prefix):] for b in robot_bodies}
        order = {f"link{i}": i for i in range(8)}
        order.update(hand=8, left_finger=9, right_finger=9)

        def far(b1, b2):
            o1, o2 = order.get(names[b1], -9), order.get(names[b2], -9)
            return abs(o1 - o2) > 2 and not (o1 >= 7 and o2 >= 7)

        self.self_pairs = [(a, b) for i, a in enumerate(self.robot_geoms) for b in self.robot_geoms[i + 1:]
                           if far(self.link_of[a], self.link_of[b])]

    # ------------------------------------------------------------------ kinematics
    def fk(self, q, opening=0.08):
        self.d.qpos[self.qadr] = q
        self.d.qpos[self.fadr] = opening / 2
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, pos, R, q0=None, iters=150, tol_p=2e-3, tol_r=2e-2, seeds=6, rng=None) -> IKResult:
        rng = rng or np.random.default_rng(0)
        starts = ([] if q0 is None else [np.asarray(q0, float)]) + [np.clip(q, self.lo, self.hi) for q in Q_SEEDS]
        starts += [np.clip(Q_SEEDS[k % len(Q_SEEDS)] + rng.normal(0, 0.5, 7), self.lo, self.hi)
                   for k in range(max(0, seeds - 1))]
        best = None
        for q in starts:
            q = q.copy()
            for _ in range(iters):
                p, Rc = self.fk(q)
                ep = pos - p
                er = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
                if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                    break
                jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
                mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
                J = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
                e = np.concatenate([ep, er])
                lam = 0.05
                dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
                # mild pull towards home posture in the nullspace
                N = np.eye(7) - np.linalg.pinv(J) @ J
                dq += 0.05 * N @ (Q_HOME - q)
                q = np.clip(q + np.clip(dq, -0.3, 0.3), self.lo, self.hi)
            p, Rc = self.fk(q)
            pe = float(np.linalg.norm(pos - p))
            re = float(np.linalg.norm(0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))))
            res = IKResult(pe < tol_p and re < tol_r, q, pe, re)
            if res.ok:
                return res
            if best is None or pe + re < best.pos_err + best.rot_err:
                best = res
        best.reason = "no IK solution"
        return best

    # ------------------------------------------------------------------ collisions
    def self_collision(self, q, margin=0.0) -> bool:
        self.fk(q)
        ft = np.zeros(6)
        for a, b in self.self_pairs:
            if np.linalg.norm(self.d.geom_xpos[a] - self.d.geom_xpos[b]) > self.m.geom_rbound[a] + self.m.geom_rbound[b]:
                continue
            if mujoco.mj_geomDistance(self.m, self.d, a, b, 0.05, ft) < -margin:
                return True
        return False

    def env_collision(self, q, env_geoms, opening=0.08, tol=0.003, skip_hand=False) -> tuple[bool, str]:
        """Arm-vs-fixed-environment collision at configuration q (fingers at `opening`)."""
        self.fk(q, opening)
        ft = np.zeros(6)
        for a in self.robot_geoms:
            if skip_hand and self.m.body(self.link_of[a]).name[len(self.prefix):] in ("hand", "left_finger", "right_finger"):
                continue
            pa, ra = self.d.geom_xpos[a], self.m.geom_rbound[a]
            for b in env_geoms:
                if np.linalg.norm(pa - self.d.geom_xpos[b]) > ra + self.m.geom_rbound[b] + 0.01:
                    continue
                if mujoco.mj_geomDistance(self.m, self.d, a, b, 0.05, ft) < -tol:
                    return True, f"{self.m.body(self.link_of[a]).name} vs {self.m.geom(b).name}"
        return False, ""
