"""Stage 4: prescribed SE(3) action envelopes and their MuJoCo distance sweep.

The action envelope is the continuous union E(a, s) = U_{tau in [0,1]} G_a(tau).
It is approximated by samples of tau chosen by one geometric resolution
criterion: no vertex of the moving composite moves more than `step` metres
between adjacent samples (endpoints tau = 0 and tau = 1 are always included).
Samples are numerical only: they have no labels and no semantic phases.
"""

import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from poc.mj_scene import IDENTITY, Composite, GeomWorld

ENVELOPE_STEP = 2e-3  # [m] base resolution: max vertex displacement between adjacent samples


@dataclass(frozen=True)
class LinearMotion:
    """Moving-frame origin translates start -> goal at fixed orientation."""
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    quat: tuple[float, float, float, float] = IDENTITY


@dataclass(frozen=True)
class HingeMotion:
    """Moving frame rotates about `axis` through `pivot` (its origin) from theta0 to theta1 [rad]."""
    pivot: tuple[float, float, float]
    axis: tuple[float, float, float]
    theta0: float
    theta1: float


def shifted(motion, offset) -> LinearMotion | HingeMotion:
    """The same motion rigidly translated by offset (SHIFT_TARGET)."""
    add = lambda p: tuple(float(a + b) for a, b in zip(p, offset))  # noqa: E731
    if isinstance(motion, LinearMotion):
        return replace(motion, start=add(motion.start), goal=add(motion.goal))
    return replace(motion, pivot=add(motion.pivot))


def _axis_quat(axis, theta: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float) / np.linalg.norm(axis), theta)
    return q


def sample_count(motion, moving: Composite, step: float) -> int:
    """Smallest number of intervals meeting the vertex-displacement bound."""
    if isinstance(motion, LinearMotion):
        length = float(np.linalg.norm(np.subtract(motion.goal, motion.start)))
    else:
        a = np.asarray(motion.axis, dtype=float) / np.linalg.norm(motion.axis)
        v = moving.corners()
        radius = float(np.max(np.linalg.norm(v - np.outer(v @ a, a), axis=1)))  # distance to hinge axis
        length = radius * abs(motion.theta1 - motion.theta0)  # arc bounds chord
    return max(1, math.ceil(length / step))


def poses(motion, moving: Composite, step: float = ENVELOPE_STEP) -> tuple[np.ndarray, list]:
    """(taus, [(pos, quat), ...]) sampling tau in [0, 1]."""
    taus = np.linspace(0.0, 1.0, sample_count(motion, moving, step) + 1)
    if isinstance(motion, LinearMotion):
        start, goal = np.asarray(motion.start), np.asarray(motion.goal)
        return taus, [(start + tau * (goal - start), np.asarray(motion.quat)) for tau in taus]
    th = motion.theta0 + taus * (motion.theta1 - motion.theta0)
    return taus, [(np.asarray(motion.pivot, dtype=float), _axis_quat(motion.axis, t)) for t in th]


def world_corners(moving: Composite, pos, quat) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return moving.corners() @ R.reshape(3, 3).T + np.asarray(pos)


@dataclass(frozen=True)
class Sweep:
    taus: np.ndarray
    distances: np.ndarray  # (n_samples, N) signed distance per sample and entity

    @property
    def d_min(self) -> np.ndarray:
        return self.distances.min(axis=0)

    @property
    def d_start(self) -> np.ndarray:
        return self.distances[0]

    @property
    def d_goal(self) -> np.ndarray:
        return self.distances[-1]

    @property
    def tau_argmin(self) -> np.ndarray:
        return self.taus[self.distances.argmin(axis=0)]


def sweep(world: GeomWorld, motion, moving: Composite, step: float = ENVELOPE_STEP) -> Sweep:
    """Signed distance of every entity to the moving composite at every envelope sample."""
    taus, frames = poses(motion, moving, step)
    rows = []
    for pos, quat in frames:
        world.set_pose(pos, quat)
        rows.append(world.entity_distances())
    return Sweep(taus, np.array(rows))
