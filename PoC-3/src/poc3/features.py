"""PoC-3 Stage 1: privileged-state feature contract (plan3.md sections 11-14, approved 13.2 amendment).

Features are a pure function of scene GEOMETRY, the prescribed action and the candidate catalogue. They
never call the oracle (no sweep, signed distance, assessment or repair table), so F, G, c, B0, C*, S*,
oracle alpha / beta, validity edges, repair status, family name, ids and seeds cannot enter. Repair cost
is not a feature: it enters the energy exactly through K(x). All lengths are scaled by L0 = 0.5 m.

FIXED CONVENTIONS
  Entities     every static scene entity plus the target fixture (when present) as one extra entity.
               Centre and half-extents come from the union AABB of all the entity's box corners (never a
               single-box assumption). Role one-hot: (movable, structural, target_fixture).
  Action       K_A = 8 samples at global normalized progress tau_k = k / 7 of the moving-frame pose:
               position / L0 (3) + rotation 6D (6) = first two columns of the 3x3 rotation matrix,
               flattened column-major [R00 R10 R20 R01 R11 R21]. A Polyline of n segments assigns
               segment j to tau in [j/n, (j+1)/n] (tau = 1 is the end of the last segment); a hinge
               keeps its pivot as frame origin and rotates. Kinematic samples only: they are NOT
               PoC-2's collision-adaptive sweep samples.
  Moving       local AABB of the moving composite's corners in the moving frame:
               [c_local / L0 (3), h_local / L0 (3)], c = (min + max) / 2, h = (max - min) / 2.
  Candidates   u_p = [kind one-hot (relocate, shift_target) (2), delta / L0 (3), proposed centre / L0 (3),
               region centre / L0 (3), region half / L0 (3)].
               RELOCATE: proposed centre = the relocated entity's actual union-AABB centre (PoC-2
               apply_option), delta = proposed - current centre; region centre / half kept separate.
               SHIFT_TARGET: delta = shift vector; proposed centre = shifted action reference point
               (LinearMotion start, first Polyline segment start, hinge pivot); region features zero.
  Affected     RELOCATE: index of the relocated entity. SHIFT_TARGET: always NO_ENTITY = -1 (zero
               affected-entity embedding), with or without a fixture. A shift is a global action-envelope
               intervention, not a local edit of the static fixture; its context comes from the scene /
               action / moving-composite encoding, kind, shift vector and shifted reference point, and
               fixture / static compatibility is handled by the exact V.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from poc.envelope import HingeMotion, LinearMotion
from poc.types import EntityRole, InterventionKind
from poc2.oracle import Polyline, apply_option

L0 = 0.5     # [m] fixed physical scale
K_A = 8      # action samples
ROLES = ("movable", "structural", "target_fixture")
ENTITY_COLUMNS = ("x", "y", "z", "hx", "hy", "hz") + tuple(f"role_{r}" for r in ROLES)
ACTION_COLUMNS = ("x", "y", "z", "R00", "R10", "R20", "R01", "R11", "R21")
MOVING_COLUMNS = ("cx_local", "cy_local", "cz_local", "hx_local", "hy_local", "hz_local")
CANDIDATE_COLUMNS = (("kind_relocate", "kind_shift_target", "dx", "dy", "dz", "px", "py", "pz")
                     + ("region_cx", "region_cy", "region_cz", "region_hx", "region_hy", "region_hz"))
NO_ENTITY = -1


@dataclass(frozen=True)
class SceneFeatures:
    """Model input for one scene (the complete contract; nothing else is fed to the model)."""
    entities: np.ndarray    # (N, 9)
    action: np.ndarray      # (K_A, 9)
    moving: np.ndarray      # (6,)
    candidates: np.ndarray  # (P, 14)
    affected: np.ndarray    # (P,) int entity index, NO_ENTITY = none

    def __post_init__(self) -> None:
        N, P = len(self.entities), len(self.candidates)
        if self.entities.shape != (N, len(ENTITY_COLUMNS)) or self.action.shape != (K_A, len(ACTION_COLUMNS)):
            raise ValueError("entity / action tensor shape violates the feature contract")
        if self.moving.shape != (len(MOVING_COLUMNS),) or self.candidates.shape != (P, len(CANDIDATE_COLUMNS)):
            raise ValueError("moving / candidate tensor shape violates the feature contract")
        if self.affected.shape != (P,) or not np.all((self.affected >= NO_ENTITY) & (self.affected < N)):
            raise ValueError("affected-entity indices out of range")


def aabb(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    return (lo + hi) / 2, (hi - lo) / 2


def boxes_aabb(boxes) -> tuple[np.ndarray, np.ndarray]:
    """Union AABB of one or more PoC-1 Box3D (rotated boxes via their corners)."""
    return aabb(np.concatenate([b.corners() for b in boxes]))


def rot6d(quat) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)[:, :2].T.reshape(6)


def _axis_quat(axis, theta: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float) / np.linalg.norm(axis), theta)
    return q


def frame_at(motion, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Moving-frame (position, quaternion) at global normalized progress tau in [0, 1]."""
    if isinstance(motion, Polyline):
        n = len(motion.segments)
        j = min(int(np.floor(tau * n)), n - 1)
        return frame_at(motion.segments[j], tau * n - j)
    if isinstance(motion, LinearMotion):
        start, goal = np.asarray(motion.start, dtype=float), np.asarray(motion.goal, dtype=float)
        return start + tau * (goal - start), np.asarray(motion.quat, dtype=float)
    if isinstance(motion, HingeMotion):
        theta = motion.theta0 + tau * (motion.theta1 - motion.theta0)
        return np.asarray(motion.pivot, dtype=float), _axis_quat(motion.axis, theta)
    raise TypeError(f"unsupported motion {type(motion).__name__}")


def reference_point(motion) -> np.ndarray:
    """Action reference point moved by SHIFT_TARGET: linear start, first polyline start, hinge pivot."""
    if isinstance(motion, Polyline):
        return reference_point(motion.segments[0])
    return np.asarray(motion.pivot if isinstance(motion, HingeMotion) else motion.start, dtype=float)


def action_tensor(motion) -> np.ndarray:
    rows = []
    for tau in np.arange(K_A) / (K_A - 1):
        pos, quat = frame_at(motion, float(tau))
        rows.append(np.concatenate([pos / L0, rot6d(quat)]))
    return np.array(rows)


def moving_tensor(moving) -> np.ndarray:
    c, h = aabb(moving.corners())
    return np.concatenate([c, h]) / L0


def _role(role: EntityRole) -> np.ndarray:
    name = {EntityRole.MOVABLE: "movable", EntityRole.STRUCTURAL: "structural"}.get(role, "target_fixture")
    return np.array([name == r for r in ROLES], dtype=float)


def entity_tensor(scene) -> np.ndarray:
    rows = []
    for e in scene.entities:
        c, h = boxes_aabb(e.boxes)
        rows.append(np.concatenate([c / L0, h / L0, _role(e.entity.role)]))
    if scene.fixture:
        c, h = boxes_aabb(scene.fixture)
        rows.append(np.concatenate([c / L0, h / L0, _role(EntityRole.TARGET)]))
    return np.array(rows).reshape(-1, len(ENTITY_COLUMNS))


def candidate_row(scene, regions: dict, option) -> tuple[np.ndarray, int]:
    iv = option.intervention
    ids = [e.eid for e in scene.entities]
    zero = np.zeros(3)
    if iv.kind is InterventionKind.RELOCATE:
        i = ids.index(iv.entity_id)
        now = boxes_aabb(scene.entities[i].boxes)[0]
        after = boxes_aabb(apply_option(scene, iv).entities[i].boxes)[0]
        reg = regions[option.region_id]
        parts = [(1.0, 0.0), after - now, after, reg.center, reg.half]
        return np.concatenate([np.asarray(parts[0]), *(np.asarray(v, dtype=float) / L0 for v in parts[1:])]), i
    if iv.kind is InterventionKind.SHIFT_TARGET:
        delta = np.asarray(iv.params, dtype=float)
        row = np.concatenate([(0.0, 1.0), delta / L0, (reference_point(scene.motion) + delta) / L0, zero, zero])
        return row, NO_ENTITY
    raise ValueError(f"{iv.kind} is not an executable repair kind")


def extract(scene, regions, options) -> SceneFeatures:
    """The full model input for one scene; ids, names and costs are deliberately dropped."""
    by_id = {r.region_id: r for r in regions}
    rows = [candidate_row(scene, by_id, o) for o in options]
    cands = np.array([r for r, _ in rows]).reshape(-1, len(CANDIDATE_COLUMNS))
    return SceneFeatures(entity_tensor(scene), action_tensor(scene.motion), moving_tensor(scene.moving), cands,
                         np.array([a for _, a in rows], dtype=np.int64))


def contract() -> dict:
    """Machine-readable feature contract (column names, scale, conventions)."""
    return {"L0_m": L0, "K_A": K_A, "entity_columns": list(ENTITY_COLUMNS), "action_columns": list(ACTION_COLUMNS),
            "moving_columns": list(MOVING_COLUMNS), "candidate_columns": list(CANDIDATE_COLUMNS),
            "affected": "RELOCATE: relocated entity index; SHIFT_TARGET: always -1 (NO_ENTITY) -> zero embedding",
            "conventions": __doc__.split("FIXED CONVENTIONS")[1].strip()}
