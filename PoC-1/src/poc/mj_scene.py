"""Stage 4: programmatic MuJoCo box scenes and a validated signed-distance query.

MuJoCo is used only as a privileged geometric oracle: static box geoms for
scene entities, one mocap body carrying the moving composite, mj_kinematics to
place it, and mj_geomDistance for signed distances. No dynamics, control, IK,
meshes, assets or XML files.

Measured behaviour of mj_geomDistance (MuJoCo 3.10, native CCD, box-box),
against analytic / QP / 15-axis-SAT truth on >1e5 random box pairs:
  * the distmax clamp is reliable: raw(distmax=e) >= e  <=>  separated by >= e
    (0 errors); it is used below as the separation certificate;
  * raw separations are exact when positive, but are occasionally reported as
    an exact 0.0 (face-to-face pairs) or, rarely, with the wrong sign;
  * raw penetrations always have the right sign and are exact for axis-aligned
    and single-axis-rotated pairs (all Stage 4 geometry); under generic 3D
    rotations EPA rarely underestimates depth (8 of ~13k, <= 10%);
  * exactly coincident geom centres report deep penetration as 0.0.
signed_distance() certifies every sign with the clamp probe, recovers a
contradicted or zero separation by bisection on the clamp, resolves coincident
centres with a 1e-9 m nudge (depth is 1-Lipschitz), and raises on the one
never-observed inconsistency instead of guessing.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from poc.types import Entity

DISTMAX = 0.5              # [m] distances >= DISTMAX are reported as DISTMAX
ZERO_PROBE = 1e-9          # [m] distmax used to test whether an exact 0.0 hides a separation
BISECTION_RES = 1e-10      # [m] resolution of the recovered separation
COINCIDENT_NUDGE = 1e-9    # [m] shift used only when two geom centres coincide exactly
IDENTITY = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Box3D:
    center: tuple[float, float, float]
    half: tuple[float, float, float]
    quat: tuple[float, float, float, float] = IDENTITY

    def corners(self) -> np.ndarray:
        """(8, 3) corner positions in the frame the box is expressed in."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(self.quat, dtype=float))
        signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
        return (signs * self.half) @ R.reshape(3, 3).T + np.asarray(self.center)


@dataclass(frozen=True)
class Entity3D:
    """Scene entity o_i: metadata plus world-frame box geoms G(o_i)."""
    entity: Entity
    boxes: tuple[Box3D, ...]

    @property
    def eid(self) -> str:
        return self.entity.entity_id


@dataclass(frozen=True)
class Composite:
    """Rigid moving geometry (object + gripper parts, or lid), boxes in the moving frame."""
    parts: tuple[tuple[str, Box3D], ...]

    def corners(self) -> np.ndarray:
        return np.concatenate([box.corners() for _, box in self.parts])


def _vec(v) -> str:
    return " ".join(repr(float(x)) for x in v)


def _geom(name: str, box: Box3D) -> str:
    return f'<geom name="{name}" type="box" pos="{_vec(box.center)}" size="{_vec(box.half)}" quat="{_vec(box.quat)}"/>'


def mjcf(entities: tuple[Entity3D, ...], moving: Composite) -> str:
    """Compact MJCF string: static entity boxes + one mocap body for the moving composite."""
    static = [_geom(f"{e.eid}__{k}", b) for e in entities for k, b in enumerate(e.boxes)]
    parts = [_geom(f"moving__{name}", b) for name, b in moving.parts]
    return ('<mujoco model="poc_stage4"><worldbody>' + "".join(static)
            + '<body name="moving" mocap="true">' + "".join(parts) + "</body></worldbody></mujoco>")


class GeomWorld:
    """Compiled model for one immutable scene; only the mocap pose changes."""

    def __init__(self, entities: tuple[Entity3D, ...], moving: Composite):
        self.model = mujoco.MjModel.from_xml_string(mjcf(entities, moving))
        self.data = mujoco.MjData(self.model)
        gid = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)  # noqa: E731
        self.entity_geoms = [[gid(f"{e.eid}__{k}") for k in range(len(e.boxes))] for e in entities]
        self.moving_geoms = [gid(f"moving__{name}") for name, _ in moving.parts]

    def set_pose(self, pos, quat) -> None:
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = quat
        mujoco.mj_kinematics(self.model, self.data)

    def _raw(self, g1: int, g2: int, distmax: float) -> float:
        return mujoco.mj_geomDistance(self.model, self.data, g1, g2, distmax, None)

    def signed_distance(self, g1: int, g2: int, distmax: float = DISTMAX) -> float:
        """Signed distance (> 0 clearance, < 0 penetration) with a certified sign."""
        r = self._raw(g1, g2, distmax)
        if self._raw(g1, g2, ZERO_PROBE) >= ZERO_PROBE:  # certified separated by >= ZERO_PROBE
            if r >= ZERO_PROBE:
                return min(r, distmax)
            lo, hi = ZERO_PROBE, distmax  # zero or wrong-sign raw value: recover the gap via the clamp
            while hi - lo > BISECTION_RES:
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if self._raw(g1, g2, mid) >= mid else (lo, mid)
            return lo
        if r >= ZERO_PROBE:
            raise RuntimeError(f"inconsistent MuJoCo distance for geoms {g1}, {g2}: raw {r} but not separated")
        if r != 0.0:
            return r
        xpos = self.data.geom_xpos
        if np.linalg.norm(xpos[g1] - xpos[g2]) <= 1e-12:  # coincident centres: nudge, query, restore
            saved = xpos[g2].copy()
            xpos[g2, 0] += COINCIDENT_NUDGE
            r = self._raw(g1, g2, distmax)
            xpos[g2] = saved
            return r
        return 0.0  # genuine touching

    def entity_distances(self) -> np.ndarray:
        """Per entity: min signed distance over (moving geom, entity geom) pairs at the current pose."""
        return np.array([min(self.signed_distance(gm, ge) for gm in self.moving_geoms for ge in geoms)
                         for geoms in self.entity_geoms])
