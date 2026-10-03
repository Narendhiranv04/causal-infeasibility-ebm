"""Small rigid-transform helpers. Quaternions are MuJoCo order (w, x, y, z) internally;
records exported to JSON use (x, y, z, qx, qy, qz, qw) as requested in the spec."""

from __future__ import annotations

import numpy as np


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def pose7(pos, quat_wxyz) -> list[float]:
    """Export pose as [x, y, z, qx, qy, qz, qw] (rounded for stable JSON)."""
    w, x, y, z = quat_wxyz
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return [round(float(v), 6) for v in (*pos, x, y, z, w)]


def from_pose7(p):
    p = list(p)
    return np.array(p[:3], float), np.array([p[6], p[3], p[4], p[5]], float)
