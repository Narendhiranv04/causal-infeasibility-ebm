"""Exact convex-piece distance used to verify MuJoCo near contact.

MuJoCo's convex distance (mj_geomDistance) is fast but, for some convex-mesh pairs, returns
wrong values near contact: spurious exact zeros, and even a negative "penetration" for pieces
that are 2.2 cm apart (native CCD; legacy libccd returned +0.4 cm for the same pair). Every
decision-relevant pair (MuJoCo distance < VERIFY_BELOW) is therefore recomputed here:

  * intersection: exact LP feasibility  exists convex weights a, b with  A^T a = B^T b;
  * separated:    exact minimum-norm point of conv(A - B) (Minkowski difference) via NNLS
                  with a heavily weighted sum-to-one row;
  * intersecting: penetration depth = min over candidate axes (face normals of both pieces)
                  of the projection overlap (separating-axis bound; an upper bound of the
                  true depth, tight for the face-face contacts that dominate here).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, nnls
from scipy.spatial import ConvexHull

VERIFY_BELOW = 0.012


def hull_normals(points: np.ndarray) -> np.ndarray:
    try:
        h = ConvexHull(points)
        n = h.equations[:, :3]
    except Exception:  # noqa: BLE001  (degenerate / flat pieces)
        n = np.eye(3)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    # deduplicate (up to sign)
    keep = []
    for v in n:
        if not any(abs(abs(v @ k) - 1) < 1e-6 for k in keep):
            keep.append(v)
    return np.array(keep)


def intersects(va: np.ndarray, vb: np.ndarray) -> bool:
    na, nb = len(va), len(vb)
    A = np.zeros((5, na + nb))
    A[:3, :na] = va.T
    A[:3, na:] = -vb.T
    A[3, :na] = 1
    A[4, na:] = 1
    r = linprog(np.zeros(na + nb), A_eq=A, b_eq=[0, 0, 0, 1, 1], bounds=(0, None), method="highs")
    return r.status == 0


def separation(va: np.ndarray, vb: np.ndarray) -> float:
    D = (va[:, None, :] - vb[None, :, :]).reshape(-1, 3)
    # prune: only extreme Minkowski points matter; keep the closest 400 to the origin plus hull
    if len(D) > 600:
        idx = np.argsort(np.linalg.norm(D, axis=1))[:400]
        try:
            idx = np.union1d(idx, ConvexHull(D).vertices)
        except Exception:  # noqa: BLE001
            pass
        D = D[idx]
    M = 1e3
    A = np.vstack([D.T, M * np.ones(len(D))])
    w, _ = nnls(A, np.array([0.0, 0.0, 0.0, M]), maxiter=50 * len(D))
    return float(np.linalg.norm(D.T @ (w / max(w.sum(), 1e-12))))


def penetration_depth(va, vb, na, nb) -> float:
    axes = np.vstack([na, nb])
    pa, pb = va @ axes.T, vb @ axes.T
    overlap = np.minimum(pa.max(0) - pb.min(0), pb.max(0) - pa.min(0))
    return float(max(overlap.min(), 0.0))


def exact_signed_distance(va, vb, na, nb) -> float:
    if intersects(va, vb):
        return -penetration_depth(va, vb, na, nb)
    return separation(va, vb)
