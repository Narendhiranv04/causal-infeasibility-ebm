"""Oracle semantics shared across stages (pure numpy; no MuJoCo import).

3D conflict contract (Stage 4). For every entity i,

    d_i = min over envelope samples tau, moving geoms g_m, entity geoms g_i of d_MuJoCo(g_m(tau), g_i)
    p_i = max(0, -d_i)                       raw penetration [m]
    c_i = max(0, p_i - CONTACT_TOL_3D)        conflict violation [m]
    F   = 1[exists i: c_i > 0],   G = sum_i c_i

so every state declared feasible has G = 0 exactly. The signed d_i is kept
alongside c_i. Diagnostic removal of an entity is an oracle query that drops
it from F; it is never an executable repair.

The repair objective J* = B F + K (B = K_max + 1) lives here once and is used
by the 2D toy oracle as well.
"""

from dataclasses import dataclass

import numpy as np

CONTACT_TOL_3D = 1e-4  # [m] penetration up to 0.1 mm counts as touching contact, not blocking


def oracle_repair_cost(F: int, K: int, k_max: int) -> int:
    """J* = B F + K with B = K_max + 1, so every feasible S beats every infeasible S."""
    if F not in (0, 1) or not 0 <= K <= k_max:
        raise ValueError(f"need F in {{0,1}} and 0 <= K <= K_max, got F={F}, K={K}, K_max={k_max}")
    return (k_max + 1) * F + K


def penetration(d) -> np.ndarray:
    return np.maximum(0.0, -np.asarray(d, dtype=float))


def conflict_3d(d, tol: float = CONTACT_TOL_3D) -> np.ndarray:
    return np.maximum(0.0, penetration(d) - tol)


def feasibility_3d(c) -> int:
    return int(np.any(np.asarray(c) > 0.0))


@dataclass(frozen=True)
class Assessment:
    """Oracle output for one scene/action; tuples are ordered like the scene entities."""
    ids: tuple[str, ...]
    d: tuple[float, ...]        # signed min distance over the whole envelope
    d_start: tuple[float, ...]  # signed distance at tau = 0 only
    d_goal: tuple[float, ...]   # signed distance at tau = 1 only
    p: tuple[float, ...]
    c: tuple[float, ...]
    G: float
    F: int
    F_start: int
    F_goal: int

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(i for i, ci in zip(self.ids, self.c) if ci > 0.0)

    @property
    def interior_only(self) -> bool:
        """Both endpoint poses are feasible but the envelope is not."""
        return self.F == 1 and self.F_start == 0 and self.F_goal == 0


def assess(ids, d, d_start, d_goal, tol: float = CONTACT_TOL_3D) -> Assessment:
    c = conflict_3d(d, tol)
    return Assessment(tuple(ids), tuple(map(float, d)), tuple(map(float, d_start)), tuple(map(float, d_goal)),
                      tuple(penetration(d).tolist()), tuple(c.tolist()), float(np.sum(c)), feasibility_3d(c),
                      feasibility_3d(conflict_3d(d_start, tol)), feasibility_3d(conflict_3d(d_goal, tol)))


def feasibility_excluding(a: Assessment, excluded) -> int:
    """Diagnostic query: F with the geoms of `excluded` entities ignored."""
    return feasibility_3d([ci for i, ci in zip(a.ids, a.c) if i not in set(excluded)])


def is_valid(static_distances) -> int:
    """V = 1 iff no pair of static post-intervention bodies penetrates beyond CONTACT_TOL_3D."""
    return int(feasibility_3d(conflict_3d(static_distances)) == 0)


def admissible_minimal_repairs(F, V, K, k_max: int) -> frozenset[int]:
    """Exact S* over admissible subsets only (V = 1): argmin of J* = B F + K, feasible ones kept.

    Validity is a filter, not a penalty inside J*, so an invalid repaired state can never win.
    """
    admissible = [x for x in range(len(F)) if V[x] == 1]
    J = {x: oracle_repair_cost(int(F[x]), int(K[x]), k_max) for x in admissible}
    j_min = min(J.values())
    return frozenset(x for x, j in J.items() if j == j_min and F[x] == 0)


def diagnostic_causes(a: Assessment) -> tuple[str, ...]:
    """Minimal diagnostic cause set: ignoring all causes restores feasibility, each is necessary."""
    causes = a.blockers
    if feasibility_excluding(a, causes) != 0:
        raise AssertionError("ignoring every blocker must restore feasibility")
    if any(feasibility_excluding(a, [k for k in causes if k != j]) == 0 for j in causes):
        raise AssertionError("every diagnostic cause must be necessary")
    return causes
