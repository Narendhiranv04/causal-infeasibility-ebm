"""Stage 1: 2D envelope oracle and exhaustive intervention ground truth.

Conflict metric stated for Stage 1 (plan.md section 3):

    c_i = phi(E, G(o_i)) = max over sampled envelope poses g of pen(g, G(o_i))

E is approximated by densely sampled rigid poses of every moving part (carried
object + gripper, or hinged panel). pen is the separating-axis minimum
translation distance between two convex polygons (0 when disjoint): how far
o_i intrudes into geometry the action occupies, in metres. Samples are a
numerical approximation of the union over tau; they carry no labels and are
never exposed as phases.

c(S), G(S) and F(S) are separate: G grades the conflict vector, F certifies
feasibility from c with a named contact tolerance, J* ranks repairs.
"""

import math
from dataclasses import dataclass, replace
from functools import reduce

import numpy as np

from poc.types import ActionSpec, Entity, EntityRole, Intervention, InterventionKind, RepairResult

CONTACT_TOL = 1e-6   # [m] intrusion <= this is touching contact, not blocking
SAMPLE_STEP = 5e-4   # [m] max displacement of any envelope vertex between consecutive samples
MAX_CANDIDATES = 10  # Stage 1 bound on P so exhaustive enumeration stays trivial

G_FORMS = {  # candidate graded-conflict functions g(c^S), compared in scripts/s1_toy.py
    "l1": lambda c: float(np.sum(c)),                # total intrusion depth [m]
    "l2sq": lambda c: float(np.sum(c * c)),          # sum of squared intrusion depths [m^2]
    "max": lambda c: float(np.max(c, initial=0.0)),  # worst single intrusion [m]
}
G_FORM = "l1"  # chosen by the Stage 1 graded-conflict study


def rect(center, size, angle: float = 0.0) -> np.ndarray:
    """Counter-clockwise (4, 2) vertices of a rotated rectangle."""
    hw, hh = size[0] / 2.0, size[1] / 2.0
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    c, s = math.cos(angle), math.sin(angle)
    return local @ np.array([[c, s], [-s, c]]) + np.asarray(center, dtype=float)


def _unit_normals(polys: np.ndarray) -> np.ndarray:
    edges = np.roll(polys, -1, axis=-2) - polys
    normals = np.stack([edges[..., 1], -edges[..., 0]], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def sat_margin(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Signed SAT overlap (M, N) of convex polygons moving (M,V,2) vs fixed (N,W,2).

    > 0: penetration depth (minimum translation distance); <= 0: the polygons
    are separated by at least -value along some axis.
    """
    m, n = len(moving), len(fixed)
    axes = np.concatenate([
        np.broadcast_to(_unit_normals(moving)[:, None], (m, n) + moving.shape[1:]),
        np.broadcast_to(_unit_normals(fixed)[None], (m, n) + fixed.shape[1:]),
    ], axis=2)
    pa = np.einsum("mnkd,mvd->mnkv", axes, moving)
    pb = np.einsum("mnkd,nwd->mnkw", axes, fixed)
    overlap = np.minimum(pa.max(-1), pb.max(-1)) - np.maximum(pa.min(-1), pb.min(-1))
    return overlap.min(-1)


@dataclass(frozen=True)
class Box2D:
    """Rectangular scene entity o_i with geometry G(o_i)."""
    entity: Entity
    center: tuple[float, float]
    size: tuple[float, float]
    angle: float = 0.0
    active: bool = True  # False only under a diagnostic intervention

    @property
    def eid(self) -> str:
        return self.entity.entity_id

    def polygon(self) -> np.ndarray:
        return rect(self.center, self.size, self.angle)


@dataclass(frozen=True)
class Translation:
    """Rigid carry along a fixed polyline. parts = ((offset, size), ...) rigidly attached boxes."""
    waypoints: tuple[tuple[float, float], ...]
    parts: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


@dataclass(frozen=True)
class Hinge:
    """Panel of (length, thickness) rotating about pivot from theta0 to theta1 [rad]."""
    pivot: tuple[float, float]
    length: float
    thickness: float
    theta0: float
    theta1: float

    @property
    def reach(self) -> float:  # largest distance of any panel point from the hinge axis
        return math.hypot(self.length, self.thickness / 2.0)


@dataclass(frozen=True)
class Scene2D:
    """Immutable state s: action a, its motion, entities, and target shift."""
    action: ActionSpec
    motion: Translation | Hinge
    entities: tuple[Box2D, ...]
    offset: tuple[float, float] = (0.0, 0.0)  # SHIFT_TARGET displacement of target + motion


def envelope(scene: Scene2D, step: float = SAMPLE_STEP) -> np.ndarray:
    """Sampled envelope E(a, s) as (M, 4, 2) convex part poses, endpoints included."""
    m, off = scene.motion, np.asarray(scene.offset, dtype=float)
    if isinstance(m, Translation):
        way = np.asarray(m.waypoints, dtype=float) + off
        pts = np.concatenate([
            a + np.linspace(0.0, 1.0, max(1, math.ceil(np.linalg.norm(b - a) / step)) + 1)[:, None] * (b - a)
            for a, b in zip(way[:-1], way[1:])
        ])
        return np.concatenate([pts[:, None, :] + rect(o, s)[None] for o, s in m.parts])
    n = max(1, math.ceil(m.reach * abs(m.theta1 - m.theta0) / step))
    th = np.linspace(m.theta0, m.theta1, n + 1)
    local = rect((m.length / 2.0, 0.0), (m.length, m.thickness))
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    x = c * local[None, :, 0] - s * local[None, :, 1]
    y = s * local[None, :, 0] + c * local[None, :, 1]
    return np.stack([x, y], axis=-1) + np.asarray(m.pivot) + off


def entity_margins(scene: Scene2D, step: float = SAMPLE_STEP) -> np.ndarray:
    """Signed intrusion margin per entity (> 0 penetration, < 0 clearance); -inf if inactive."""
    polys = np.stack([b.polygon() for b in scene.entities])
    margins = sat_margin(envelope(scene, step), polys).max(axis=0)
    return np.where([b.active for b in scene.entities], margins, -np.inf)


def conflict(scene: Scene2D, step: float = SAMPLE_STEP) -> np.ndarray:
    """Conflict vector c(s, a) >= 0, one entry per entity."""
    return np.maximum(entity_margins(scene, step), 0.0)


def graded_conflict(c: np.ndarray, form: str = G_FORM) -> float:
    """G = g(c^S)."""
    return G_FORMS[form](np.asarray(c, dtype=float))


def feasibility(c: np.ndarray) -> int:
    """F: 0 if the action is feasible (no intrusion beyond CONTACT_TOL), else 1."""
    return int(np.any(np.asarray(c) > CONTACT_TOL))


def oracle_repair_cost(F: int, K: int, k_max: int) -> int:
    """J* = B F + K with B = K_max + 1, so every feasible S beats every infeasible S."""
    if F not in (0, 1) or not 0 <= K <= k_max:
        raise ValueError(f"need F in {{0,1}} and 0 <= K <= K_max, got F={F}, K={K}, K_max={k_max}")
    return (k_max + 1) * F + K


def apply_intervention(scene: Scene2D, iv: Intervention) -> Scene2D:
    """do(I_p): returns a new scene, never mutates the input."""
    if iv.kind is InterventionKind.SHIFT_TARGET:
        if iv.entity_id != scene.action.target_id:
            raise ValueError(f"SHIFT_TARGET must act on target {scene.action.target_id!r}")
        dx, dy = iv.params
        return replace(scene, offset=(scene.offset[0] + dx, scene.offset[1] + dy))
    ids = [b.eid for b in scene.entities]
    i = ids.index(iv.entity_id)
    box = scene.entities[i]
    if iv.kind is InterventionKind.RELOCATE:
        if not box.entity.movable:
            raise ValueError(f"structural entity {box.eid!r} cannot be relocated")
        new = replace(box, center=(float(iv.params[0]), float(iv.params[1])))
    else:  # diagnostic REMOVE / DISABLE_COLLISION: geometry ignored, vector length kept
        new = replace(box, active=False)
    return replace(scene, entities=scene.entities[:i] + (new,) + scene.entities[i + 1:])


def do(scene: Scene2D, S) -> Scene2D:
    return reduce(apply_intervention, S, scene)


def diagnostic_causes(scene: Scene2D, step: float = SAMPLE_STEP) -> tuple[str, ...]:
    """Minimal cause set via diagnostic removal: removing all causes restores
    feasibility, and every cause is necessary. Never an executable repair."""
    c = conflict(scene, step)
    causes = tuple(b.eid for b, ci in zip(scene.entities, c) if ci > CONTACT_TOL)
    def removed(ids):
        return conflict(do(scene, [Intervention(f"diag_{e}", InterventionKind.REMOVE, e) for e in ids]), step)
    assert feasibility(removed(causes)) == 0
    assert all(feasibility(removed([e for e in causes if e != k])) == 1 for k in causes)
    return causes


@dataclass(frozen=True)
class Case2D:
    name: str
    family: str
    scene: Scene2D
    candidates: tuple[Intervention, ...]
    expected: frozenset[frozenset[str]]  # manually derived minimal repair sets

    def __post_init__(self) -> None:
        if len(self.candidates) > MAX_CANDIDATES:
            raise ValueError(f"P={len(self.candidates)} exceeds {MAX_CANDIDATES}")
        if any(iv.is_diagnostic for iv in self.candidates):
            raise ValueError("diagnostic interventions are not repair candidates")
        moved = [iv.entity_id for iv in self.candidates if iv.kind is InterventionKind.RELOCATE]
        if len(moved) != len(set(moved)):
            raise ValueError("at most one relocation per entity keeps do(S) well defined")

    @property
    def k_max(self) -> int:
        return sum(iv.length for iv in self.candidates)


def subset(candidates: tuple[Intervention, ...], x: int) -> tuple[Intervention, ...]:
    """S(x) = {I_p : x_p = 1}, bit p of integer x is x_p."""
    return tuple(iv for p, iv in enumerate(candidates) if x >> p & 1)


def evaluate(case: Case2D, S: tuple[Intervention, ...], step: float = SAMPLE_STEP) -> RepairResult:
    c = conflict(do(case.scene, S), step)
    F, K = feasibility(c), sum(iv.length for iv in S)
    return RepairResult(S, tuple(c.tolist()), graded_conflict(c), F, K, oracle_repair_cost(F, K, case.k_max))


def enumerate_subsets(case: Case2D, step: float = SAMPLE_STEP) -> list[RepairResult]:
    """Exhaustive table indexed by bitmask x over all 2^P subsets."""
    return [evaluate(case, subset(case.candidates, x), step) for x in range(2 ** len(case.candidates))]


def minimal_repairs(results: list[RepairResult]) -> frozenset[frozenset[str]]:
    """All argmin_S J*(S) that are feasible (empty if no feasible repair exists)."""
    j_min = min(r.J for r in results)
    return frozenset(r.selected_ids for r in results if r.J == j_min and r.feasible)


OBJ_SIZE = (0.10, 0.06)  # carried object [m]
LANE = 0.04              # carry-lane half width, set by the trailing gripper
CARRY = Translation(
    waypoints=((0.0, 0.0), (0.8, 0.0)),
    parts=(((0.0, 0.0), OBJ_SIZE), ((-0.07, 0.0), (0.04, 2 * LANE))),  # object, gripper
)
LID = Hinge(pivot=(0.0, 0.0), length=0.30, thickness=0.02, theta0=0.0, theta1=math.pi / 2)
BOX, HBOX = (0.06, 0.06), (0.04, 0.04)  # movable box sizes for carry / lid scenes
STAGING_Y = 1.0                        # staging row, far from every envelope
STRUCTURAL, MOVABLE = EntityRole.STRUCTURAL, EntityRole.MOVABLE


def carry_scene(*entities: Box2D) -> Scene2D:
    return Scene2D(ActionSpec("carry", "obj"), CARRY, entities)


def lid_scene(*entities: Box2D) -> Scene2D:
    return Scene2D(ActionSpec("open", "lid"), LID, entities)


def side_box(eid: str, x: float, side: int, depth: float, size=BOX, role=MOVABLE) -> Box2D:
    """Box beside the carry lane; depth > 0 intrudes into the lane, < 0 is clearance."""
    return Box2D(Entity(eid, role), (x, side * (LANE + size[1] / 2 - depth)), size)


def radial_box(eid: str, deg: float, depth: float, size=HBOX, role=MOVABLE) -> Box2D:
    """Radially aligned box at angle deg; depth > 0 intrudes inside the panel reach."""
    a, r = math.radians(deg), LID.reach - depth + size[0] / 2
    return Box2D(Entity(eid, role), (r * math.cos(a), r * math.sin(a)), size, a)


def relocate(eid: str, slot: int) -> Intervention:
    return Intervention(f"r_{eid}", InterventionKind.RELOCATE, eid, params=(0.1 * slot, STAGING_Y))


def shift(target: str, dx: float, dy: float) -> Intervention:
    return Intervention(f"shift_{target}", InterventionKind.SHIFT_TARGET, target, params=(dx, dy))


def _relocations(scene: Scene2D) -> list[Intervention]:
    return [relocate(b.eid, k) for k, b in enumerate(scene.entities) if b.entity.movable]


def _case(name, family, scene, candidates, *expected) -> Case2D:
    return Case2D(name, family, scene, tuple(candidates), frozenset(frozenset(e) for e in expected))


def translation_cases() -> list[Case2D]:
    t1 = carry_scene(side_box("b1", 0.25, +1, -0.002), side_box("b2", 0.55, -1, -0.001),
                     Box2D(Entity("b3", MOVABLE), (0.85 + 0.002 + BOX[0] / 2, 0.0), BOX))
    t2 = carry_scene(side_box("b1", 0.40, +1, 0.015), side_box("d1", 0.20, -1, -0.010))
    t3 = carry_scene(side_box("b1", 0.25, +1, 0.010), side_box("b2", 0.55, -1, 0.020),
                     side_box("d1", 0.40, +1, -0.010))
    t4 = carry_scene(*[side_box(f"b{k + 1}", 0.10 + 0.15 * k, (-1) ** k, 0.005 * (k + 1)) for k in range(5)],
                     side_box("d1", 0.175, -1, -0.005), side_box("d2", 0.475, +1, -0.005))
    t5 = carry_scene(side_box("b1", 0.40, +1, 0.010), side_box("d1", 0.20, -1, -0.050))
    t6 = carry_scene(side_box("wall", 0.40, +1, 0.010, size=(0.20, 0.06), role=STRUCTURAL),
                     side_box("m", 0.40, -1, -0.005), side_box("d1", 0.15, +1, -0.010))
    return [
        _case("T1_feasible", "feasible_hard_negative", t1, _relocations(t1), ()),
        _case("T2_one", "one_blocker", t2, _relocations(t2), ("r_b1",)),
        _case("T3_two", "two_independent", t3, _relocations(t3), ("r_b1", "r_b2")),
        _case("T4_five", "five_independent", t4, _relocations(t4), tuple(f"r_b{k}" for k in range(1, 6))),
        _case("T5_redundant", "redundant_alternatives", t5,
              _relocations(t5) + [shift("obj", 0.0, -0.03)], ("r_b1",), ("shift_obj",)),
        _case("T6_coupled", "coupled", t6, [shift("obj", 0.0, -0.025)] + _relocations(t6), ("shift_obj", "r_m")),
    ]


def hinge_cases() -> list[Case2D]:
    far = radial_box("d1", 97, 0.010)  # inside the panel reach but beyond the swing end
    h1 = lid_scene(radial_box("b1", 45, -0.002), far)
    h2 = lid_scene(radial_box("b1", 45, 0.010), far)
    h3 = lid_scene(radial_box("b1", 30, 0.008), radial_box("b2", 65, 0.015), far)
    h4 = lid_scene(*[radial_box(f"b{k + 1}", 12 + 18 * k, 0.004 + 0.002 * k) for k in range(5)], far)
    h5 = lid_scene(radial_box("shelf", 45, 0.010, role=STRUCTURAL), radial_box("d1", 60, -0.030))
    back = -0.02 / math.sqrt(2)  # move the box 2 cm away from the overhang
    return [
        _case("H1_feasible", "feasible_hard_negative", h1, _relocations(h1), ()),
        _case("H2_one", "one_blocker", h2, _relocations(h2), ("r_b1",)),
        _case("H3_two", "two_independent", h3, _relocations(h3), ("r_b1", "r_b2")),
        _case("H4_five", "five_independent", h4, _relocations(h4), tuple(f"r_b{k}" for k in range(1, 6))),
        _case("H5_immovable_cause", "immovable_cause", h5,
              [shift("lid", back, back)] + _relocations(h5), ("shift_lid",)),
    ]


def all_cases() -> list[Case2D]:
    return translation_cases() + hinge_cases()


BOUNDARY_KINDS = ("carry_side", "carry_end", "lid_reach", "lid_end")


def boundary_scene(kind: str, depth: float) -> Scene2D:
    """One movable box placed depth inside (> 0) or outside (< 0) an envelope boundary."""
    if kind == "carry_side":
        return carry_scene(side_box("b", 0.40, +1, depth))
    if kind == "carry_end":
        front = CARRY.waypoints[-1][0] + OBJ_SIZE[0] / 2
        return carry_scene(Box2D(Entity("b", MOVABLE), (front - depth + BOX[0] / 2, 0.0), BOX))
    if kind == "lid_reach":
        return lid_scene(radial_box("b", 45, depth))
    if kind == "lid_end":
        edge = -LID.thickness / 2  # panel face at the end of the swing (theta1 = 90 deg)
        return lid_scene(Box2D(Entity("b", MOVABLE), (edge + depth - HBOX[0] / 2, 0.15), HBOX))
    raise ValueError(f"unknown boundary kind {kind!r}")
