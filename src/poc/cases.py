"""Stage 4: 3D scene families (cupboard insertion, cupboard extraction, hinged
lid opening), executable interventions, and the canonical cases.

Frames: metres, z up. The cupboard opening is the plane x = 0; its interior is
x in [0, 0.40], y in [-0.30, 0.30], z in [0, 0.25]. Near-boundary pairs sit
DELTA on either side of an analytically known envelope boundary.
"""

import math
from dataclasses import dataclass, replace
from functools import cached_property, reduce
from itertools import combinations

import numpy as np

from poc.envelope import ENVELOPE_STEP, HingeMotion, LinearMotion, Sweep, shifted, sweep
from poc.mj_scene import DISTMAX, IDENTITY, Box3D, Composite, Entity3D, GeomWorld
from poc.oracle import Assessment, assess, is_valid
from poc.types import ActionSpec, Entity, EntityRole, Intervention, InterventionKind

STRUCTURAL, MOVABLE = EntityRole.STRUCTURAL, EntityRole.MOVABLE
DELTA = 1e-3     # [m] near-boundary perturbation on each side of the boundary
HARD_GAP = 3e-3  # [m] clearance of the closest entity in feasible hard negatives


@dataclass(frozen=True)
class Scene3D:
    """Immutable state s: action a, moving composite, prescribed motion, entities."""
    family: str
    action: ActionSpec
    moving: Composite
    motion: LinearMotion | HingeMotion
    entities: tuple[Entity3D, ...]
    fixture: tuple[Box3D, ...] = ()  # static target geometry that moves with SHIFT_TARGET (not a conflict entity)


def apply_intervention(scene: Scene3D, iv: Intervention) -> Scene3D:
    """do(I_p) for executable interventions; returns a new scene."""
    if iv.is_diagnostic:
        raise ValueError("diagnostic interventions are oracle queries (feasibility_excluding), not scene edits")
    if iv.kind is InterventionKind.SHIFT_TARGET:
        if iv.entity_id != scene.action.target_id:
            raise ValueError(f"SHIFT_TARGET must act on target {scene.action.target_id!r}")
        # pre-action repositioning macro: the target (and its fixture) moves, then the same skill is retried
        fixture = tuple(replace(b, center=tuple(c + o for c, o in zip(b.center, iv.params))) for b in scene.fixture)
        return replace(scene, motion=shifted(scene.motion, iv.params), fixture=fixture)
    ids = [e.eid for e in scene.entities]
    i = ids.index(iv.entity_id)
    ent = scene.entities[i]
    if not ent.entity.movable:
        raise ValueError(f"structural entity {ent.eid!r} cannot be relocated")
    move = [p - c for p, c in zip(iv.params, ent.boxes[0].center)]  # params: new reference centre
    boxes = tuple(replace(b, center=tuple(c + m for c, m in zip(b.center, move))) for b in ent.boxes)
    return replace(scene, entities=scene.entities[:i] + (replace(ent, boxes=boxes),) + scene.entities[i + 1:])


def do(scene: Scene3D, S) -> Scene3D:
    return reduce(apply_intervention, S, scene)


def evaluate(scene: Scene3D, step: float = ENVELOPE_STEP) -> tuple[Assessment, Sweep]:
    """Fresh MuJoCo model per call: identical scenes give identical results."""
    sw = sweep(GeomWorld(scene.entities, scene.moving), scene.motion, scene.moving, step)
    return assess([e.eid for e in scene.entities], sw.d_min, sw.d_start, sw.d_goal), sw


@dataclass(frozen=True)
class Case3D:
    name: str
    kind: str                              # feasible | single_blocker | near_boundary_out/in | immovable_cause
    scene: Scene3D
    cause: tuple[str, ...] = ()            # expected diagnostic cause set
    repair: tuple[Intervention, ...] = ()  # executable repair expected to restore feasibility
    boundary: str = ""                     # analytic boundary of a near-boundary pair

    def __post_init__(self) -> None:
        if any(iv.is_diagnostic for iv in self.repair):
            raise ValueError("a diagnostic intervention is never an executable repair")


def _ent(eid: str, role: EntityRole, center, half) -> Entity3D:
    return Entity3D(Entity(eid, role), (Box3D(tuple(center), tuple(half)),))


def relocate(eid: str, center) -> Intervention:
    return Intervention(f"r_{eid}", InterventionKind.RELOCATE, eid, params=tuple(center))


def shift(target: str, offset) -> Intervention:
    return Intervention(f"shift_{target}", InterventionKind.SHIFT_TARGET, target, params=tuple(offset))


# ------------------------------------------------------- cupboard families
LANE = 0.05      # composite half-width in y: outer faces of the fingers / palm
OBJ_TOP = 0.11   # top face of the carried object (bottom 1 cm above the shelf)
CARRY_Z = 0.06   # height of the moving-frame origin (object centre)
DEEP, OUTSIDE = (0.30, 0.0, CARRY_Z), (-0.35, 0.0, CARRY_Z)
STAGING = (0.10, -0.20, 0.06)  # free shelf area for a relocated neighbour
GRIPPED = Composite(parts=(    # carried object + two fingers + palm + wrist, object-centred frame
    ("object", Box3D((0.0, 0.0, 0.0), (0.04, 0.03, 0.05))),
    ("finger_l", Box3D((0.0, 0.04, 0.0), (0.03, 0.01, 0.02))),
    ("finger_r", Box3D((0.0, -0.04, 0.0), (0.03, 0.01, 0.02))),
    ("palm", Box3D((-0.06, 0.0, 0.0), (0.02, 0.05, 0.02))),
    ("wrist", Box3D((-0.18, 0.0, 0.0), (0.10, 0.02, 0.02))),
))


def cupboard() -> tuple[Entity3D, ...]:
    return (_ent("shelf", STRUCTURAL, (0.20, 0.0, -0.01), (0.20, 0.30, 0.01)),
            _ent("top", STRUCTURAL, (0.20, 0.0, 0.26), (0.20, 0.30, 0.01)),
            _ent("wall_l", STRUCTURAL, (0.20, 0.31, 0.12), (0.20, 0.01, 0.14)),
            _ent("wall_r", STRUCTURAL, (0.20, -0.31, 0.12), (0.20, 0.01, 0.14)),
            _ent("back", STRUCTURAL, (0.41, 0.0, 0.12), (0.01, 0.32, 0.14)))


def neighbour(x: float, depth: float) -> Entity3D:
    """Movable box on the shelf; its inner face is at y = LANE - depth (depth > 0 intrudes)."""
    return _ent("box", MOVABLE, (x, LANE + 0.04 - depth, 0.06), (0.04, 0.04, 0.06))


def jamb(depth: float) -> Entity3D:
    """Structural side stile of the opening frame; inner face at y = LANE - depth."""
    y_in = LANE - depth
    return _ent("jamb", STRUCTURAL, (-0.01, (y_in + 0.30) / 2, 0.12), (0.01, (0.30 - y_in) / 2, 0.14))


def header(depth: float) -> Entity3D:
    """Structural top rail of the opening frame; underside at z = OBJ_TOP - depth."""
    z_lo = OBJ_TOP - depth
    return _ent("header", STRUCTURAL, (-0.01, 0.0, (z_lo + 0.26) / 2), (0.01, 0.30, (0.26 - z_lo) / 2))


def insertion(*extra: Entity3D) -> Scene3D:
    return Scene3D("insertion", ActionSpec("insert", "obj"), GRIPPED, LinearMotion(OUTSIDE, DEEP), cupboard() + extra)


def extraction(*extra: Entity3D) -> Scene3D:
    return Scene3D("extraction", ActionSpec("extract", "obj"), GRIPPED, LinearMotion(DEEP, OUTSIDE), cupboard() + extra)


# ------------------------------------------------------------ hinged lid
BOX_H, LID_L, LID_W, LID_T = 0.15, 0.20, 0.24, 0.01  # box height; lid length, width, thickness
LID_REACH = math.hypot(LID_L, LID_T)                 # farthest lid point from the hinge axis
OPEN = math.radians(110.0)
LID = Composite(parts=(("lid", Box3D((LID_L / 2, 0.0, LID_T / 2), (LID_L / 2, LID_W / 2, LID_T / 2))),))


def beam(depth: float) -> Entity3D:
    """Structural bar above the hinge; underside at z = BOX_H + LID_REACH - depth."""
    z_u = BOX_H + LID_REACH - depth
    return _ent("beam", STRUCTURAL, (0.0, 0.0, z_u + 0.01), (0.05, 0.20, 0.01))


def front_shelf() -> Entity3D:
    """Structural shelf over the front half of the box, underside at z = 0.30."""
    return _ent("shelf", STRUCTURAL, (0.20, 0.0, 0.31), (0.15, 0.20, 0.01))


def bottle(top: float) -> Entity3D:
    """Movable object standing behind the hinge, height `top`."""
    return _ent("bottle", MOVABLE, (-0.09, 0.0, top / 2), (0.03, 0.03, top / 2))


def lid_scene(*entities: Entity3D) -> Scene3D:
    # rotating about -y takes the lid's +x (towards the box front) to +z: the lid opens upwards
    return Scene3D("hinge", ActionSpec("open_lid", "lid"), LID,
                   HingeMotion((0.0, 0.0, BOX_H), (0.0, -1.0, 0.0), 0.0, OPEN), entities)


def canonical_cases() -> list[Case3D]:
    side = "neighbour inner face vs finger outer face y = LANE"
    top = "header underside vs object top face z = OBJ_TOP"
    reach = "beam underside vs lid reach z = BOX_H + hypot(LID_L, LID_T)"
    return [
        Case3D("A1_feasible", "feasible", insertion(neighbour(0.15, -HARD_GAP), header(-HARD_GAP))),
        Case3D("A2_single_blocker", "single_blocker", insertion(neighbour(0.15, 0.015)),
               ("box",), (relocate("box", STAGING),)),
        Case3D("A3_boundary_out", "near_boundary_out", insertion(neighbour(0.15, -DELTA)), boundary=side),
        Case3D("A3_boundary_in", "near_boundary_in", insertion(neighbour(0.15, DELTA)), ("box",), boundary=side),
        Case3D("A4_immovable_cause", "immovable_cause", insertion(jamb(0.010)),
               ("jamb",), (shift("obj", (0.0, -0.03, 0.0)),)),
        Case3D("B1_feasible", "feasible", extraction(jamb(-HARD_GAP), neighbour(0.06, -HARD_GAP))),
        Case3D("B2_single_blocker", "single_blocker", extraction(neighbour(0.06, 0.012)),
               ("box",), (relocate("box", STAGING),)),
        Case3D("B3_boundary_out", "near_boundary_out", extraction(header(-DELTA)), boundary=top),
        Case3D("B3_boundary_in", "near_boundary_in", extraction(header(DELTA)), ("header",), boundary=top),
        Case3D("B4_immovable_cause", "immovable_cause", extraction(jamb(0.010)),
               ("jamb",), (shift("obj", (0.0, -0.03, 0.0)),)),
        Case3D("C1_feasible", "feasible", lid_scene(beam(-HARD_GAP), bottle(0.25))),
        Case3D("C2_single_blocker", "single_blocker", lid_scene(bottle(0.33)),
               ("bottle",), (relocate("bottle", (-0.30, 0.30, 0.165)),)),
        Case3D("C3_boundary_out", "near_boundary_out", lid_scene(beam(-DELTA)), boundary=reach),
        Case3D("C3_boundary_in", "near_boundary_in", lid_scene(beam(DELTA)), ("beam",), boundary=reach),
        Case3D("C4_immovable_cause", "immovable_cause", lid_scene(front_shelf()),
               ("shelf",), (shift("lid", (-0.12, 0.0, 0.0)),)),
    ]


# ================================================================ Stage 5
# Interpretation boundaries. The oracle certifies that the PRESCRIBED next-action
# envelope is (in)feasible; it does not prove that no other motion exists.
# RELOCATE and SHIFT_TARGET are high-level corrective macros: Stage 5 validates
# their resulting static geometry (V), their combinatorics and whether they
# restore target-action feasibility, not the low-level motion that executes them.

FIXTURE = "target_fixture"
PARK = (50.0, 50.0, 50.0)  # moving composite parked far away during static validity queries


def _pair_distance(world: GeomWorld, ga, gb) -> float:
    return min(world.signed_distance(a, b) for a in ga for b in gb)


def validity(scene: Scene3D) -> tuple[int, float]:
    """(V, margin): movable entities and the target fixture must not penetrate any other static body."""
    bodies = scene.entities + ((Entity3D(Entity(FIXTURE, EntityRole.TARGET), scene.fixture),) if scene.fixture else ())
    world = GeomWorld(bodies, scene.moving)
    world.set_pose(PARK, IDENTITY)
    d = [_pair_distance(world, world.entity_geoms[i], world.entity_geoms[j]) for i, j in combinations(range(len(bodies)), 2)
         if bodies[i].entity.movable or bodies[j].entity.movable]
    return is_valid(d), min(d, default=DISTMAX)


@dataclass(frozen=True)
class Stage5Case:
    name: str
    structure: str                        # independent | coupled | substitutable | mixed | validity | perturbed
    scene: Scene3D
    candidates: tuple[Intervention, ...]  # executable candidate set I (P <= 10)
    expected: frozenset[frozenset[str]]   # manually derived admissible minimal repairs

    def __post_init__(self) -> None:
        moved = [iv.entity_id for iv in self.candidates if iv.kind is InterventionKind.RELOCATE]
        shifts = [iv for iv in self.candidates if iv.kind is InterventionKind.SHIFT_TARGET]
        if len(self.candidates) > 10 or any(iv.is_diagnostic for iv in self.candidates):
            raise ValueError("need P <= 10 executable candidates")
        if len(moved) != len(set(moved)) or len(shifts) > 1:
            raise ValueError("at most one relocation per entity and one repositioning macro")

    @property
    def k_max(self) -> int:
        return sum(iv.length for iv in self.candidates)


@dataclass(frozen=True)
class Table:
    """Exhaustive oracle table over all 2^P subsets, rows indexed by bitmask x."""
    ids: tuple[str, ...]
    d: np.ndarray        # (2^P, N) signed envelope distance d^S
    d_start: np.ndarray
    d_goal: np.ndarray
    V: np.ndarray        # (2^P,) post-intervention static validity
    margin: np.ndarray   # (2^P,) min static pair distance behind V
    K: np.ndarray        # (2^P,) repair length

    def assessment(self, x: int) -> Assessment:
        return assess(self.ids, self.d[x], self.d_start[x], self.d_goal[x])

    @cached_property
    def rows(self) -> list[Assessment]:
        return [self.assessment(x) for x in range(len(self.K))]

    @property
    def G(self) -> np.ndarray:
        return np.array([a.G for a in self.rows])

    @property
    def F(self) -> np.ndarray:
        return np.array([a.F for a in self.rows])

    @property
    def C(self) -> np.ndarray:
        return np.array([a.c for a in self.rows])


def exhaustive_table(case: Stage5Case, step: float = ENVELOPE_STEP) -> Table:
    """All 2^P subsets. d_i depends only on entity i's pose and the envelope, so every
    pose variant is swept once and rows are assembled exactly (tested against evaluate(do(S)))."""
    scene, cands = case.scene, case.candidates
    reloc = {iv.entity_id: iv for iv in cands if iv.kind is InterventionKind.RELOCATE}
    shift_iv = next((iv for iv in cands if iv.kind is InterventionKind.SHIFT_TARGET), None)
    scenes = [scene] + ([apply_intervention(scene, shift_iv)] if shift_iv else [])
    pose = [[e] + ([apply_intervention(scene, reloc[e.eid]).entities[i]] if e.eid in reloc else [])
            for i, e in enumerate(scene.entities)]
    flat = [Entity3D(Entity(f"{e.eid}.v{k}", e.entity.role), e.boxes) for vs in pose for k, e in enumerate(vs)]
    col = {(i, k): n for n, (i, k) in enumerate((i, k) for i, vs in enumerate(pose) for k in range(len(vs)))}
    sweeps = [sweep(GeomWorld(tuple(flat), scene.moving), sc.motion, scene.moving, step) for sc in scenes]
    fixtures = [Entity3D(Entity(f"{FIXTURE}.v{m}", EntityRole.TARGET), sc.fixture) for m, sc in enumerate(scenes)
                if sc.fixture]
    static = tuple(flat) + tuple(fixtures)
    world = GeomWorld(static, scene.moving)
    world.set_pose(PARK, IDENTITY)
    pair = {}
    rows = {k: [] for k in ("d", "d_start", "d_goal", "V", "margin", "K")}
    for x in range(2 ** len(cands)):
        S = [iv for p, iv in enumerate(cands) if x >> p & 1]
        m = int(shift_iv in S)
        cols = [col[(i, int(e.eid in reloc and reloc[e.eid] in S))] for i, e in enumerate(scene.entities)]
        for key, arr in (("d", sweeps[m].d_min), ("d_start", sweeps[m].d_start), ("d_goal", sweeps[m].d_goal)):
            rows[key].append(arr[cols])
        bodies = cols + ([len(flat) + m] if fixtures else [])
        for a, b in combinations(bodies, 2):
            if (a, b) not in pair and (static[a].entity.movable or static[b].entity.movable):
                pair[(a, b)] = _pair_distance(world, world.entity_geoms[a], world.entity_geoms[b])
        dist = [pair[(a, b)] for a, b in combinations(bodies, 2) if (a, b) in pair]
        rows["V"].append(is_valid(dist))
        rows["margin"].append(min(dist, default=DISTMAX))
        rows["K"].append(sum(iv.length for iv in S))
    return Table(tuple(e.eid for e in scene.entities), **{k: np.array(v) for k, v in rows.items()})


# ------------------------------------------------------- Stage 5 catalogue
BLOCK = (0.03, 0.03, 0.05)        # half-size of a shelf box
SLOTS = (0.04, 0.11, 0.18, 0.25, 0.32)  # x positions of shelf boxes along a lane
BASE = Box3D((LID_L / 2, 0.0, BOX_H / 2), (LID_L / 2, LID_W / 2, BOX_H / 2))  # hinged box body


def side_box(eid: str, x: float, side: int, depth: float, lane_y: float = 0.0, half=BLOCK) -> Entity3D:
    """Movable shelf box beside a lane centred at lane_y; depth > 0 intrudes past the finger faces."""
    return _ent(eid, MOVABLE, (x, lane_y + side * (LANE + half[1] - depth), half[2]), half)


def book(eid: str, x: float) -> Entity3D:
    """Flat movable item lying across the lane: blocks the object from below, whatever its lateral offset."""
    return _ent(eid, MOVABLE, (x, -0.02, 0.01), (0.025, 0.07, 0.01))


def item(eid: str, x: float, y: float, top: float, half_xy=(0.03, 0.02)) -> Entity3D:
    """Movable object standing on the table near the hinged box."""
    return _ent(eid, MOVABLE, (x, y, top / 2), (half_xy[0], half_xy[1], top / 2))


def staged(ent: Entity3D, y: float, x: float | None = None) -> Intervention:
    c = ent.boxes[0].center
    return relocate(ent.eid, (c[0] if x is None else x, y, c[2]))


def _s5(name, structure, scene, candidates, *expected) -> Stage5Case:
    return Stage5Case(name, structure, scene, tuple(candidates), frozenset(frozenset(e) for e in expected))


def _lane_boxes(k: int, lane_y: float, xs, depths=None) -> list[Entity3D]:
    spots = [(x, side) for x in xs for side in (+1, -1)]
    return [side_box(f"b{i + 1}", *spots[i], (depths or {}).get(i, 0.004 + 0.003 * i), lane_y) for i in range(k)]


def insertion_blockers(k: int, depths=None, name=None, structure="independent") -> Stage5Case:
    blockers = _lane_boxes(k, 0.0, SLOTS, depths)
    used = {(b.boxes[0].center[0], b.boxes[0].center[1] > 0) for b in blockers}
    free = [(x, s) for x in reversed(SLOTS) for s in (+1, -1) if (x, s > 0) not in used][:2]
    distract = [side_box(f"d{j + 1}", x, s, -0.004) for j, (x, s) in enumerate(free)]
    ents = blockers + distract
    cands = [staged(e, 0.25 if e.boxes[0].center[1] > 0 else -0.25) for e in ents]
    return _s5(name or f"S5_ins_k{k}", structure, insertion(*ents), cands, tuple(f"r_b{i + 1}" for i in range(k)
               if (depths or {}).get(i, 1.0) > 0))


def extraction_blockers(k: int) -> Stage5Case:
    lane, xs = -0.12, SLOTS[:3]  # boxes stay clear of the palm and wrist at the start grasp
    blockers = _lane_boxes(k, lane, xs)
    ents = blockers + [side_box("d1", xs[2], -1, -0.004, lane)]  # k <= 5 leaves this spot free
    cands = [staged(e, 0.15 if e.boxes[0].center[1] > lane else 0.25) for e in ents]
    scene = Scene3D("extraction", ActionSpec("extract", "obj"), GRIPPED,
                    LinearMotion((0.30, lane, CARRY_Z), (-0.35, lane, CARRY_Z)), cupboard() + tuple(ents),
                    (Box3D((0.30, lane, CARRY_Z), (0.04, 0.03, 0.05)),))
    return _s5(f"S5_ext_k{k}", "independent", scene, cands, tuple(f"r_b{i + 1}" for i in range(k)))


def hinge_blockers(k: int) -> Stage5Case:
    ys = (-0.10, -0.05, 0.0, 0.05, 0.10)[:k] if k == 5 else (-0.10, 0.0, 0.10)[:k]
    ents = [item(f"b{i + 1}", -0.09, y, 0.325 + 0.005 * i) for i, y in enumerate(ys)] + [item("d1", -0.22, 0.0, 0.30)]
    cands = [relocate(e.eid, (-0.45, -0.20 + 0.08 * j, e.boxes[0].center[2])) for j, e in enumerate(ents)]
    return _s5(f"S5_hinge_k{k}", "independent", replace(lid_scene(*ents), fixture=(BASE,)), cands,
               tuple(f"r_b{i + 1}" for i in range(k)))


def insertion_coupled(mixed: bool) -> Stage5Case:
    """3D analogue of T6: the jamb forces a lateral repositioning that pushes the lane into box nb.
    mixed=True adds three flat books lying across the lane (independent, repositioning-invariant)."""
    nb, dis = side_box("nb", SLOTS[1], -1, -0.010, half=(0.02, 0.03, 0.05)), side_box("d1", SLOTS[3], +1, -0.004)
    books = [book(f"bk{i + 1}", x) for i, x in enumerate((0.06, 0.16, 0.26))] if mixed else []
    cands = [shift("obj", (0.0, -0.03, 0.0)), staged(nb, -0.25), staged(dis, 0.25, 0.33)] + [staged(b, 0.22) for b in books]
    return _s5("S5_ins_mixed" if mixed else "S5_ins_coupled", "mixed" if mixed else "coupled",
               insertion(jamb(0.010), nb, dis, *books), cands, ("shift_obj", "r_nb") + tuple(f"r_{b.eid}" for b in books))


def hinge_repositioning(name: str, structure: str, bt_x: float, bt_top: float) -> Stage5Case:
    """The front shelf forces moving the box back 12 cm; bottle bt then blocks the lid sweep
    (tall, further back: coupled in G) or the moved box body itself (low, close: coupled in V)."""
    scene = replace(lid_scene(front_shelf(), item("bt", bt_x, 0.0, bt_top), item("d1", -0.30, 0.15, 0.30)),
                    fixture=(BASE,))
    cands = [shift("lid", (-0.12, 0.0, 0.0)), staged(scene.entities[1], 0.35, -0.45), staged(scene.entities[2], -0.35, -0.45)]
    return _s5(name, structure, scene, cands, ("shift_lid", "r_bt"))


def substitutable_cases() -> list[Stage5Case]:
    """One conflict, two equal-cost executable repairs (relocate the blocker, or reposition the target)."""
    ins = insertion(side_box("b1", SLOTS[2], +1, 0.008), side_box("d1", SLOTS[3], -1, -0.030))
    lid = replace(lid_scene(item("bt", -0.09, 0.0, 0.33), item("d1", -0.30, 0.15, 0.30)), fixture=(BASE,))
    return [
        _s5("S5_ins_substitutable", "substitutable", ins, [shift("obj", (0.0, -0.02, 0.0)), staged(ins.entities[-2], 0.25),
            staged(ins.entities[-1], -0.25)], ("r_b1",), ("shift_obj",)),
        _s5("S5_hinge_substitutable", "substitutable", lid, [shift("lid", (0.06, 0.0, 0.0)),
            staged(lid.entities[0], 0.35, -0.45), staged(lid.entities[1], -0.35, -0.45)], ("r_bt",), ("shift_lid",)),
    ]


def stage5_cases() -> list[Stage5Case]:
    cases = [insertion_blockers(k) for k in range(1, 7)] + [extraction_blockers(k) for k in (2, 4, 5)]
    cases += [hinge_blockers(k) for k in (2, 3, 5)] + substitutable_cases()
    cases += [insertion_coupled(mixed=False), insertion_coupled(mixed=True),
              hinge_repositioning("S5_hinge_coupled", "coupled", -0.18, 0.30),
              hinge_repositioning("S5_hinge_validity", "validity", -0.10, 0.10)]
    for tag, dep in (("p_minus", 0.0065), ("p_plus", 0.0075), ("cross_in", 0.001), ("cross_out", -0.001)):
        cases.append(insertion_blockers(3, {1: dep}, f"S5_ins_k3_{tag}", "perturbed"))
    return cases
