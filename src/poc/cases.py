"""Stage 4: 3D scene families (cupboard insertion, cupboard extraction, hinged
lid opening), executable interventions, and the canonical cases.

Frames: metres, z up. The cupboard opening is the plane x = 0; its interior is
x in [0, 0.40], y in [-0.30, 0.30], z in [0, 0.25]. Near-boundary pairs sit
DELTA on either side of an analytically known envelope boundary.
"""

import math
from dataclasses import dataclass, replace
from functools import reduce

from poc.envelope import ENVELOPE_STEP, HingeMotion, LinearMotion, Sweep, shifted, sweep
from poc.mj_scene import Box3D, Composite, Entity3D, GeomWorld
from poc.oracle import Assessment, assess
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


def apply_intervention(scene: Scene3D, iv: Intervention) -> Scene3D:
    """do(I_p) for executable interventions; returns a new scene."""
    if iv.is_diagnostic:
        raise ValueError("diagnostic interventions are oracle queries (feasibility_excluding), not scene edits")
    if iv.kind is InterventionKind.SHIFT_TARGET:
        if iv.entity_id != scene.action.target_id:
            raise ValueError(f"SHIFT_TARGET must act on target {scene.action.target_id!r}")
        return replace(scene, motion=shifted(scene.motion, iv.params))
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
