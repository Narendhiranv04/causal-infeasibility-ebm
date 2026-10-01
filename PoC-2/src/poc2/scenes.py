"""PoC-2 Stage 1: deterministic make-space shelf-insertion mechanism cases (plan2.md section 17).

A gripped object (PoC-1 GRIPPED composite) is inserted straight into a cupboard
shelf. Movable boxes stand on the shelf beside the insertion lane; placement
regions are free shelf areas a relocated box can be moved to. All geometry is
built with the validated PoC-1 insertion builders (`poc.cases`).

Geometry reminder (metres): the lane's outer faces are y = +/-LANE = +/-0.05; a
side box with depth > 0 intrudes into the lane by that depth, depth < 0 is
clearance. Repositioning the target shifts the whole insertion (start and goal)
sideways before retrying the same skill.
"""

from dataclasses import dataclass

from poc import cases as cs
from poc.types import Intervention, InterventionKind
from poc2.types import PlacementRegion, RepairOption

REGION_HALF = (0.035, 0.035, 0.05)  # [m] footprint of one placement region (one box fits)
NARROW = (0.02, 0.03, 0.05)         # half-size of a narrow box (fits between other boxes)


@dataclass(frozen=True)
class MakeSpaceCase:
    name: str
    mechanism: str
    scene: cs.Scene3D
    regions: tuple[PlacementRegion, ...]
    options: tuple[RepairOption, ...]
    expected_causes: frozenset[frozenset[str]]   # manual C*
    expected_repairs: frozenset[frozenset[str]]  # manual S*

    def __post_init__(self) -> None:
        movable = [e for e in self.scene.entities if e.entity.movable]
        if not 3 <= len(movable) <= 6 or not 2 <= len(self.regions) <= 4 or len(self.options) > 10:
            raise ValueError(f"{self.name}: need 3-6 movable objects, 2-4 regions and P <= 10")
        ids = {e.eid: e for e in self.scene.entities}
        regions = {r.region_id for r in self.regions}
        for o in self.options:
            iv = o.intervention
            if iv.kind is InterventionKind.RELOCATE and (o.region_id not in regions or not ids[iv.entity_id].entity.movable):
                raise ValueError(f"{self.name}: {o.option_id} must move a movable object into a declared region")


def region(rid: str, x: float, y: float) -> PlacementRegion:
    return PlacementRegion(rid, (x, y, REGION_HALF[2]), REGION_HALF)


def to_region(ent, reg: PlacementRegion, cost: int = 1) -> RepairOption:
    """Relocate a shelf box so it stands centred in a placement region."""
    z = ent.boxes[0].center[2]
    iv = Intervention(f"{ent.eid}->{reg.region_id}", InterventionKind.RELOCATE, ent.eid, length=cost,
                      params=(reg.center[0], reg.center[1], z))
    return RepairOption(iv, reg.region_id)


def reposition(dy: float, name: str = "shift", cost: int = 1) -> RepairOption:
    """Reposition the target (and its insertion lane) sideways by dy, then retry the same skill."""
    return RepairOption(Intervention(name, InterventionKind.SHIFT_TARGET, "obj", length=cost, params=(0.0, dy, 0.0)))


def _box(eid, slot, side, depth, half=cs.BLOCK):
    return cs.side_box(eid, cs.SLOTS[slot], side, depth, half=half)


def _case(name, mechanism, entities, regions, options, causes, *repairs) -> MakeSpaceCase:
    return MakeSpaceCase(name, mechanism, cs.insertion(*entities), tuple(regions), tuple(options),
                         frozenset({frozenset(causes)}) if causes else frozenset(),
                         frozenset(frozenset(r) for r in repairs))


def independent() -> MakeSpaceCase:
    """Two independent blockers, each with two destinations (b1's far region costs 2)."""
    b1, b2, d1 = _box("b1", 1, +1, 0.010), _box("b2", 3, -1, 0.015), _box("d1", 4, +1, -0.006)
    r1, r2, r3, r4 = region("r1", 0.06, 0.22), region("r2", 0.20, 0.22), region("r3", 0.20, -0.22), region("r4", 0.33, -0.22)
    opts = [to_region(b1, r1), to_region(b1, r2, cost=2), to_region(b2, r3), to_region(b2, r4), to_region(d1, r2)]
    return _case("M1_independent", "independent", (b1, b2, d1), (r1, r2, r3, r4), opts, ("b1", "b2"),
                 ("b1->r1", "b2->r3"), ("b1->r1", "b2->r4"))


def placement_competition() -> MakeSpaceCase:
    """b1 and b2 can each use rA alone, but not together; b1's alternative rB costs 2."""
    b1, b2, d1 = _box("b1", 1, +1, 0.010), _box("b2", 2, -1, 0.012), _box("d1", 4, +1, -0.006)
    rA, rB = region("rA", 0.25, -0.22), region("rB", 0.06, -0.22)
    opts = [to_region(b1, rA), to_region(b1, rB, cost=2), to_region(b2, rA), to_region(d1, rB)]
    return _case("M2_placement_competition", "placement_competition", (b1, b2, d1), (rA, rB), opts, ("b1", "b2"),
                 ("b1->rB", "b2->rA"))


def substitutable() -> MakeSpaceCase:
    """One blocker removable either by relocating it or by repositioning the target."""
    b1, d1, d2 = _box("b1", 2, +1, 0.008), _box("d1", 3, -1, -0.030), _box("d2", 4, +1, -0.010)
    r1, r2 = region("r1", 0.06, 0.22), region("r2", 0.33, -0.22)
    opts = [to_region(b1, r1), reposition(-0.02), to_region(d1, r2), to_region(d2, r1)]
    return _case("M3_substitutable", "substitutable", (b1, d1, d2), (r1, r2), opts, ("b1",),
                 ("b1->r1",), ("shift",))


def envelope_coupling() -> MakeSpaceCase:
    """The jamb forces a sideways repositioning, which pushes the lane into nb (clear before)."""
    nb, d1, d2 = _box("nb", 1, -1, -0.010, NARROW), _box("d1", 3, +1, -0.004), _box("d2", 4, -1, -0.045)
    r1, r2, r3 = region("r1", 0.06, -0.22), region("r2", 0.20, -0.22), region("r3", 0.20, 0.22)
    opts = [reposition(-0.03), to_region(nb, r1), to_region(nb, r2, cost=2), to_region(d1, r3), to_region(d2, r2)]
    return _case("M4_envelope_coupling", "envelope_coupling", (cs.jamb(0.010), nb, d1, d2), (r1, r2, r3), opts,
                 ("jamb",), ("shift", "nb->r1"))


def distractors() -> MakeSpaceCase:
    """One blocker among distractor candidates; region r2 is already occupied by d1."""
    b1, d2, d3 = _box("b1", 2, +1, 0.012), _box("d2", 1, -1, -0.010), _box("d3", 4, +1, -0.008)
    r1, r2, r3 = region("r1", 0.06, 0.22), region("r2", 0.20, -0.22), region("r3", 0.33, -0.22)
    d1 = cs._ent("d1", cs.MOVABLE, (0.20, -0.22, cs.BLOCK[2]), cs.BLOCK)  # standing inside r2
    opts = [to_region(b1, r1), to_region(b1, r2), to_region(d1, r3), to_region(d2, r3), to_region(d3, r1)]
    return _case("M5_distractors", "distractors", (b1, d1, d2, d3), (r1, r2, r3), opts, ("b1",), ("b1->r1",))


def structural_cause() -> MakeSpaceCase:
    """The structural jamb is the cause; the only executable repair repositions the target."""
    d1, d2, d3 = _box("d1", 3, +1, -0.004), _box("d2", 2, -1, -0.045), _box("d3", 0, +1, -0.010)
    r1, r2 = region("r1", 0.06, 0.22), region("r2", 0.33, -0.22)
    opts = [reposition(-0.03), to_region(d1, r1), to_region(d2, r2), to_region(d3, r2)]
    return _case("M6_structural_cause", "structural_cause", (cs.jamb(0.010), d1, d2, d3), (r1, r2), opts,
                 ("jamb",), ("shift",))


def stage1_cases() -> list[MakeSpaceCase]:
    return [independent(), placement_competition(), substitutable(), envelope_coupling(), distractors(),
            structural_cause()]
