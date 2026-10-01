"""PoC-2 records (plan2.md sections 8 and 16). Records and invariants only.

No geometry, exhaustive search, Möbius analysis, QUBO or dataset generation
lives here. Entity / intervention records are reused from the frozen PoC-1
package (`poc.types`) rather than duplicated:

    I_p            RepairOption.intervention  (poc.types.Intervention; executable only)
    M(S)           RepairGroup                (mutually exclusive alternatives)
    B0, C*         CauseResult                (original causal diagnosis)
    S*             RepairOracleResult         (minimal valid corrective sets)
"""

from dataclasses import dataclass
from enum import Enum

from poc.types import EntityRole, Intervention, InterventionKind

FAMILIES = ("make_space", "storage_insertion", "storage_extraction", "articulated_opening")


def _check_ids(ids, what: str) -> None:
    if any(not i for i in ids):
        raise ValueError(f"{what}: empty id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{what}: duplicate ids {sorted(ids)}")


def _check_box(center, half, what: str) -> None:
    if len(center) != 3 or len(half) != 3 or any(h <= 0 for h in half):
        raise ValueError(f"{what}: need 3D centre and positive 3D half-extents")


@dataclass(frozen=True)
class PlacementRegion:
    """Axis-aligned box region where a relocated object may be placed."""
    region_id: str
    center: tuple[float, float, float]
    half: tuple[float, float, float]

    def __post_init__(self) -> None:
        _check_ids([self.region_id], "PlacementRegion")
        _check_box(self.center, self.half, f"PlacementRegion {self.region_id!r}")


@dataclass(frozen=True)
class RepairOption:
    """One executable candidate intervention I_p (a diagnostic is never a repair option)."""
    intervention: Intervention
    region_id: str | None = None  # destination region of a relocation; None for target repositioning

    def __post_init__(self) -> None:
        if self.intervention.is_diagnostic:
            raise ValueError("diagnostic interventions are never executable repair options")
        relocate = self.intervention.kind is InterventionKind.RELOCATE
        if relocate != (self.region_id is not None):
            raise ValueError("a relocation names its destination region; a repositioning names none")

    @property
    def option_id(self) -> str:
        return self.intervention.intervention_id

    @property
    def cost(self) -> int:
        return self.intervention.length


@dataclass(frozen=True)
class RepairGroup:
    """Mutually exclusive alternatives: at most one member may be selected (choice consistency M)."""
    group_id: str
    option_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_ids([self.group_id], "RepairGroup")
        _check_ids(self.option_ids, f"RepairGroup {self.group_id!r}")
        if len(self.option_ids) < 2:
            raise ValueError(f"RepairGroup {self.group_id!r} needs at least two alternatives")


@dataclass(frozen=True)
class CauseResult:
    """Original causal diagnosis: F0, direct blockers B0 and all minimal causal sets C*."""
    F0: int
    blockers: frozenset[str]                 # B0 = {o_i : c_i^0 > 0}
    minimal_causes: frozenset[frozenset[str]]  # all inclusion-minimal C with F^{do(ignore C)} = 0

    def __post_init__(self) -> None:
        if self.F0 not in (0, 1):
            raise ValueError("F0 must be 0 or 1")
        if self.F0 == 0 and (self.blockers or self.minimal_causes):
            raise ValueError("a feasible action has no blockers and no causal sets")
        if self.F0 == 1 and not (self.blockers and self.minimal_causes):
            raise ValueError("an infeasible action needs a non-empty B0 and at least one causal set")
        if any(not C or not C <= self.blockers for C in self.minimal_causes):
            raise ValueError("every causal set is a non-empty subset of B0")
        if any(a < b for a in self.minimal_causes for b in self.minimal_causes):
            raise ValueError("causal sets must be inclusion-minimal")


class RepairStatus(Enum):
    FEASIBLE = "FEASIBLE"                                  # original action already feasible: S* = {{}}
    REPAIRED = "REPAIRED"                                  # at least one valid corrective set exists
    NO_RECOURSE_IN_CATALOGUE = "NO_RECOURSE_IN_CATALOGUE"  # none inside the candidate catalogue


@dataclass(frozen=True)
class RepairOracleResult:
    """Exact corrective outcome: all tied minimal sets with M = V = 1 and F = 0."""
    status: RepairStatus
    minimal_repairs: frozenset[frozenset[str]]
    cost: int | None              # K(S*) shared by every tied minimal repair
    P: int                        # candidate count
    n_choice_invalid: int         # subsets with M(S) = 0
    n_static_invalid: int         # choice-consistent subsets with V(S) = 0

    def __post_init__(self) -> None:
        if self.P < 0 or self.n_choice_invalid < 0 or self.n_static_invalid < 0:
            raise ValueError("counts must be non-negative")
        if self.n_choice_invalid + self.n_static_invalid > 2 ** self.P:
            raise ValueError("more invalid subsets than subsets")
        if self.status is RepairStatus.NO_RECOURSE_IN_CATALOGUE:
            if self.minimal_repairs or self.cost is not None:
                raise ValueError("no recourse: no repair and no cost (never invent a repair)")
        elif not self.minimal_repairs or self.cost is None or self.cost < 0:
            raise ValueError("a repaired / feasible result needs repairs and a cost")
        if self.status is RepairStatus.FEASIBLE and (self.minimal_repairs != {frozenset()} or self.cost != 0):
            raise ValueError("an already feasible action has S* = {{}} at cost 0")
        if self.status is RepairStatus.REPAIRED and frozenset() in self.minimal_repairs:
            raise ValueError("an infeasible action cannot be repaired by the empty set")

    @property
    def n_tied(self) -> int:
        return len(self.minimal_repairs)


@dataclass(frozen=True)
class SceneRecord:
    """Deterministically reconstructable scene metadata plus its oracle labels."""
    scene_id: str
    seed: int
    family: str
    regions: tuple[PlacementRegion, ...]
    options: tuple[RepairOption, ...]
    groups: tuple[RepairGroup, ...]
    cause: CauseResult
    repair: RepairOracleResult

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        _check_ids([self.scene_id], "SceneRecord")
        _check_ids([r.region_id for r in self.regions], f"scene {self.scene_id!r} regions")
        option_ids = [o.option_id for o in self.options]
        _check_ids(option_ids, f"scene {self.scene_id!r} options")
        regions = {r.region_id for r in self.regions}
        if any(o.region_id is not None and o.region_id not in regions for o in self.options):
            raise ValueError("a relocation must target a declared placement region")
        if any(i not in option_ids for g in self.groups for i in g.option_ids):
            raise ValueError("a repair group may only contain declared options")
        if self.repair.P != len(self.options):
            raise ValueError("repair.P must equal the number of candidate options")
        if any(i not in option_ids for S in self.repair.minimal_repairs for i in S):
            raise ValueError("a minimal repair may only contain declared executable options")


__all__ = ["FAMILIES", "PlacementRegion", "RepairOption", "RepairGroup", "CauseResult", "RepairStatus",
           "RepairOracleResult", "SceneRecord", "EntityRole", "Intervention", "InterventionKind"]
