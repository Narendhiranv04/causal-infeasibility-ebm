"""Shared data records for the PoC (plan.md section 3).

Only immutable records and enums live here. No geometry, no oracle,
no solver logic. Symbols follow the plan's notation:

    a            ActionSpec          next manipulation action
    o_i          Entity              known scene entity
    I_p          Intervention        candidate intervention
    S            tuple[Intervention] selected intervention set, S(x) = {I_p : x_p = 1}
    c^S          RepairResult.conflict   per-entity conflict after do(S)
    G(S)         RepairResult.G      graded remaining conflict, G = g(c^S)
    F(S)         RepairResult.F      binary feasibility oracle (0 feasible, 1 infeasible)
    K(S)         RepairResult.K      number of primitive repair actions in S
    J*(S)        RepairResult.J      oracle repair objective B*F(S) + K(S)

Uppercase field names (G, F, K, J) intentionally mirror the plan's notation.
"""

from dataclasses import dataclass
from enum import Enum


class EntityRole(Enum):
    """What an entity is in the scene, independent of its geometry."""

    TARGET = "target"          # object manipulated by the action a
    MOVABLE = "movable"        # can be relocated by an executable repair
    STRUCTURAL = "structural"  # fixed scene structure (wall, shelf, frame)


class InterventionKind(Enum):
    """Kinds of do(.) operations.

    Executable kinds are robot repairs and may appear in a repair set S.
    Diagnostic kinds only establish causality (e.g. "remove wall") and
    must never be reported as executable robot repairs.
    """

    RELOCATE = "relocate"                     # move a movable entity to a staging pose
    SHIFT_TARGET = "shift_target"             # shift the target object before retry
    REMOVE = "remove"                         # diagnostic: delete entity from scene
    DISABLE_COLLISION = "disable_collision"   # diagnostic: ignore entity geometry

    @property
    def is_diagnostic(self) -> bool:
        return self in _DIAGNOSTIC_KINDS

    @property
    def is_executable(self) -> bool:
        return not self.is_diagnostic


_DIAGNOSTIC_KINDS = frozenset({InterventionKind.REMOVE, InterventionKind.DISABLE_COLLISION})


@dataclass(frozen=True)
class ActionSpec:
    """Next manipulation action a.

    `params` parametrizes the continuous motion over tau in [0, 1]
    (e.g. path endpoints, hinge axis and angle range). Any trajectory
    sampling used to approximate the envelope E(a, s) is a numerical
    detail of the oracle, never a semantic action phase.
    """

    action_id: str
    target_id: str
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id or not self.target_id:
            raise ValueError("ActionSpec requires non-empty action_id and target_id")


@dataclass(frozen=True)
class Entity:
    """Geometry-free metadata for a scene entity o_i.

    Geometry G(o_i) is stage-specific (2D polygons, MuJoCo geoms) and is
    owned by the stage module, keyed by `entity_id`.
    """

    entity_id: str
    role: EntityRole

    @property
    def movable(self) -> bool:
        return self.role is not EntityRole.STRUCTURAL


@dataclass(frozen=True)
class Intervention:
    """Candidate intervention I_p.

    `entity_id` is the repair target (the entity acted on), which may
    differ from the entity that causes the conflict. `length` is the
    explicit integer macro length contributing to K(S). `params` holds
    e.g. a staging pose or shift vector.
    """

    intervention_id: str
    kind: InterventionKind
    entity_id: str
    length: int = 1
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.intervention_id or not self.entity_id:
            raise ValueError("Intervention requires non-empty intervention_id and entity_id")
        if not isinstance(self.length, int) or isinstance(self.length, bool) or self.length < 1:
            raise ValueError(f"Intervention length must be an integer >= 1, got {self.length!r}")

    @property
    def is_diagnostic(self) -> bool:
        return self.kind.is_diagnostic


@dataclass(frozen=True)
class RepairResult:
    """Oracle evaluation of one executable intervention set S.

    Invariants enforced at construction:
      - S contains only executable interventions (no diagnostics),
      - S has set semantics (no duplicate intervention ids),
      - c^S >= 0 elementwise and G >= 0,
      - F in {0, 1},
      - K == sum of intervention lengths in S.
    """

    interventions: tuple[Intervention, ...]
    conflict: tuple[float, ...]
    G: float
    F: int
    K: int
    J: float

    def __post_init__(self) -> None:
        ids = [i.intervention_id for i in self.interventions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Repair set has duplicate interventions: {ids}")
        diagnostic = [i.intervention_id for i in self.interventions if i.is_diagnostic]
        if diagnostic:
            raise ValueError(f"Diagnostic interventions are not executable repairs: {diagnostic}")
        if any(c < 0.0 for c in self.conflict):
            raise ValueError("Conflict values c_i must be >= 0")
        if self.G < 0.0:
            raise ValueError("Graded conflict G must be >= 0")
        if self.F not in (0, 1):
            raise ValueError(f"Feasibility F must be 0 or 1, got {self.F!r}")
        expected_k = sum(i.length for i in self.interventions)
        if self.K != expected_k:
            raise ValueError(f"K={self.K} does not match summed intervention lengths {expected_k}")

    @property
    def feasible(self) -> bool:
        return self.F == 0

    @property
    def selected_ids(self) -> frozenset[str]:
        """S as an order-independent set of intervention ids."""
        return frozenset(i.intervention_id for i in self.interventions)
