"""PoC: intervention-grounded next-action infeasibility + structured repair.

Notation follows plan.md section 3. Shared record types live in `poc.types`.
"""

from poc.types import (
    ActionSpec,
    Entity,
    EntityRole,
    Intervention,
    InterventionKind,
    RepairResult,
)

__all__ = [
    "ActionSpec",
    "Entity",
    "EntityRole",
    "Intervention",
    "InterventionKind",
    "RepairResult",
]
