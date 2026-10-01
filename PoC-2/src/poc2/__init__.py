"""PoC-2: causal diagnosis and structured corrective recourse (see PoC-2/plan2.md).

PoC-2 imports the frozen PoC-1 package `poc` for geometry, the oracle, energy,
QUBO and Hopfield utilities; it owns only the repair-space and structural logic.
"""

from poc2.types import (
    FAMILIES,
    CauseResult,
    PlacementRegion,
    RepairGroup,
    RepairOption,
    RepairOracleResult,
    RepairStatus,
    SceneRecord,
)

__all__ = ["FAMILIES", "CauseResult", "PlacementRegion", "RepairGroup", "RepairOption", "RepairOracleResult",
           "RepairStatus", "SceneRecord"]
