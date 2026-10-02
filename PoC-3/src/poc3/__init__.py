"""PoC-3: learned intervention energy for minimal corrective recourse (see PoC-3/plan3.md).

PoC-3 imports the frozen PoC-1 package `poc` and the frozen PoC-2 package `poc2` (pinned at
POC2_FROZEN_SHA) for scenes, the exact oracle and exact M / V / K; it owns only the learning logic.
"""

from poc3.types import (
    BOOTSTRAP_SEED,
    DATASET_SEEDS,
    POC1_APPROVED_SHA,
    POC2_FROZEN_SHA,
    TRAINING_SEEDS,
    VALIDATED_VERSIONS,
    RunEnvironment,
)

__all__ = ["BOOTSTRAP_SEED", "DATASET_SEEDS", "POC1_APPROVED_SHA", "POC2_FROZEN_SHA", "TRAINING_SEEDS",
           "VALIDATED_VERSIONS", "RunEnvironment"]
