"""PoC-3 shared records: frozen dependency baseline and reproducibility record (plan3.md sections 45, 47, 75).

PoC-1 and PoC-2 are imported, never copied or modified. Their approved state is pinned by commit and
by git tree hash; tests/test_core.py verifies that every tracked file still matches it.
"""

import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

# Frozen dependency baseline.
POC2_FROZEN_SHA = "bb6f3352d16597fcf7a7294ffb251fac83d3ac31"   # PoC-2 complete (Stage 6)
POC1_APPROVED_SHA = "89bcfd73099231fe8026a8bbe428d00d260bb5c7"  # last approved PoC-1 change (signed-distance fix)
POC1_TREE = "f5ba99f3b9ae4a7200eac6459332f014916459b7"          # PoC-1/ tree at both commits above
POC2_TREE = "d5e6b6c2d71e0eed61f960f41842322d74b1dc0e"          # PoC-2/ tree at POC2_FROZEN_SHA

# Library versions PoC-3 Stage 0 was validated with (local build tags such as "+cu128" ignored).
VALIDATED_VERSIONS = {"numpy": "2.4.0", "torch": "2.9.1", "mujoco": "3.10.0"}

# Global seeds (plan3.md section 75).
DATASET_SEEDS = (31, 41, 51)   # dataset namespaces
TRAINING_SEEDS = (7, 17, 27)
BOOTSTRAP_SEED = 107

REPO = Path(__file__).resolve().parents[3]


def base_version(v: str) -> str:
    return v.split("+")[0]


@dataclass(frozen=True)
class RunEnvironment:
    """What every PoC-3 result JSON records: interpreter, library versions, git SHA, device, determinism."""
    python: str
    numpy: str
    torch: str
    mujoco: str
    git_sha: str
    git_dirty: bool
    device: str
    deterministic: dict

    @classmethod
    def capture(cls, repo: Path = REPO) -> "RunEnvironment":
        import mujoco
        import numpy
        import torch

        def git(*args: str) -> str:
            try:
                return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                                      text=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                return ""

        device = f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
        return cls(python=platform.python_version(), numpy=numpy.__version__, torch=torch.__version__,
                   mujoco=mujoco.__version__, git_sha=git("rev-parse", "HEAD") or "unknown",
                   git_dirty=bool(git("status", "--porcelain")), device=device,
                   deterministic={"torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                  "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                  "cudnn_benchmark": torch.backends.cudnn.benchmark})

    def to_json(self) -> dict:
        return asdict(self)
