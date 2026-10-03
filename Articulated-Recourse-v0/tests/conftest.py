import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def problems():
    """Canonical problems (coarse resolution, with robot admissibility), built once."""
    from artrecourse.canonical import ALL
    from artrecourse.interventions import build_problem
    from artrecourse.oracle import solve

    out = {}
    for v in ("V3", "V5", "V7"):
        pb = build_problem(ALL[v]())
        out[v] = (pb, solve(pb))
    return out
