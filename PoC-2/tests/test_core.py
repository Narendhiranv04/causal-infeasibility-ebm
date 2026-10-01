"""PoC-2 Stage 0 contract tests: repository split integrity, imports, record invariants."""

import shutil
import subprocess
from pathlib import Path

import pytest

import poc
import poc.types
import poc2
from poc.types import Intervention, InterventionKind
from poc2 import types as t2

REPO = Path(__file__).resolve().parents[2]
FROZEN = "1d15bb57812d771e352174bdcb5e5d8fef214294"  # validated PoC-1 commit before the split
MOVED = ("plan.md", "pyproject.toml", "src", "scripts", "tests")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def frozen_files() -> dict[str, str]:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    try:
        listing = _git("ls-tree", "-r", FROZEN, "--", *MOVED)
    except subprocess.CalledProcessError:
        pytest.skip("frozen PoC-1 commit not present in this clone")
    return {line.split("\t")[1]: line.split()[2] for line in listing.splitlines()}


# ------------------------------------------------------------ repository split

def test_poc1_files_are_byte_identical_to_the_frozen_commit(frozen_files):
    for path, blob in frozen_files.items():
        assert _git("hash-object", f"PoC-1/{path}").strip() == blob, f"PoC-1/{path} content changed"


def test_poc1_tracked_tree_is_exactly_the_frozen_tree(frozen_files):
    tracked = set(_git("ls-files", "PoC-1").split())
    assert tracked == {f"PoC-1/{p}" for p in frozen_files}


def test_poc1_imports_from_its_new_location():
    import poc.cases  # noqa: F401  (MuJoCo-backed modules import cleanly too)
    import poc.hopfield  # noqa: F401
    assert Path(poc.__file__).resolve().parent == REPO / "PoC-1" / "src" / "poc"


def test_poc2_imports_and_reuses_poc1_records():
    assert Path(poc2.__file__).resolve().parent == REPO / "PoC-2" / "src" / "poc2"
    assert t2.Intervention is poc.types.Intervention and t2.InterventionKind is poc.types.InterventionKind
    source = (REPO / "PoC-2" / "src" / "poc2" / "types.py").read_text()
    assert "class Intervention" not in source and "class Entity" not in source, "no duplicated PoC-1 records"


# ------------------------------------------------------------ record invariants

def _relocate(oid: str, entity: str, cost: int = 1) -> Intervention:
    return Intervention(oid, InterventionKind.RELOCATE, entity, length=cost, params=(0.0, 0.0, 0.0))


def test_diagnostic_interventions_are_never_repair_options():
    with pytest.raises(ValueError, match="diagnostic"):
        t2.RepairOption(Intervention("d", InterventionKind.REMOVE, "wall"))


def test_relocation_names_a_region_and_repositioning_does_not():
    assert t2.RepairOption(_relocate("A_r1", "A"), "r1").cost == 1
    with pytest.raises(ValueError, match="region"):
        t2.RepairOption(_relocate("A_r1", "A"))
    shift = Intervention("shift", InterventionKind.SHIFT_TARGET, "obj", params=(0.0, -0.03, 0.0))
    with pytest.raises(ValueError, match="region"):
        t2.RepairOption(shift, "r1")
    assert t2.RepairOption(shift).region_id is None


def test_repair_group_is_a_set_of_at_least_two_alternatives():
    t2.RepairGroup("A", ("A_r1", "A_r2"))
    for bad in (("A_r1",), ("A_r1", "A_r1")):
        with pytest.raises(ValueError):
            t2.RepairGroup("A", bad)


def test_placement_region_needs_positive_extent():
    with pytest.raises(ValueError):
        t2.PlacementRegion("r1", (0.0, 0.0, 0.0), (0.1, 0.0, 0.1))


def test_cause_result_invariants():
    B0 = frozenset({"jamb", "box"})
    t2.CauseResult(1, B0, frozenset({frozenset({"jamb", "box"})}))
    t2.CauseResult(0, frozenset(), frozenset())
    for F0, blockers, causes in [
        (0, B0, frozenset()),                                            # feasible with blockers
        (1, B0, frozenset()),                                            # infeasible without a cause
        (1, B0, frozenset({frozenset({"wall"})})),                       # cause outside B0
        (1, B0, frozenset({frozenset({"jamb"}), frozenset({"jamb", "box"})})),  # not inclusion-minimal
    ]:
        with pytest.raises(ValueError):
            t2.CauseResult(F0, blockers, causes)


def test_repair_oracle_result_invariants():
    S = frozenset({frozenset({"A_r1", "shift"}), frozenset({"B_r2", "shift"})})
    assert t2.RepairOracleResult(t2.RepairStatus.REPAIRED, S, 2, 6, 3, 1).n_tied == 2
    t2.RepairOracleResult(t2.RepairStatus.FEASIBLE, frozenset({frozenset()}), 0, 3, 0, 0)
    t2.RepairOracleResult(t2.RepairStatus.NO_RECOURSE_IN_CATALOGUE, frozenset(), None, 3, 0, 0)
    for args in [
        (t2.RepairStatus.NO_RECOURSE_IN_CATALOGUE, S, 2, 6, 0, 0),         # invented repair
        (t2.RepairStatus.REPAIRED, frozenset({frozenset()}), 0, 3, 0, 0),  # empty set repairs nothing
        (t2.RepairStatus.REPAIRED, S, 2, 2, 5, 0),                         # more invalid subsets than 2^P
    ]:
        with pytest.raises(ValueError):
            t2.RepairOracleResult(*args)


def test_scene_record_cross_references():
    region = t2.PlacementRegion("r1", (0.1, 0.2, 0.05), (0.05, 0.05, 0.05))
    opts = (t2.RepairOption(_relocate("A_r1", "A"), "r1"),)
    cause = t2.CauseResult(1, frozenset({"A"}), frozenset({frozenset({"A"})}))
    repair = t2.RepairOracleResult(t2.RepairStatus.REPAIRED, frozenset({frozenset({"A_r1"})}), 1, 1, 0, 0)
    t2.SceneRecord("s0", 7, "make_space", (region,), opts, (), cause, repair)
    with pytest.raises(ValueError, match="family"):
        t2.SceneRecord("s0", 7, "drawer", (region,), opts, (), cause, repair)
    with pytest.raises(ValueError, match="region"):
        t2.SceneRecord("s0", 7, "make_space", (), opts, (), cause, repair)
    bad = t2.RepairOracleResult(t2.RepairStatus.REPAIRED, frozenset({frozenset({"ghost"})}), 1, 1, 0, 0)
    with pytest.raises(ValueError, match="declared"):
        t2.SceneRecord("s0", 7, "make_space", (region,), opts, (), cause, bad)
