"""PoC-3 Stage 0 contract tests: frozen dependency integrity, imports, recorded environment, layout."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import poc
import poc2
import poc3
from poc3 import types as t3

REPO = Path(__file__).resolve().parents[2]
POC3 = REPO / "PoC-3"
LAYOUT = {  # plan3.md section 45 + approved Stage-4.5 / 4.6A-C diagnostic exceptions; nothing else without approval
    "src/poc3": {"__init__.py", "types.py", "features.py", "dataset.py", "model.py", "train.py", "inference.py",
                 "metrics.py", "relational.py", "feasibility.py"},
    "scripts": {"s1_features.py", "s2_dataset.py", "s3_baselines.py", "s4_energy.py", "s5_solvers.py", "s6_ood.py",
                "s7_verdict.py", "s45_diagnostic.py", "s46a_error_audit.py",
                "s46b_relational_pair.py", "s46c_feasibility_supervision.py"},
    "tests": {"test_core.py", "test_features.py", "test_dataset.py", "test_model.py", "test_training.py",
              "test_inference.py", "test_metrics.py", "test_relational.py", "test_s46a.py",
              "test_s46b.py", "test_feasibility.py", "test_s46c.py"},
}
HARD_MAX_LINES = 500


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def git_ok() -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    try:
        _git("cat-file", "-e", f"{t3.POC2_FROZEN_SHA}^{{commit}}")
        _git("cat-file", "-e", f"{t3.POC1_APPROVED_SHA}^{{commit}}")
    except subprocess.CalledProcessError:
        pytest.skip("frozen commits not present in this clone")


# ------------------------------------------------------------ frozen dependency integrity

def test_recorded_tree_hashes_match_the_frozen_commits(git_ok):
    assert _git("rev-parse", f"{t3.POC2_FROZEN_SHA}:PoC-1").strip() == t3.POC1_TREE
    assert _git("rev-parse", f"{t3.POC1_APPROVED_SHA}:PoC-1").strip() == t3.POC1_TREE
    assert _git("rev-parse", f"{t3.POC2_FROZEN_SHA}:PoC-2").strip() == t3.POC2_TREE


@pytest.mark.parametrize("folder", ["PoC-1", "PoC-2"])
def test_tracked_files_are_unchanged_from_the_frozen_state(git_ok, folder):
    frozen = _git("ls-tree", "-r", t3.POC2_FROZEN_SHA, "--", folder)
    index = _git("ls-files", "-s", "--", folder)
    blobs = lambda listing, col: {ln.split("\t")[1]: ln.split()[col] for ln in listing.splitlines()}
    assert blobs(index, 1) == blobs(frozen, 2), "index differs from the frozen commit"
    assert _git("diff", "--name-only", t3.POC2_FROZEN_SHA, "--", folder) == "", "working tree differs"
    assert _git("ls-files", "--others", "--exclude-standard", "--", folder) == "", "untracked files added"


def test_frozen_poc2_is_an_ancestor_of_head(git_ok):
    subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", t3.POC2_FROZEN_SHA, "HEAD"], check=True)


# ------------------------------------------------------------ imports and environment record

def test_poc3_imports_alongside_frozen_packages():
    assert set(poc3.__all__) <= set(dir(poc3))
    assert poc3.POC2_FROZEN_SHA.startswith("bb6f335")
    for pkg, folder in ((poc, "PoC-1"), (poc2, "PoC-2"), (poc3, "PoC-3")):
        assert Path(pkg.__file__).resolve().is_relative_to(REPO / folder / "src"), "imported from a copy"


def test_environment_records_versions_device_and_determinism():
    env = t3.RunEnvironment.capture()
    rec = json.loads(json.dumps(env.to_json()))
    assert set(rec) == {"python", "numpy", "torch", "mujoco", "git_sha", "git_dirty", "device", "deterministic"}
    assert {k: t3.base_version(rec[k]) for k in t3.VALIDATED_VERSIONS} == t3.VALIDATED_VERSIONS
    assert rec["device"] == "cpu" or rec["device"].startswith("cuda:")
    assert set(rec["deterministic"]) == {"torch_deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark"}


def test_seeds_match_the_plan():
    assert (t3.DATASET_SEEDS, t3.TRAINING_SEEDS, t3.BOOTSTRAP_SEED) == ((31, 41, 51), (7, 17, 27), 107)


# ------------------------------------------------------------ repository layout discipline

def test_only_planned_files_exist_within_line_limits():
    for sub, allowed in LAYOUT.items():
        found = {p.name for p in (POC3 / sub).glob("*.py")} if (POC3 / sub).exists() else set()
        assert found <= allowed, f"unplanned files in PoC-3/{sub}: {found - allowed}"
        for name in found:
            assert len((POC3 / sub / name).read_text().splitlines()) <= HARD_MAX_LINES
    top = {p.name for p in POC3.iterdir()} - {"out", ".pytest_cache"}
    assert top <= {"plan3.md", "pyproject.toml", "src", "scripts", "tests"}, top
