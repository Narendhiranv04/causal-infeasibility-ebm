"""Stage 4 contract tests: MuJoCo signed distance, 3D envelope oracle, scene families."""

import io
import math
import tokenize
from pathlib import Path

import numpy as np
import pytest

from poc import cases as cs
from poc import envelope as ev
from poc.mj_scene import DISTMAX, Box3D, Composite, Entity3D, GeomWorld
from poc.oracle import CONTACT_TOL_3D, assess, conflict_3d, diagnostic_causes, feasibility_excluding
from poc.types import Entity, EntityRole, Intervention, InterventionKind

CUBE = Entity3D(Entity("cube", EntityRole.STRUCTURAL), (Box3D((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)),))
CASES = {c.name: c for c in cs.canonical_cases()}


def _pair(half=(0.1, 0.1, 0.1), quat=(1.0, 0.0, 0.0, 0.0)):
    world = GeomWorld((CUBE,), Composite((("probe", Box3D((0.0, 0.0, 0.0), half, quat)),)))
    def d(pos):
        world.set_pose(pos, (1.0, 0.0, 0.0, 0.0))
        return world.signed_distance(world.moving_geoms[0], world.entity_geoms[0][0])
    return world, d


@pytest.fixture(scope="module")
def results():
    return {n: cs.evaluate(c.scene) for n, c in CASES.items()}


# ------------------------------------------------------- geometry sanity

@pytest.mark.parametrize("x, expected", [
    (0.5, 0.3), (0.25, 0.05),                        # positive separation
    (0.2, 0.0), (0.200001, 1e-6), (0.199999, -1e-6),  # touching / near touching
    (0.199, -0.001), (0.19, -0.01),                  # shallow penetration
    (0.1, -0.1), (0.02, -0.18),                      # deeper penetration
])
def test_axis_aligned_box_signed_distance(x, expected):
    _, d = _pair()
    assert d((x, 0.0, 0.0)) == pytest.approx(expected, abs=1e-9)


def test_diagonal_and_rotated_boxes():
    _, d = _pair()
    assert d((0.3, 0.3, 0.0)) == pytest.approx(math.sqrt(2) * 0.1, abs=1e-9)
    q45 = (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))  # 45 deg about z
    _, dr = _pair(quat=q45)
    corner = 0.1 + 0.1 * math.sqrt(2)
    assert dr((0.3, 0.0, 0.0)) == pytest.approx(0.3 - corner, abs=1e-9)
    assert dr((0.2, 0.0, 0.0)) == pytest.approx(0.2 - corner, abs=1e-9)


def test_exact_zero_ambiguities_are_resolved():
    world, d = _pair(half=(0.1, 0.05, 0.05))
    world.set_pose((0.05, 0.25, 0.05), (1.0, 0.0, 0.0, 0.0))
    raw = world._raw(world.moving_geoms[0], world.entity_geoms[0][0], DISTMAX)
    assert raw == 0.0, "MuJoCo 3.10 native box-box: spurious 0.0 for this separated face-to-face pair"
    assert d((0.05, 0.25, 0.05)) == pytest.approx(0.1, abs=1e-9)
    _, dc = _pair()
    assert dc((0.0, 0.0, 0.0)) == pytest.approx(-0.2, abs=2e-9)  # coincident centres (raw 0.0)


def test_wrong_sign_separation_is_certified_and_recovered():
    """Hinge pose found by 1 mm sampling of C3_boundary_out: raw MuJoCo reports -0.1044 for a +0.1044 gap."""
    scene = CASES["C3_boundary_out"].scene
    world = GeomWorld(scene.entities, scene.moving)
    taus, frames = ev.poses(scene.motion, scene.moving, 1e-3)
    world.set_pose(*frames[149])
    gm, ge = world.moving_geoms[0], world.entity_geoms[0][0]
    assert world._raw(gm, ge, DISTMAX) < 0.0, "MuJoCo 3.10 native box-box: sign-flipped separation"
    assert world.signed_distance(gm, ge) == pytest.approx(0.1043819, abs=1e-6)


def test_distances_beyond_distmax_are_clamped():
    _, d = _pair()
    assert d((0.2 + DISTMAX + 0.1, 0.0, 0.0)) == DISTMAX


def test_conflict_tolerance_semantics():
    ids = ("a", "b")
    a = assess(ids, [-0.5 * CONTACT_TOL_3D, 0.02], [0.0, 0.0], [0.0, 0.0])
    assert a.F == 0 and a.G == 0.0 and a.p[0] > 0.0  # touching penetration kept in p, not in c
    b = assess(ids, [-2 * CONTACT_TOL_3D, 0.02], [0.0, 0.0], [0.0, 0.0])
    assert b.F == 1 and b.c[0] == pytest.approx(CONTACT_TOL_3D) and b.d[0] == -2 * CONTACT_TOL_3D
    assert np.all(conflict_3d([0.1, 0.0, -CONTACT_TOL_3D]) == 0.0)


# -------------------------------------------------------------- envelope

@pytest.mark.parametrize("name", ["A2_single_blocker", "B2_single_blocker", "C2_single_blocker"])
@pytest.mark.parametrize("step", [ev.ENVELOPE_STEP, ev.ENVELOPE_STEP / 2])
def test_sampling_respects_vertex_displacement_bound(name, step):
    scene = CASES[name].scene
    taus, frames = ev.poses(scene.motion, scene.moving, step)
    assert taus[0] == 0.0 and taus[-1] == 1.0 and np.allclose(np.diff(taus), taus[1])
    V = np.array([ev.world_corners(scene.moving, p, q) for p, q in frames])
    assert np.linalg.norm(np.diff(V, axis=0), axis=-1).max() <= step + 1e-12


def test_no_phase_identifiers_in_the_implementation():
    for path in Path("src/poc").glob("*.py"):
        names = [t.string.lower() for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
                 if t.type == tokenize.NAME]
        assert not any("phase" in n for n in names), path


# --------------------------------------------------------- scene families

@pytest.mark.parametrize("name", sorted(CASES))
def test_canonical_labels_and_diagnostic_causes(name, results):
    case, (a, _) = CASES[name], results[name]
    assert a.F == int(case.kind not in ("feasible", "near_boundary_out"))
    assert diagnostic_causes(a) == case.cause
    assert a.F == 1 or a.G == 0.0, "every feasible state has G = 0"
    if case.cause:
        assert feasibility_excluding(a, case.cause) == 0


@pytest.mark.parametrize("name", sorted(n for n, c in CASES.items() if c.repair))
def test_executable_repair_restores_feasibility(name):
    case = CASES[name]
    assert cs.evaluate(cs.do(case.scene, case.repair))[0].F == 0


@pytest.mark.parametrize("family", "ABC")
def test_near_boundary_pairs_flip_at_the_analytic_boundary(family, results):
    (out_name,) = [n for n in CASES if n.startswith(family) and n.endswith("out")]
    in_name = out_name.replace("out", "in")
    eid = CASES[in_name].cause[0]
    d = {n: dict(zip(results[n][0].ids, results[n][0].d))[eid] for n in (out_name, in_name)}
    assert d[out_name] == pytest.approx(cs.DELTA, abs=1e-5) and d[in_name] == pytest.approx(-cs.DELTA, abs=1e-5)
    assert (results[out_name][0].F, results[in_name][0].F) == (0, 1)


@pytest.mark.parametrize("name", ["A2_single_blocker", "A4_immovable_cause", "B2_single_blocker",
                                  "B4_immovable_cause", "C3_boundary_in", "C4_immovable_cause"])
def test_endpoints_free_but_envelope_infeasible(name, results):
    a, sw = results[name]
    assert a.interior_only
    i = a.ids.index(CASES[name].cause[0])
    assert sw.d_start[i] > 0 and sw.d_goal[i] > 0 and sw.distances[1:-1, i].min() < -CONTACT_TOL_3D


@pytest.mark.parametrize("name", [n for n, c in CASES.items() if c.kind == "immovable_cause"])
def test_structural_cause_differs_from_repair_target(name):
    case = CASES[name]
    structural = {e.eid for e in case.scene.entities if not e.entity.movable}
    assert set(case.cause) <= structural
    assert {iv.entity_id for iv in case.repair}.isdisjoint(case.cause)
    with pytest.raises(ValueError, match="structural"):
        cs.apply_intervention(case.scene, cs.relocate(case.cause[0], (1.0, 1.0, 1.0)))


def test_diagnostics_are_not_scene_edits_or_repairs():
    remove = Intervention("d_jamb", InterventionKind.REMOVE, "jamb")
    with pytest.raises(ValueError, match="oracle queries"):
        cs.apply_intervention(CASES["A4_immovable_cause"].scene, remove)
    with pytest.raises(ValueError, match="never an executable repair"):
        cs.Case3D("bad", "immovable_cause", CASES["A4_immovable_cause"].scene, ("jamb",), (remove,))


@pytest.mark.parametrize("name", sorted(CASES))
def test_finer_sampling_does_not_change_labels(name, results):
    base, _ = results[name]
    fine, _ = cs.evaluate(CASES[name].scene, ev.ENVELOPE_STEP / 2)
    assert fine.F == base.F and fine.blockers == base.blockers
    # d is 1-Lipschitz in rigid motion and every pose lies within step/2 of a sample
    assert np.max(np.abs(np.subtract(fine.d, base.d))) <= 0.75 * ev.ENVELOPE_STEP


def test_rebuilt_scene_reproduces_identical_conflict(results):
    for name in ("A2_single_blocker", "C4_immovable_cause"):
        again, _ = cs.evaluate(CASES[name].scene)
        assert again == results[name][0]
