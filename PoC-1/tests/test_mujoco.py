"""Stage 4 contract tests: MuJoCo signed distance, 3D envelope oracle, scene families."""

import io
import math
import tokenize
from dataclasses import replace
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


# ------------------------------------------------------------ Stage 5

from poc import energy as en  # noqa: E402
from poc.oracle import admissible_minimal_repairs, oracle_repair_cost  # noqa: E402

S5 = {c.name: c for c in cs.stage5_cases()}


@pytest.fixture(scope="module")
def tables():
    return {n: cs.exhaustive_table(c) for n, c in S5.items()}


def _subset(case, x):
    return [iv for p, iv in enumerate(case.candidates) if x >> p & 1]


def _repairs(case, T, xs):
    return frozenset(frozenset(iv.intervention_id for iv in _subset(case, x)) for x in xs)


@pytest.mark.parametrize("name", ["S5_ins_coupled", "S5_hinge_validity", "S5_ext_k2", "S5_hinge_substitutable"])
def test_exhaustive_table_equals_direct_intervention_evaluation(name, tables):
    case, T = S5[name], tables[name]
    for x in range(len(T.K)):
        direct, _ = cs.evaluate(cs.do(case.scene, _subset(case, x)))
        assert direct.d == tuple(T.d[x]) and direct.d_start == tuple(T.d_start[x]) and direct.d_goal == tuple(T.d_goal[x])
        assert cs.validity(cs.do(case.scene, _subset(case, x)))[0] == T.V[x]


@pytest.mark.parametrize("name", sorted(S5))
def test_stage5_admissible_minimal_repairs_match_expectation(name, tables):
    case, T = S5[name], tables[name]
    assert _repairs(case, T, admissible_minimal_repairs(T.F, T.V, T.K, case.k_max)) == case.expected
    assert T.V[0] == 1, "every catalogue scene is valid before intervention"
    if case.structure != "validity":
        assert np.all(T.V == 1), "distinct staging poses: no invalid subset outside the validity probe"


def test_validity_filter_blocks_a_feasible_but_invalid_repair(tables):
    case, T = S5["S5_hinge_validity"], tables["S5_hinge_validity"]
    x_shift = 1 << [iv.intervention_id for iv in case.candidates].index("shift_lid")
    assert T.F[x_shift] == 0 and T.V[x_shift] == 0  # the lid clears, but the moved box sits inside the low bottle
    J = [oracle_repair_cost(int(f), int(k), case.k_max) for f, k in zip(T.F, T.K)]
    assert x_shift in en.argmin_sets(J), "without V the invalid single repositioning would win"
    assert x_shift not in admissible_minimal_repairs(T.F, T.V, T.K, case.k_max)


def test_admissible_minimal_repairs_semantics():
    F, V, K = np.array([1, 0, 0, 0]), np.array([1, 0, 1, 1]), np.array([0, 1, 1, 2])
    assert admissible_minimal_repairs(F, V, K, 2) == {2}
    assert admissible_minimal_repairs(np.ones(2, int), np.ones(2, int), np.array([0, 1]), 1) == frozenset()


def test_significance_rule():
    base, fine = np.array([0.0, 0.02, 0.02, 5e-7, 1e-3, -0.01]), np.array([0.0, 0.02, -0.02, 5e-7, 7e-4, -0.0101])
    assert en.significant(base, fine, 1e-6).tolist() == [False, True, False, False, False, True]


@pytest.mark.parametrize("name, k", [("S5_ins_k6", 6), ("S5_ext_k5", 5), ("S5_hinge_k5", 5)])
def test_high_cardinality_repairs_are_unary(name, k, tables):
    case, T = S5[name], tables[name]
    assert {len(s) for s in _repairs(case, T, admissible_minimal_repairs(T.F, T.V, T.K, case.k_max))} == {k}
    assert np.max(np.abs(T.G - en.approximate(T.G, 1))) <= 1e-12
    assert len(case.candidates) > k, "includes distractor candidates"


@pytest.mark.parametrize("name, macro, other", [("S5_ins_coupled", "shift_obj", "r_nb"),
                                                ("S5_hinge_coupled", "shift_lid", "r_bt")])
def test_repositioning_creates_significant_physical_pair_term(name, macro, other, tables):
    case, T = S5[name], tables[name]
    fine = cs.exhaustive_table(case, ev.ENVELOPE_STEP / 2)
    names = [iv.intervention_id for iv in case.candidates]
    x = (1 << names.index(macro)) | (1 << names.index(other))
    a_b, a_f = en.mobius(T.G), en.mobius(fine.G)
    assert a_b[x] < -0.01 and en.significant(a_b, a_f, 1e-6)[x]
    assert abs(T.G[1 << names.index(other)] - T.G[0]) <= 1e-12, "relocating the neighbour alone does nothing"


def test_perturbations_change_the_repair_only_across_the_boundary(tables):
    base = S5["S5_ins_k3"].expected
    for tag in ("p_minus", "p_plus", "cross_in"):
        assert S5[f"S5_ins_k3_{tag}"].expected == base
    assert S5["S5_ins_k3_cross_out"].expected == {frozenset({"r_b1", "r_b3"})}


def test_shift_target_moves_the_fixture_with_the_target():
    case = S5["S5_hinge_coupled"]
    moved = cs.do(case.scene, [case.candidates[0]])
    assert moved.fixture[0].center[0] == pytest.approx(case.scene.fixture[0].center[0] - 0.12)
    assert moved.motion.pivot[0] == pytest.approx(case.scene.motion.pivot[0] - 0.12)


def _fixture_into_structure_case():
    """Front shelf forces moving the box back 12 cm, but a low STRUCTURAL block sits where the box body would go."""
    block = cs._ent("block", EntityRole.STRUCTURAL, (-0.10, 0.0, 0.05), (0.03, 0.10, 0.05))  # below the lid pivot
    scene = replace(cs.lid_scene(cs.front_shelf(), block, cs.item("d1", -0.30, 0.15, 0.30)), fixture=(cs.BASE,))
    cands = (cs.shift("lid", (-0.12, 0.0, 0.0)), cs.staged(scene.entities[2], -0.35, -0.45))
    return cs.Stage5Case("fixture_probe", "validity", scene, cands, frozenset())


def test_shifted_target_fixture_penetrating_structure_is_invalid():
    case = _fixture_into_structure_case()
    moved = cs.do(case.scene, [case.candidates[0]])
    assert cs.validity(case.scene)[0] == 1 and cs.validity(moved)[0] == 0
    T = cs.exhaustive_table(case)
    x_shift = 1
    assert T.F[x_shift] == 0 and T.V[x_shift] == 0 and T.margin[x_shift] < -CONTACT_TOL_3D
    assert x_shift not in admissible_minimal_repairs(T.F, T.V, T.K, case.k_max)
    assert admissible_minimal_repairs(T.F, T.V, T.K, case.k_max) == frozenset()  # no admissible repair exists


@pytest.mark.parametrize("r1, r2, checked", [
    ("movable", "movable", True), ("movable", "structural", True), ("target", "movable", True),
    ("target", "structural", True), ("structural", "structural", False)])
def test_validity_pair_rule_by_role(r1, r2, checked):
    assert cs.checked_pair(Entity("a", EntityRole(r1)), Entity("b", EntityRole(r2))) is checked
