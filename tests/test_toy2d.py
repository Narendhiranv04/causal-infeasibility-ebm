"""Stage 1 contract tests: 2D envelope oracle + exhaustive intervention truth."""

import itertools

import numpy as np
import pytest

from poc.toy2d import (
    BOUNDARY_KINDS, CONTACT_TOL, LID, SAMPLE_STEP, Case2D, Intervention, InterventionKind,
    all_cases, apply_intervention, boundary_scene, conflict, diagnostic_causes, do,
    entity_margins, enumerate_subsets, envelope, evaluate, feasibility, graded_conflict,
    minimal_repairs, oracle_repair_cost, side_box, subset,
)
from poc.types import RepairResult

CASES = {case.name: case for case in all_cases()}


@pytest.fixture(scope="module")
def tables():
    return {name: enumerate_subsets(case) for name, case in CASES.items()}


def _by_ids(case, ids):
    return tuple(iv for iv in case.candidates if iv.intervention_id in ids)


# -------------------------------------------------------------- oracle / geometry

@pytest.mark.parametrize("name", ["T1_feasible", "H1_feasible"])
def test_feasible_hard_negatives_have_zero_conflict(name):
    scene = CASES[name].scene
    c = conflict(scene)
    assert np.all(c <= CONTACT_TOL) and feasibility(c) == 0 and graded_conflict(c) == 0.0
    assert np.all(entity_margins(scene) > -0.01), "hard negative: every entity within 1 cm"


def test_inserting_a_blocker_produces_positive_conflict():
    scene = CASES["T1_feasible"].scene
    blocked = type(scene)(scene.action, scene.motion, scene.entities + (side_box("new", 0.4, -1, 0.01),))
    c = conflict(blocked)
    assert c[-1] == pytest.approx(0.01, abs=1e-12)  # lateral lane intrusion, exact for translation
    assert np.all(c[:-1] == conflict(scene)) and feasibility(c) == 1


@pytest.mark.parametrize("kind", BOUNDARY_KINDS)
@pytest.mark.parametrize("delta", [1e-3, 1e-4, 1e-5])
def test_near_boundary_pairs_flip_for_geometric_reasons(kind, delta):
    outside, inside = boundary_scene(kind, -delta), boundary_scene(kind, +delta)
    assert feasibility(conflict(outside)) == 0 and feasibility(conflict(inside)) == 1
    assert entity_margins(outside)[0] == pytest.approx(-delta, abs=1e-9)  # clearance = delta
    assert entity_margins(inside)[0] == pytest.approx(+delta, abs=1e-9)   # intrusion = delta


@pytest.mark.parametrize("kind", BOUNDARY_KINDS)
def test_contact_tolerance_is_the_feasibility_threshold(kind):
    for depth, label in [(0.0, 0), (0.5 * CONTACT_TOL, 0), (2 * CONTACT_TOL, 1)]:
        assert feasibility(conflict(boundary_scene(kind, depth))) == label


def test_F_is_computed_from_c_with_tolerance_not_from_G():
    c_touch = np.full(5, 0.9 * CONTACT_TOL)  # five sub-tolerance touches
    assert feasibility(c_touch) == 0 and graded_conflict(c_touch) > CONTACT_TOL
    assert feasibility(np.array([0.0, 1.1 * CONTACT_TOL])) == 1


def test_hinge_sampling_respects_vertex_step_bound():
    env = envelope(CASES["H2_one"].scene)
    assert np.linalg.norm(np.diff(env, axis=0), axis=-1).max() <= SAMPLE_STEP + 1e-12
    assert np.linalg.norm(env, axis=-1).max() == pytest.approx(LID.reach, abs=1e-12)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("factor", [4.0, 0.25])
def test_labels_stable_under_sampling_refinement(name, factor, tables):
    case = CASES[name]
    for S in [()] + [tuple(_by_ids(case, s)) for s in case.expected]:
        base, other = evaluate(case, S), evaluate(case, S, step=SAMPLE_STEP * factor)
        assert base.F == other.F
        assert np.max(np.abs(np.subtract(base.conflict, other.conflict)), initial=0.0) <= SAMPLE_STEP


# ----------------------------------------------------------- interventions / truth

@pytest.mark.parametrize("name", sorted(CASES))
def test_exhaustive_minimal_repairs_match_manual_expectation(name, tables):
    assert minimal_repairs(tables[name]) == CASES[name].expected


@pytest.mark.parametrize("name", ["T4_five", "H4_five"])
def test_five_blocker_cases_certify_minimal_cardinality_five(name, tables):
    results = tables[name]
    assert {len(s) for s in minimal_repairs(results)} == {5}
    assert all(not r.feasible for r in results if len(r.interventions) <= 4)
    assert len(diagnostic_causes(CASES[name].scene)) == 5


def test_every_feasible_repair_beats_every_infeasible_one(tables):
    for results in tables.values():
        feasible = [r.J for r in results if r.feasible]
        infeasible = [r.J for r in results if not r.feasible]
        if feasible and infeasible:
            assert max(feasible) < min(infeasible)


def test_oracle_repair_cost_uses_B_equal_K_max_plus_one():
    assert oracle_repair_cost(0, 3, 7) == 3 and oracle_repair_cost(1, 0, 7) == 8
    with pytest.raises(ValueError):
        oracle_repair_cost(0, 8, 7)


def test_coupled_case_is_a_physical_interaction():
    case = CASES["T6_coupled"]
    G = {ids: evaluate(case, _by_ids(case, ids)) for ids in [(), ("shift_obj",), ("r_m",), ("shift_obj", "r_m")]}
    assert G[("r_m",)].G == G[()].G, "relocating m alone changes nothing: m is clear of the original lane"
    assert G[("shift_obj",)].F == 1 and G[("r_m",)].F == 1 and G[("shift_obj", "r_m")].F == 0
    beta = G[("shift_obj", "r_m")].G - G[("shift_obj",)].G - G[("r_m",)].G + G[()].G
    assert beta < -CONTACT_TOL


def test_immovable_cause_differs_from_repair_target(tables):
    case = CASES["H5_immovable_cause"]
    assert diagnostic_causes(case.scene) == ("shelf",)
    (repair,) = minimal_repairs(tables[case.name])
    assert {iv.entity_id for iv in _by_ids(case, repair)} == {"lid"}
    with pytest.raises(ValueError, match="structural"):
        apply_intervention(case.scene, Intervention("r_shelf", InterventionKind.RELOCATE, "shelf", params=(0, 1)))


def test_diagnostics_are_never_repair_candidates():
    case = CASES["H2_one"]
    diag = Intervention("d_b1", InterventionKind.REMOVE, "b1")
    with pytest.raises(ValueError, match="diagnostic"):
        Case2D("bad", "x", case.scene, case.candidates + (diag,), frozenset())
    with pytest.raises(ValueError, match="not executable"):
        RepairResult((diag,), (0.0, 0.0), 0.0, 0, 1, 1)


@pytest.mark.parametrize("name", ["T4_five", "T6_coupled", "H5_immovable_cause"])
def test_commuting_interventions_are_order_independent(name):
    case = CASES[name]
    S = case.candidates[:3]
    ref = conflict(do(case.scene, S))
    for perm in itertools.permutations(S):
        assert np.array_equal(conflict(do(case.scene, perm)), ref)


def test_enumeration_does_not_mutate_the_scene(tables):
    for name, case in CASES.items():
        fresh = {c.name: c for c in all_cases()}[name]
        assert case.scene == fresh.scene
        assert np.array_equal(conflict(case.scene), np.asarray(tables[name][0].conflict))


def test_exhaustive_enumeration_is_deterministic(tables):
    again = enumerate_subsets(CASES["H4_five"])
    assert again == tables["H4_five"]


def test_subset_bitmask_matches_x_vector():
    case = CASES["T4_five"]
    x = 0b1010011
    S = subset(case.candidates, x)
    assert [iv.intervention_id for iv in S] == [case.candidates[p].intervention_id
                                                for p in range(len(case.candidates)) if x >> p & 1]
