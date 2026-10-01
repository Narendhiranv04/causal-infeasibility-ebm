"""PoC-2 Stage 1 contract tests: deterministic make-space recourse oracle (plan2.md section 17)."""

import math
from dataclasses import replace

import numpy as np
import pytest

from poc import cases as cs
from poc.envelope import ENVELOPE_STEP
from poc.types import InterventionKind
from poc2 import oracle as orc
from poc2 import scenes
from poc2.types import RepairStatus, SceneRecord

CASES = {c.name: c for c in scenes.stage1_cases()}


@pytest.fixture(scope="module")
def tables():
    return {n: orc.repair_table(c.scene, c.options) for n, c in CASES.items()}


def _idx(case, option_id: str) -> int:
    return [o.option_id for o in case.options].index(option_id)


def _x(case, *option_ids) -> int:
    return sum(1 << _idx(case, i) for i in option_ids)


# ------------------------------------------------------------ labels

@pytest.mark.parametrize("name", sorted(CASES))
def test_causal_sets_match_manual_expectation(name, tables):
    cause = orc.causes(tables[name].original)
    assert cause.minimal_causes == CASES[name].expected_causes
    assert cause.minimal_causes == {cause.blockers}, "entity-separable conflict: the minimal cause set is B0"


@pytest.mark.parametrize("name", sorted(CASES))
def test_minimal_repairs_match_manual_expectation(name, tables):
    repair = orc.repair_result(tables[name])
    assert repair.status is RepairStatus.REPAIRED and repair.minimal_repairs == CASES[name].expected_repairs
    options = {o.option_id: o for o in CASES[name].options}
    assert all(not options[i].intervention.is_diagnostic for S in repair.minimal_repairs for i in S)


def test_labels_are_deterministic(tables):
    for name, case in CASES.items():
        again = orc.repair_table(case.scene, case.options)
        assert orc.repair_result(again) == orc.repair_result(tables[name])
        assert orc.causes(again.original) == orc.causes(tables[name].original)


@pytest.mark.parametrize("name", sorted(CASES))
def test_finer_sampling_keeps_F_causes_and_repairs(name, tables):
    fine = orc.repair_table(CASES[name].scene, CASES[name].options, ENVELOPE_STEP / 2)
    assert fine.original.F == tables[name].original.F
    assert orc.causes(fine.original) == orc.causes(tables[name].original)
    assert orc.repair_result(fine).minimal_repairs == orc.repair_result(tables[name]).minimal_repairs


@pytest.mark.parametrize("name", sorted(CASES))
def test_table_matches_direct_intervention_evaluation(name, tables):
    case, T = CASES[name], tables[name]
    for x in np.flatnonzero(T.M == 1):
        a, v = orc.direct_row(case.scene, case.options, int(x))
        assert (a.F, v) == (T.F[x], T.V[x]) and a.G == pytest.approx(T.G[x], abs=1e-12)


# ------------------------------------------------------------ M, V, F are distinct

def test_contradictory_same_object_choices_have_M0_and_no_geometry(tables):
    case, T = CASES["M1_independent"], tables["M1_independent"]
    x = _x(case, "b1->r1", "b1->r2")
    assert T.M[x] == 0 and T.F[x] == -1 and T.V[x] == -1 and math.isnan(T.G[x])
    with pytest.raises(ValueError, match="contradictory"):
        orc.direct_row(case.scene, case.options, x)


def test_option_membership_in_choice_groups_is_unambiguous():
    for case in CASES.values():
        members = [i for g in orc.choice_groups(case.options) for i in g.option_ids]
        assert len(members) == len(set(members))
    case = CASES["M1_independent"]
    groups = orc.choice_groups(case.options)
    overlapping = groups + (replace(groups[0], group_id="extra"),)
    cause = orc.causes(orc.repair_table(case.scene, case.options).original)
    repair = orc.repair_result(orc.repair_table(case.scene, case.options))
    with pytest.raises(ValueError, match="at most one"):
        SceneRecord(case.name, 7, "make_space", case.regions, case.options, overlapping, cause, repair)
    assert SceneRecord(case.name, 7, "make_space", case.regions, case.options, groups, cause, repair)


def test_target_action_feasible_while_repaired_scene_invalid(tables):
    case, T = CASES["M2_placement_competition"], tables["M2_placement_competition"]
    x = _x(case, "b1->rA", "b2->rA")  # cheapest set, but both boxes end up in the same region
    assert (T.M[x], T.V[x], T.F[x], T.K[x]) == (1, 0, 0, 2)
    assert frozenset({"b1->rA", "b2->rA"}) not in orc.repair_result(T).minimal_repairs


def test_occupied_region_makes_a_single_relocation_invalid(tables):
    case, T = CASES["M5_distractors"], tables["M5_distractors"]
    x = _x(case, "b1->r2")  # r2 is already occupied by d1
    assert (T.M[x], T.V[x], T.F[x]) == (1, 0, 0)


# ------------------------------------------------------------ mechanisms

def test_placement_competition_is_a_validity_edge_not_a_geometric_pair(tables):
    case, T = CASES["M2_placement_competition"], tables["M2_placement_competition"]
    p, q = _idx(case, "b1->rA"), _idx(case, "b2->rA")
    assert orc.validity_edge(T, p, q)
    assert orc.pair_effect(T, p, q) == pytest.approx(0.0, abs=1e-12)


def test_envelope_changing_coupling_is_a_true_geometric_pair(tables):
    case, T = CASES["M4_envelope_coupling"], tables["M4_envelope_coupling"]
    p, q = _idx(case, "shift"), _idx(case, "nb->r1")
    assert orc.pair_effect(T, p, q) < -0.01
    assert T.G[1 << q] == pytest.approx(T.G[0], abs=1e-12), "relocating nb alone changes nothing"
    assert not orc.validity_edge(T, p, q)


def test_substitutable_repairs_overlap_positively(tables):
    case, T = CASES["M3_substitutable"], tables["M3_substitutable"]
    assert orc.pair_effect(T, _idx(case, "b1->r1"), _idx(case, "shift")) > 0.005


def test_pair_effect_is_undefined_for_choice_constrained_pairs(tables):
    case, T = CASES["M4_envelope_coupling"], tables["M4_envelope_coupling"]
    assert orc.pair_effect(T, _idx(case, "nb->r1"), _idx(case, "nb->r2")) is None


def test_distractor_candidates_are_never_in_the_minimal_repair():
    for case in CASES.values():
        distractors = {o.option_id for o in case.options if o.intervention.entity_id.startswith("d")}
        assert distractors and not any(distractors & S for S in case.expected_repairs)


def test_structural_cause_is_repaired_on_a_different_entity(tables):
    case, T = CASES["M6_structural_cause"], tables["M6_structural_cause"]
    cause, repair = orc.causes(T.original), orc.repair_result(T)
    moved = {o.intervention.entity_id for o in case.options if any(o.option_id in S for S in repair.minimal_repairs)}
    assert cause.minimal_causes == {frozenset({"jamb"})} and moved == {"obj"}
    assert not any(o.intervention.entity_id == "jamb" for o in case.options), "structure is never a repair target"


def test_no_recourse_inside_the_catalogue_is_reported_not_invented():
    case = CASES["M6_structural_cause"]
    options = tuple(o for o in case.options if o.intervention.kind is not InterventionKind.SHIFT_TARGET)
    repair = orc.repair_result(orc.repair_table(case.scene, options))
    assert repair.status is RepairStatus.NO_RECOURSE_IN_CATALOGUE and not repair.minimal_repairs


def test_feasible_scene_needs_no_repair():
    case = CASES["M1_independent"]
    clear = cs.insertion(cs.side_box("d1", cs.SLOTS[4], +1, -0.006), cs.side_box("d2", cs.SLOTS[2], -1, -0.02))
    options = tuple(o for o in case.options if o.intervention.entity_id == "d1")
    T = orc.repair_table(clear, options)
    repair = orc.repair_result(T)
    assert orc.causes(T.original).F0 == 0 and repair.status is RepairStatus.FEASIBLE
    assert repair.minimal_repairs == {frozenset()} and repair.cost == 0
