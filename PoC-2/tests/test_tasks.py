"""PoC-2 Stage 4 contract tests: benchmark-inspired task families (plan2.md section 20)."""

import json

import numpy as np
import pytest

from poc.envelope import ENVELOPE_STEP, HingeMotion, LinearMotion, sweep
from poc.mj_scene import GeomWorld
from poc2 import dataset as ds
from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2.oracle import Polyline, RepairTable
from poc2.types import RepairStatus

REG = tasks.regression_cases()


@pytest.fixture(scope="module")
def small():
    return {fam: ds.generate(["repairable", "negative"], ds.MAX_ATTEMPTS_FAMILY, fam) for fam in tasks.FAMILIES}


# ------------------------------------------------------------ distinct prescribed envelopes

def test_each_family_has_its_own_prescribed_envelope():
    motions = {fam: tasks.build(spec)[0].motion for fam, (spec, _, _) in REG.items()}
    ms = scenes.build(ds.sample_spec(np.random.default_rng([7, 0]), "x", "repairable"))[0].motion
    assert isinstance(ms, LinearMotion) and ms.start[2] == ms.goal[2]                # straight horizontal insertion
    si = motions["storage_insertion"].segments
    assert si[0].start[2] == si[0].goal[2] and si[1].start[:2] == si[1].goal[:2]      # transfer at height, then descend
    assert si[1].goal[2] < si[1].start[2]
    se = motions["storage_extraction"].segments
    assert se[0].start[:2] == se[0].goal[:2] and se[0].goal[2] > se[0].start[2]       # lift first
    assert se[1].goal[0] < se[1].start[0] and se[1].start[2] == se[1].goal[2]         # then withdraw outward
    ao = motions["articulated_opening"]
    assert isinstance(ao, HingeMotion) and abs(ao.axis[2]) == 1.0 and ao.axis[:2] == (0.0, 0.0)  # vertical hinge


def test_polyline_envelope_is_the_union_of_its_segments():
    scene = tasks.build(REG["storage_insertion"][0])[0]
    world = GeomWorld(scene.entities, scene.moving)
    combined = orc.envelope_sweep(world, scene.motion, scene.moving)
    parts = [sweep(world, seg, scene.moving) for seg in scene.motion.segments]
    assert np.array_equal(combined.d_min, np.minimum(*[p.d_min for p in parts]))
    assert np.array_equal(combined.d_start, parts[0].d_start) and np.array_equal(combined.d_goal, parts[-1].d_goal)


def test_polyline_repositioning_moves_every_segment_and_the_fixture():
    spec = REG["storage_extraction"][0]
    scene, _, options = tasks.build(spec)
    shift = next(o for o in options if o.option_id.startswith("shift"))
    moved = orc.apply_option(scene, shift.intervention)
    dy = shift.intervention.params[1]
    for a, b in zip(scene.motion.segments, moved.motion.segments):
        assert b.start[1] == pytest.approx(a.start[1] + dy) and b.goal[1] == pytest.approx(a.goal[1] + dy)
    assert moved.fixture[0].center[1] == pytest.approx(scene.fixture[0].center[1] + dy)


# ------------------------------------------------------------ regression mechanism cases

@pytest.mark.parametrize("family", tasks.FAMILIES)
def test_regression_case_matches_manual_expectation(family):
    spec, causes, repairs = REG[family]
    lab = ds.label(spec)
    assert lab["table"].V[0] == 1 and lab["cause"].minimal_causes == causes and lab["repair"].minimal_repairs == repairs
    assert ds.labels_of(ds.label(spec, ENVELOPE_STEP / 2)) == ds.labels_of(lab)


@pytest.mark.parametrize("family", tasks.FAMILIES)
def test_regression_table_matches_direct_evaluation(family):
    scene, _, options = tasks.build(REG[family][0])
    T = orc.repair_table(scene, options)
    for x in np.flatnonzero(T.M == 1):
        a, v = orc.direct_row(scene, options, int(x))
        assert (a.F, v) == (T.F[x], T.V[x]) and a.G == pytest.approx(T.G[x], abs=1e-12)


def test_lift_clears_a_flat_item_lying_in_the_extraction_lane():
    lab = ds.label(REG["storage_extraction"][0])
    assert "p1" not in lab["cause"].blockers and lab["cause"].blockers == {"b1"}


def test_sliding_the_appliance_makes_a_counter_item_a_blocker():
    scene, _, options = tasks.build(REG["articulated_opening"][0])
    T = orc.repair_table(scene, options)
    ids = list(T.option_ids)
    p, q = ids.index("shiftx+0.060"), ids.index("b1->r1")
    assert orc.pair_effect(T, p, q) < -0.01 and T.G[1 << q] == pytest.approx(T.G[0], abs=1e-12)


# ------------------------------------------------------------ randomized families

@pytest.mark.parametrize("family", tasks.FAMILIES)
def test_family_generation_is_deterministic_valid_and_unit_cost(family, small):
    accepted, log = small[family]
    again, log2 = ds.generate(["repairable", "negative"], ds.MAX_ATTEMPTS_FAMILY, family)
    assert [ds.to_record_json(*r) == ds.to_record_json(*s) for r, s in zip(accepted, again)] == [True, True]
    assert log == log2
    for spec, intent, attempt, lab in accepted:
        assert lab["table"].V[0] == 1 and ds.P_RANGE[0] <= lab["repair"].P <= ds.P_RANGE[1]
        assert lab["repair"].status is (RepairStatus.FEASIBLE if intent == "negative" else RepairStatus.REPAIRED)
        assert all(o.cost == 1 for o in lab["options"])
        regions = {r.region_id: r for r in spec.regions}
        halves = {o.eid: o.half for o in spec.objects}
        for o in lab["options"]:
            if o.region_id is not None:
                assert scenes.fits_in_region(o.intervention.params, halves[o.intervention.entity_id], regions[o.region_id])


@pytest.mark.parametrize("family", tasks.FAMILIES)
def test_task_spec_round_trips_through_json(family, small):
    for spec, intent, attempt, lab in small[family][0]:
        again = ds.spec_from_json(json.loads(json.dumps(ds.spec_to_json(spec))))
        assert again == spec and ds.labels_of(ds.label(again)) == ds.labels_of(lab)


def test_make_space_records_are_unchanged_by_the_generalized_dataset_code():
    spec = ds.sample_spec(np.random.default_rng([ds.SEED, 0]), "ms00", "repairable")
    assert ds.family_of(spec) == "make_space" and ds.stream("make_space", 4) == [ds.SEED, 4]
    assert ds.stream("storage_extraction", 4) == [ds.SEED, 2, 4]


def test_no_admissible_subset_means_no_recourse():
    n = 4
    T = RepairTable(("e",), ("a", "b"), np.ones(n, int), np.zeros(n, int), np.ones(n, int), np.full(n, 0.01),
                    np.zeros((n, 1)), np.array([0, 1, 1, 2]), None)
    assert orc.repair_result(T).status is RepairStatus.NO_RECOURSE_IN_CATALOGUE
