"""PoC-2 Stage 2 contract tests: randomized make-space dataset (plan2.md section 18)."""

import numpy as np
import pytest

from poc.envelope import ENVELOPE_STEP
from poc.types import Intervention, InterventionKind
from poc2 import dataset as ds
from poc2 import oracle as orc
from poc2 import scenes
from poc2.types import RepairOption

SLOTS = ["repairable", "negative", "repairable"]


@pytest.fixture(scope="module")
def small():
    return ds.generate(SLOTS)


# ------------------------------------------------------------ region-fit invariant

def test_relocation_must_fit_inside_its_region():
    big = scenes.shelf_box("big", (0.1, 0.0, 0.06), (0.05, 0.03, 0.06))  # wider than any region
    reg = scenes.region("r1", 0.06, 0.22)
    with pytest.raises(ValueError, match="does not fit"):
        scenes.to_region(big, reg)
    tall = scenes.shelf_box("tall", (0.1, 0.0, 0.08), (0.03, 0.03, 0.08))  # taller than the region
    with pytest.raises(ValueError, match="does not fit"):
        scenes.to_region(tall, reg)


def test_case_validation_rejects_a_non_fitting_relocation():
    case = scenes.distractors()
    bad = RepairOption(Intervention("b1->r1x", InterventionKind.RELOCATE, "b1", params=(0.06, 0.30, 0.05)), "r1")
    with pytest.raises(ValueError, match="does not fit"):
        scenes.MakeSpaceCase("bad", "x", case.scene, case.regions, case.options + (bad,), frozenset(), frozenset())


def test_every_generated_relocation_fits(small):
    for spec, _, _, lab in small[0]:
        regions = {r.region_id: r for r in spec.regions}
        halves = {o.eid: o.half for o in spec.objects}
        for o in lab["options"]:
            if o.region_id is not None:
                assert scenes.fits_in_region(o.intervention.params, halves[o.intervention.entity_id], regions[o.region_id])


# ------------------------------------------------------------ generator contract

def test_sampling_is_seeded_and_inside_declared_ranges():
    a = ds.sample_spec(np.random.default_rng([ds.SEED, 3]), "x", "repairable")
    assert a == ds.sample_spec(np.random.default_rng([ds.SEED, 3]), "x", "repairable")
    assert ds.N_OBJECTS[0] <= len(a.objects) <= ds.N_OBJECTS[1] and ds.N_REGIONS[0] <= len(a.regions) <= ds.N_REGIONS[1]
    for o in a.objects:
        assert all(lo <= h <= hi for h, (lo, hi) in zip(o.half, ds.OBJ_HALF))
    assert all(lo <= h <= hi for h, (lo, hi) in zip(a.target_half, ds.TARGET_HALF))


def test_generation_is_deterministic_and_meets_the_requested_composition(small):
    accepted, log = small
    again, log2 = ds.generate(SLOTS)
    lines = [ds.to_record_json(*row) for row in accepted]
    assert ds.digest(lines) == ds.digest([ds.to_record_json(*row) for row in again]) and log == log2
    assert [i for _, i, _, _ in accepted] == SLOTS and log["attempts"] >= len(SLOTS)
    for spec, intent, _, lab in accepted:
        assert lab["table"].V[0] == 1 and ds.P_RANGE[0] <= lab["repair"].P <= ds.P_RANGE[1]
        assert lab["repair"].status.value == ("FEASIBLE" if intent == "negative" else "REPAIRED")


def test_rejection_budget_is_enforced():
    with pytest.raises(RuntimeError, match="composition not met"):
        ds.generate(SLOTS, max_attempts=1)


def test_a_feasible_scene_is_rejected_for_the_wrong_intent(small):
    spec, _, _, lab = small[0][1]  # an accepted hard negative
    assert ds.rejection_reason("repairable", lab) == "repairable_not_infeasible"


def test_spec_round_trip_rebuilds_identical_labels(small):
    for spec, intent, attempt, lab in small[0]:
        again = ds.spec_from_json(ds.spec_to_json(spec))
        assert again == spec and ds.labels_of(ds.label(again)) == ds.labels_of(lab)


# ------------------------------------------------------------ labels

def test_unit_cost_makes_K_the_subset_size(small):
    for _, _, _, lab in small[0]:
        T = lab["table"]
        assert all(o.cost == 1 for o in lab["options"])
        assert all(T.K[x] == bin(x).count("1") for x in range(len(T.K)))


def test_causal_label_is_the_direct_blocker_set(small):
    for _, _, _, lab in small[0]:
        cause = lab["cause"]
        assert cause.minimal_causes == ({cause.blockers} if cause.blockers else set())


def test_hard_negatives_have_a_near_boundary_object(small):
    for spec, intent, _, lab in small[0]:
        if intent == "negative":
            d = dict(zip(lab["table"].original.ids, lab["table"].original.d))
            closest = min(d[o.eid] for o in spec.objects)
            assert ds.HARD_NEG_CLEARANCE[0] - 1e-9 <= closest <= ds.HARD_NEG_CLEARANCE[1] + 1e-9


def test_table_matches_direct_evaluation_on_a_generated_scene(small):
    spec = small[0][0][0]
    scene, _, options = scenes.build(spec)
    T = orc.repair_table(scene, options)
    for x in np.flatnonzero(T.M == 1):
        a, v = orc.direct_row(scene, options, int(x))
        assert (a.F, v) == (T.F[x], T.V[x]) and a.G == pytest.approx(T.G[x], abs=1e-12)


def test_finer_sampling_keeps_F_causes_and_repairs(small):
    for spec, _, _, lab in small[0]:
        assert ds.labels_of(ds.label(spec, ENVELOPE_STEP / 2)) == ds.labels_of(lab)
