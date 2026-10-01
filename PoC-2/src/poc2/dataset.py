"""PoC-2 Stage 2: small randomized make-space structural dataset (plan2.md section 18). Oracle only.

Every randomization range is an explicit constant below. Scenes are drawn from
seeded streams default_rng([SEED, attempt]); a draw is accepted only by the
declared intent / validity / catalogue criteria (never by interaction
statistics), and every rejection is counted with its reason.

All interventions have unit cost, so K(S) = |S|; tied minimal repairs are kept.

Causal labels: under the current separable per-entity oracle (PoC-1 conflict
c_i depends only on entity i), the diagnostic minimal causal set is C* = {B0}
by construction. C* is retained as the direct-blocker / localization label;
variation of C* across scenes is not nontrivial causal-set discovery.
"""

import hashlib
import json

import numpy as np

from poc.envelope import ENVELOPE_STEP
from poc2 import oracle as orc
from poc2.scenes import MakeSpaceSpec, ObjectSpec, build, lane_half
from poc2.types import PlacementRegion, RepairStatus, SceneRecord

SEED = 7
FAMILY = "make_space"
N_REPAIRABLE, N_NEGATIVE = 30, 10       # target composition (40 scenes)
MAX_ATTEMPTS = 400                      # rejection-sampling budget for the whole dataset
P_RANGE = (5, 10)                       # candidate count
N_OBJECTS = (3, 6)                      # movable objects
N_REGIONS = (2, 4)                      # placement regions
N_BLOCKERS = (1, 3)                     # lane intruders in a repairable scene
MAX_OCCUPIED = 2                        # regions already occupied by an object
OBJ_HALF = ((0.020, 0.035), (0.020, 0.035), (0.030, 0.060))     # [m] movable box half-extents
TARGET_HALF = ((0.030, 0.045), (0.025, 0.035), (0.040, 0.060))  # [m] carried object half-extents
LANE_Y = (-0.04, 0.04)                  # [m] lateral insertion lane
LANE_SLOTS_X = (0.05, 0.12, 0.19, 0.26, 0.33)  # [m] positions of boxes beside the lane (both sides)
REGION_SPOTS = ((0.06, 0.235), (0.20, 0.235), (0.33, 0.235), (0.06, -0.235), (0.20, -0.235), (0.33, -0.235))
REGION_HALF = (0.04, 0.04, 0.07)        # [m] placement region (every movable box fits)
BLOCKER_DEPTH = (0.001, 0.020)          # [m] intrusion into the lane (1 mm = near-boundary)
DISTRACTOR_CLEARANCE = (0.005, 0.030)   # [m] lane-side distractor clearance
HARD_NEG_CLEARANCE = (0.001, 0.005)     # [m] closest box of a feasible hard negative
SHIFT_DY = 0.03                         # [m] lateral repositioning alternatives: -dy, +dy
SHIFT_COUNT_P = (0.4, 0.4, 0.2)         # probability of offering 0, 1 or 2 repositioning alternatives
CAUSAL_NOTE = ("separable per-entity oracle: diagnostic C* = {B0} by construction; kept as the direct "
               "blocker / localization label, not evidence of nontrivial causal-set discovery")


def _u(rng, lo_hi) -> float:
    return float(rng.uniform(*lo_hi))


def sample_spec(rng: np.random.Generator, scene_id: str, intent: str) -> MakeSpaceSpec:
    """One scene draw inside the declared ranges; intent is 'repairable' or 'negative'."""
    target = tuple(_u(rng, r) for r in TARGET_HALF)
    lane_y, L = _u(rng, LANE_Y), lane_half(target)
    n = int(rng.integers(N_OBJECTS[0], N_OBJECTS[1] + 1))
    spots = rng.permutation(len(REGION_SPOTS))[: int(rng.integers(N_REGIONS[0], N_REGIONS[1] + 1))]
    regions = tuple(PlacementRegion(f"r{j + 1}", (*REGION_SPOTS[s], REGION_HALF[2]), REGION_HALF)
                    for j, s in enumerate(spots))
    k = int(rng.integers(N_BLOCKERS[0], N_BLOCKERS[1] + 1)) if intent == "repairable" else 0
    k = min(k, n)
    occupied = min(int(rng.integers(0, MAX_OCCUPIED + 1)), len(regions) - 1, n - k - (intent == "negative"))
    lane_slots = rng.permutation(2 * len(LANE_SLOTS_X))[: n - occupied]
    objects, occ_of = [], {}
    for j in range(n):
        half = tuple(_u(rng, r) for r in OBJ_HALF)
        if j < occupied:                                   # standing inside a placement region
            reg = regions[j]
            center, occ_of[f"o{j + 1}"] = (reg.center[0], reg.center[1], half[2]), reg.region_id
        else:                                              # beside the lane: blocker or distractor
            slot = lane_slots[j - occupied]
            x, side = LANE_SLOTS_X[slot // 2], (1 if slot % 2 == 0 else -1)
            role = j - occupied
            depth = (_u(rng, BLOCKER_DEPTH) if role < k else
                     -_u(rng, HARD_NEG_CLEARANCE) if intent == "negative" and role == 0 else
                     -_u(rng, DISTRACTOR_CLEARANCE))
            center = (x, lane_y + side * (L + half[1] - depth), half[2])
        objects.append([f"o{j + 1}", center, half])
    shifts = {0: (), 1: (float(rng.choice([-SHIFT_DY, SHIFT_DY])),), 2: (-SHIFT_DY, SHIFT_DY)}[
        int(rng.choice(3, p=SHIFT_COUNT_P))]
    pairs = [(eid, r.region_id) for eid, _, _ in objects for r in regions if occ_of.get(eid) != r.region_id]
    first = {}
    for idx in rng.permutation(len(pairs)):                # every object gets one legal destination
        first.setdefault(pairs[idx][0], pairs[idx])
    chosen = set(first.values())
    extra = [pairs[i] for i in rng.permutation(len(pairs)) if pairs[i] not in chosen]
    target_p = int(rng.integers(P_RANGE[0], P_RANGE[1] + 1))
    chosen |= set(extra[: max(0, target_p - len(chosen) - len(shifts))])
    specs = tuple(ObjectSpec(eid, tuple(c), tuple(h), tuple(r.region_id for r in regions if (eid, r.region_id) in chosen))
                  for eid, c, h in objects)
    return MakeSpaceSpec(scene_id, target, lane_y, regions, specs, shifts)


def label(spec: MakeSpaceSpec, step: float = ENVELOPE_STEP) -> dict:
    """Exact oracle labels for one spec (table, causes, minimal repairs)."""
    scene, regions, options = build(spec)
    table = orc.repair_table(scene, options, step)
    cause, repair = orc.causes(table.original), orc.repair_result(table)
    record = SceneRecord(spec.scene_id, SEED, FAMILY, regions, options, orc.choice_groups(options), cause, repair)
    return {"table": table, "cause": cause, "repair": repair, "record": record, "options": options}


def rejection_reason(intent: str, lab: dict) -> str | None:
    t, repair = lab["table"], lab["repair"]
    if t.V[0] != 1:
        return "original_static_invalid"
    if not P_RANGE[0] <= repair.P <= P_RANGE[1]:
        return "P_out_of_range"
    if intent == "negative":
        return None if t.F[0] == 0 else "negative_not_feasible"
    if t.F[0] != 1:
        return "repairable_not_infeasible"
    return None if repair.status is RepairStatus.REPAIRED else "no_recourse_in_catalogue"


def intents() -> list[str]:
    """Fixed composition: every fourth slot is a feasible hard negative (30 + 10)."""
    return ["negative" if j % 4 == 3 else "repairable" for j in range(N_REPAIRABLE + N_NEGATIVE)]


def generate(slots: list[str] | None = None, max_attempts: int = MAX_ATTEMPTS):
    """Sequential seeded draws; returns (accepted [(spec, intent, attempt, labels)], rejection log)."""
    slots, accepted, rejected, attempt = slots or intents(), [], {}, 0
    for j, intent in enumerate(slots):
        while True:
            if attempt >= max_attempts:
                raise RuntimeError(f"composition not met within {max_attempts} attempts: {rejected}")
            spec = sample_spec(np.random.default_rng([SEED, attempt]), f"ms{j:02d}", intent)
            attempt += 1
            lab = label(spec)
            reason = rejection_reason(intent, lab)
            if reason is None:
                accepted.append((spec, intent, attempt - 1, lab))
                break
            rejected[reason] = rejected.get(reason, 0) + 1
    return accepted, {"attempts": attempt, "max_attempts": max_attempts, "reasons": rejected}


def spec_to_json(spec: MakeSpaceSpec) -> dict:
    return {"scene_id": spec.scene_id, "target_half": list(spec.target_half), "lane_y": spec.lane_y,
            "regions": [{"id": r.region_id, "center": list(r.center), "half": list(r.half)} for r in spec.regions],
            "objects": [{"id": o.eid, "center": list(o.center), "half": list(o.half), "destinations": list(o.destinations)}
                        for o in spec.objects],
            "shifts": list(spec.shifts)}


def spec_from_json(d: dict) -> MakeSpaceSpec:
    return MakeSpaceSpec(d["scene_id"], tuple(d["target_half"]), d["lane_y"],
                         tuple(PlacementRegion(r["id"], tuple(r["center"]), tuple(r["half"])) for r in d["regions"]),
                         tuple(ObjectSpec(o["id"], tuple(o["center"]), tuple(o["half"]), tuple(o["destinations"]))
                               for o in d["objects"]),
                         tuple(d["shifts"]))


def _sets(fs) -> list[list[str]]:
    return sorted(sorted(s) for s in fs)


def to_record_json(spec: MakeSpaceSpec, intent: str, attempt: int, lab: dict) -> dict:
    a, cause, repair = lab["table"].original, lab["cause"], lab["repair"]
    return {
        "scene_id": spec.scene_id, "seed": SEED, "stream": [SEED, attempt], "family": FAMILY, "intent": intent,
        "spec": spec_to_json(spec),
        "candidate_interventions": [{"id": o.option_id, "entity": o.intervention.entity_id,
                                     "kind": o.intervention.kind.value, "region": o.region_id,
                                     "params": list(o.intervention.params), "cost": o.cost} for o in lab["options"]],
        "candidate_groups": {g.group_id: list(g.option_ids) for g in lab["record"].groups},
        "original": {"F": a.F, "G": a.G, "c": dict(zip(a.ids, a.c))},
        "B0": sorted(cause.blockers), "C_star": _sets(cause.minimal_causes), "S_star": _sets(repair.minimal_repairs),
        "status": repair.status.value, "P": repair.P, "n_causal_sets": len(cause.minimal_causes),
        "causal_set_size": len(cause.blockers), "repair_size": repair.cost, "n_tied_repairs": repair.n_tied,
        "n_choice_invalid": repair.n_choice_invalid, "n_static_invalid": repair.n_static_invalid,
    }


def digest(lines: list[dict]) -> str:
    return hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in lines).encode()).hexdigest()


def labels_of(lab: dict) -> tuple:
    """(F0, C*, S*) used by the refinement and reconstruction checks."""
    return lab["table"].F[0], lab["cause"].minimal_causes, lab["repair"].minimal_repairs


STAGE2_DIGEST = "a523986584bd35f0abaf2cf0262c4d83b0895e714a00f39b5d6d3ad4eaf32bb5"  # approved fixed dataset


def load_dataset(path) -> list[dict]:
    """The fixed Stage-2 records, verified against the approved digest (never regenerated here)."""
    lines = [json.loads(s) for s in open(path).read().splitlines()]
    if digest(lines) != STAGE2_DIGEST:
        raise ValueError(f"{path} is not the approved Stage-2 dataset")
    return lines
