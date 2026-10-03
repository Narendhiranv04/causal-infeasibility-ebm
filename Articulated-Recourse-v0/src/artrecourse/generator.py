"""Randomized scene generation (quick v0).

Sampling only proposes scenes. Acceptance is STRUCTURAL: every proposal is solved by the
oracle and accepted for a variant only if variants.check_variant passes on the resulting
geometry, the coarse/fine check agrees and no consulted entry is ambiguous. Proposals are
biased towards each variant with "layout families" (the canonical building blocks with
randomised assets, jitter, slots and distractors), but the label always comes from the
oracle. Every rejection reason is counted.

Split hygiene: a scene signature = (sorted asset ids, rounded initial poses, sorted
candidate catalogue). Signatures are unique across all splits. The split is a pure function
of the seed namespace (train 101 / val 211 / test 307); asset-disjoint / unseen-graph splits
can later be produced by filtering on `asset_ids` and `dependency_signature` in the records.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .assets import selected_assets
from .interventions import InterventionSpec as I
from .interventions import ObjSpec as O
from .interventions import SceneSpec, build_problem
from .oracle import solve
from .quality import coarse_fine_check, quality_report
from .records import build_record, write_record
from .variants import check_variant, repair_edges

RACK_BAYS = ["upper_left_front", "upper_left_back", "upper_right_front", "upper_right_back"]
COUNTER_R = ["counter_buffer_left", "counter_buffer_center", "counter_buffer_right", "counter_buffer_far_right"]
COUNTER_L = ["counter_left_near", "counter_left_mid", "counter_left_far"]


def by_cat(cat):
    return sorted(a for a, x in selected_assets().items() if x.category == cat)


def _j(rng, s=0.02):
    return (float(rng.uniform(-s, s)), float(rng.uniform(-s, s)))


class Proposer:
    """Variant-biased layout families. Each returns a SceneSpec (or None)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.cats = {c: by_cat(c) for c in ("mug", "cup", "bowl", "plate", "pan", "bottle", "can", "box")}

    def pick(self, cat):
        return str(self.rng.choice(self.cats[cat]))

    # -- building blocks -------------------------------------------------------------
    def pan_out(self, objs, ivs, key="pan"):
        objs.append(O(key, self.pick("pan") if self.rng.random() < 0.3 else "pan/pan_2", "upper_center_back", -90,
                      (float(self.rng.uniform(-0.01, 0.01)), -0.005)))
        ivs += [I(key, "upper_tines_handle_left", 180), I(key, "upper_tines_handle_right", 0)]

    def tall_bottle(self, objs, ivs, key, bay, dests):
        objs.append(O(key, self.pick("bottle"), bay, 0, _j(self.rng, 0.008)))
        ivs += [I(key, d, 0) for d in dests]

    def corridor_box(self, objs, ivs, key, side="right", dests=None):
        if side == "right":
            objs.append(O(key, self.pick("box"), "counter_buffer_right", 0,
                          (-0.07 + float(self.rng.uniform(-0.02, 0.02)), -0.055 + float(self.rng.uniform(-0.01, 0.01)))))
        else:
            objs.append(O(key, self.pick("box"), "counter_left_near", 0,
                          (-0.10 + float(self.rng.uniform(-0.02, 0.02)), -0.075 + float(self.rng.uniform(-0.01, 0.01)))))
        ivs += [I(key, d, 0) for d in (dests or (["counter_buffer_left", "counter_left_mid"] if side == "right"
                                                 else ["counter_left_near"]))]

    def distractors(self, objs, ivs, n, free_slots):
        cats = ["mug", "cup", "bowl", "can", "plate"]
        for k in range(n):
            if not free_slots:
                return
            slot = free_slots.pop(int(self.rng.integers(len(free_slots))))
            cat = str(self.rng.choice(cats))
            yaw = 90 if cat == "mug" else 0
            objs.append(O(f"{cat}{k}", self.pick(cat), slot, yaw, _j(self.rng, 0.01)))
            if free_slots:
                ivs.append(I(f"{cat}{k}", str(self.rng.choice(free_slots)), yaw))

    # -- families -----------------------------------------------------------------------
    def propose(self, variant: str) -> SceneSpec:
        rng = self.rng
        objs, ivs = [], []
        free = COUNTER_L.copy()
        if variant == "V0":
            self.pan_out(objs, ivs)
            free += COUNTER_R
        elif variant == "V1":
            self.pan_out(objs, ivs)
            self.tall_bottle(objs, ivs, "bottle", "upper_right_front", ["counter_buffer_center", "counter_buffer_right"])
            free += ["counter_buffer_far_right"]
        elif variant in ("V2", "V6"):
            self.tall_bottle(objs, ivs, "bottle", "upper_right_back", ["counter_buffer_center", "counter_buffer_right"])
            objs.append(O("occupant", self.pick(str(rng.choice(["bowl", "can", "plate"]))), "counter_buffer_center", 0,
                          _j(rng, 0.01)))
            ivs.append(I("occupant", "counter_buffer_left", 0))
            if variant == "V6":
                self.corridor_box(objs, ivs, "box", dests=["counter_buffer_far_right"])
                objs.append(O("bottle_c", self.pick("bottle"), "counter_buffer_right", 0, (-0.03, 0.04)))
                ivs.append(I("bottle_c", "counter_buffer_left", 0, (-0.04, 0.03)))
        elif variant == "V3":
            self.tall_bottle(objs, ivs, "bottle", "upper_right_back", ["counter_buffer_far_right", "counter_buffer_right"])
            self.corridor_box(objs, ivs, "box")
        elif variant == "V4":
            self.tall_bottle(objs, ivs, "bottle_r", "upper_right_back", ["counter_buffer_far_right", "counter_buffer_right"])
            self.corridor_box(objs, ivs, "box_r", "right", ["counter_buffer_left"])
            self.tall_bottle(objs, ivs, "bottle_l", "upper_left_back", ["counter_left_far", "counter_left_mid"])
            self.corridor_box(objs, ivs, "box_l", "left")
            free = []
        elif variant == "V5":
            self.tall_bottle(objs, ivs, "bottle_a", "upper_right_back", ["counter_buffer_far_right", "counter_buffer_right"])
            self.corridor_box(objs, ivs, "box", "right", ["counter_buffer_center", "counter_left_mid"])
            objs.append(O("bottle_c", self.pick("bottle"), "counter_buffer_right", 0,
                          (-0.07 + float(rng.uniform(-0.01, 0.01)), 0.04)))
            ivs.append(I("bottle_c", "counter_buffer_left", 0))
        elif variant == "V7":
            self.pan_out(objs, ivs)
            self.tall_bottle(objs, ivs, "bottle_r", "upper_right_back", ["counter_buffer_far_right", "counter_buffer_right"])
            self.tall_bottle(objs, ivs, "bottle_l", "upper_left_back", ["counter_buffer_right", "counter_left_mid"])
            self.corridor_box(objs, ivs, "box")
            free = ["counter_left_near", "counter_left_far", "counter_buffer_far_right"]
        elif variant == "C0":   # hard negative: something close to the sweep, but feasible
            objs.append(O("pan", "pan/pan_2", "upper_tines_handle_left", 180, _j(rng, 0.005)))
            objs.append(O("cup", self.pick("cup"), str(rng.choice(["upper_left_front", "upper_left_back"])), 0, _j(rng, 0.01)))
            ivs.append(I("cup", "counter_buffer_center", 0))
            free += COUNTER_R
        elif variant == "C1":   # no recourse in catalogue: blocker whose only destinations are blocked/invalid
            self.tall_bottle(objs, ivs, "bottle", "upper_right_back", ["counter_buffer_center"])
            objs.append(O("occupant", self.pick("bowl"), "counter_buffer_center", 0, _j(rng, 0.01)))
            ivs.append(I("occupant", "counter_buffer_right", 0))
            objs.append(O("occupant2", self.pick("can"), "counter_buffer_right", 0, _j(rng, 0.01)))
        n_extra = int(rng.integers(max(0, 6 - len(objs)), max(1, 10 - len(objs))))
        if variant == "V7":
            n_extra = max(n_extra, 7 - len(objs))
        self.distractors(objs, ivs, min(n_extra, 9 - len(objs)), [f for f in free if f not in {o.slot for o in objs}])
        return SceneSpec("", objs, ivs, variant_requested=variant)


def signature(spec: SceneSpec) -> str:
    d = spec.to_dict()
    key = json.dumps({"objects": sorted((o["asset_id"], o["slot"], o["yaw_deg"], tuple(np.round(o["dxy"], 4)))
                                        for o in d["objects"]),
                      "ivs": sorted((i["obj"], i["slot"], i["yaw_deg"], tuple(np.round(i["dxy"], 4)))
                                    for i in d["interventions"])}, sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def generate(out: Path, seeds: dict, quota: dict, max_attempts_per_accepted=400, log=print):
    out.mkdir(parents=True, exist_ok=True)
    seen, report = set(), {"accepted": Counter(), "attempted": Counter(), "rejections": Counter(), "records": []}
    t0 = time.time()
    for split, seed in seeds.items():
        rng = np.random.default_rng(seed)
        prop = Proposer(rng)
        for variant, n_needed in quota[split].items():
            got = attempts = 0
            while got < n_needed:
                attempts += 1
                report["attempted"][(split, variant)] += 1
                if attempts > max_attempts_per_accepted * max(1, n_needed):
                    raise RuntimeError(f"STOP: cannot reach quota {split}/{variant}: {got}/{n_needed}")
                spec = prop.propose(variant)
                sig = signature(spec)
                if sig in seen:
                    report["rejections"]["duplicate_signature"] += 1
                    continue
                spec.scene_id = f"{split}_{variant}_{got:04d}"
                spec.split, spec.seed = split, seed
                try:
                    pb = build_problem(spec)
                except Exception as e:  # noqa: BLE001
                    report["rejections"][f"build_error:{type(e).__name__}"] += 1
                    continue
                if pb.problems:
                    report["rejections"]["static_quality"] += 1
                    continue
                res = solve(pb)
                ok, why = check_variant(variant, res, {"objects": spec.objects, "candidate_interventions": pb.ivs})
                if not ok:
                    for w in why:
                        report["rejections"][f"{variant}:{w.split(' ')[0]}"] += 1
                    continue
                if res["consulted_ambiguities"]:
                    report["rejections"]["numerical_tolerance_dependency"] += 1
                    continue
                fine = coarse_fine_check(pb, res)
                if not fine["agree"]:
                    report["rejections"]["coarse_fine_disagreement"] += 1
                    continue
                q = quality_report(pb, res, fine)
                rec, ai, al = build_record(pb, res, variant, True, q)
                rec["signature"] = sig
                rec["dependency_signature"] = sorted({(e["cause"]["type"]) for e in repair_edges(res)})
                write_record(out / split / spec.scene_id, rec, ai, al, name="scene.json")
                seen.add(sig)
                got += 1
                report["accepted"][(split, variant)] += 1
                report["records"].append({"scene_id": spec.scene_id, "variant": variant, "split": split,
                                          "L": res["optimal_cost"], "n_edges": len(repair_edges(res)),
                                          "assets": sorted(o.asset_id for o in spec.objects)})
                log(f"{spec.scene_id} accepted ({attempts} attempts, {time.time() - t0:.0f}s)")
    report["runtime_s"] = time.time() - t0
    return report
