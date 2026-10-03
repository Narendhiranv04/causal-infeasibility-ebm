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

class Proposer:
    """v0.1 proposer: canonical layout families on the named slot topology, with the assets
    re-drawn within each category from the accepted pool and 0-2 distractors on free slots
    compatible with their category. No free offsets (slot-local jitter is a later option)."""

    DISTRACTOR_SLOTS = {"cup": ["U1", "U2", "U4", "B4", "B8"], "can": ["B4", "B8"], "mug": ["B4", "B8"]}

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.cats = {c: by_cat(c) for c in ("mug", "cup", "bowl", "plate", "pan", "bottle", "can", "box")}

    def propose(self, variant: str) -> SceneSpec:
        from .canonical import ALL

        base = ALL.get(variant, ALL["V0"])()
        assets = selected_assets()
        objs = []
        for o in base.objects:
            cat = assets[o.asset_id].category
            pool = self.cats.get(cat, [o.asset_id])
            objs.append(O(o.key, str(self.rng.choice(pool)), o.slot, o.orient))
        used = {o.slot for o in objs}
        ivs = list(base.interventions)
        for k in range(int(self.rng.integers(0, 3))):
            cat = str(self.rng.choice(list(self.DISTRACTOR_SLOTS)))
            free = [sl for sl in self.DISTRACTOR_SLOTS[cat] if sl not in used]
            if not free or len(objs) >= 9:
                break
            sl = str(self.rng.choice(free))
            used.add(sl)
            objs.append(O(f"{cat}_d{k}", str(self.rng.choice(self.cats[cat])), sl,
                          "handle_back" if cat == "mug" else "upright"))
        return SceneSpec("", objs, ivs, variant_requested=variant, target=base.target, articulation=base.articulation)


def signature(spec: SceneSpec) -> str:
    d = spec.to_dict()
    key = json.dumps({"objects": sorted((o["asset_id"], o["slot"], o["orient"], tuple(np.round(o["dxy"], 4)))
                                        for o in d["objects"]),
                      "ivs": sorted((i["obj"], i["slot"], i["orient"], tuple(np.round(i["dxy"], 4)))
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
