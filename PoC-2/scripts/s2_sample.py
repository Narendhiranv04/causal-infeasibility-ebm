"""PoC-2 Stage 2: generate the fixed 40-scene randomized make-space dataset (plan2.md section 18).

Run from the repository root:

    PYTHONPATH=PoC-1/src:PoC-2/src python PoC-2/scripts/s2_sample.py

Writes exactly PoC-2/out/s2/scenes.jsonl and PoC-2/out/s2/summary.json. Oracle only:
no pairwise-hypothesis analysis in this stage.
"""

import json
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from poc.envelope import ENVELOPE_STEP
from poc.oracle import CONTACT_TOL_3D
from poc2 import dataset as ds
from poc2 import oracle as orc
from poc2.scenes import build

OUT = Path(__file__).resolve().parents[1] / "out" / "s2"
N_DIRECT = 10                 # first scenes (dataset order) checked subset-by-subset against direct do(S)
DIRECT_TOL = 1e-12            # [m] |G_table - G_direct|


def direct_check(spec) -> dict:
    scene, _, options = build(spec)
    table = orc.repair_table(scene, options)
    n, mismatches, worst = 0, 0, 0.0
    for x in np.flatnonzero(table.M == 1):
        a, v = orc.direct_row(scene, options, int(x))
        n += 1
        mismatches += (a.F != table.F[x]) or (v != table.V[x])
        worst = max(worst, abs(a.G - table.G[x]))
    return {"scene_id": spec.scene_id, "P": len(options), "n_subsets": n, "label_mismatches": int(mismatches),
            "max_abs_dG": worst}


def dist(values) -> dict:
    return {str(k): v for k, v in sorted(Counter(values).items())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    accepted, rejection = ds.generate()
    runtime = time.perf_counter() - t0
    lines = [ds.to_record_json(spec, intent, attempt, lab) for spec, intent, attempt, lab in accepted]
    (OUT / "scenes.jsonl").write_text("".join(json.dumps(r, default=lambda v: v.item()) + "\n" for r in lines))
    lines = [json.loads(s) for s in (OUT / "scenes.jsonl").read_text().splitlines()]  # exactly what was stored

    again, _ = ds.generate()
    regenerated = [json.loads(json.dumps(ds.to_record_json(s, i, a, lab), default=lambda v: v.item()))
                   for s, i, a, lab in again]
    rebuilt = [json.loads(json.dumps(ds.to_record_json(ds.spec_from_json(r["spec"]), r["intent"], r["stream"][1],
                                                       ds.label(ds.spec_from_json(r["spec"]))),
                                     default=lambda v: v.item())) for r in lines]
    fine = [ds.labels_of(ds.label(spec, ENVELOPE_STEP / 2)) == ds.labels_of(lab) for spec, _, _, lab in accepted]
    direct = [direct_check(spec) for spec, _, _, _ in accepted[:N_DIRECT]]

    rep = [r for r in lines if r["intent"] == "repairable"]
    neg = [r for r in lines if r["intent"] == "negative"]
    checks = {
        "n_scenes": len(lines), "n_repairable": len(rep), "n_hard_negatives": len(neg),
        "reproducible_regeneration": ds.digest(regenerated) == ds.digest(lines),
        "reconstructs_from_records": all(a == b for a, b in zip(rebuilt, lines)),
        "original_scenes_statically_valid": all(lab["table"].V[0] == 1 for *_, lab in accepted),
        "hard_negatives_feasible": all(r["original"]["F"] == 0 and r["status"] == "FEASIBLE" for r in neg),
        "repairables_have_valid_repair": all(r["original"]["F"] == 1 and r["status"] == "REPAIRED" for r in rep),
        "base_vs_fine_F_Cstar_Sstar_stable": all(fine), "n_base_fine_checked": len(fine),
        "direct_vs_table_agree": all(d["label_mismatches"] == 0 and d["max_abs_dG"] <= DIRECT_TOL for d in direct),
        "C_star_equals_B0_everywhere": all(r["C_star"] == ([r["B0"]] if r["B0"] else []) for r in lines),
        "all_unit_cost": all(o["cost"] == 1 for r in lines for o in r["candidate_interventions"]),
        "no_manual_edits": "scenes accepted only by intent/validity/catalogue criteria; ranges fixed in dataset.py",
    }
    summary = {
        "stage": 2, "seed": ds.SEED, "family": ds.FAMILY, "mujoco_version": mujoco.__version__,
        "causal_note": ds.CAUSAL_NOTE, "cost_note": "unit repair cost for every intervention: K(S) = |S|; ties kept",
        "constants": {k: getattr(ds, k) for k in ("N_REPAIRABLE", "N_NEGATIVE", "MAX_ATTEMPTS", "P_RANGE", "N_OBJECTS",
                                                  "N_REGIONS", "N_BLOCKERS", "MAX_OCCUPIED", "OBJ_HALF", "TARGET_HALF",
                                                  "LANE_Y", "LANE_SLOTS_X", "REGION_SPOTS", "REGION_HALF",
                                                  "BLOCKER_DEPTH", "DISTRACTOR_CLEARANCE", "HARD_NEG_CLEARANCE",
                                                  "SHIFT_DY", "SHIFT_COUNT_P")}
        | {"CONTACT_TOL_3D": CONTACT_TOL_3D, "ENVELOPE_STEP": ENVELOPE_STEP},
        "rejection_sampling": rejection, "checks": checks, "direct_vs_table": direct,
        "distributions": {
            "status": dist(r["status"] for r in lines), "P": dist(r["P"] for r in lines),
            "causal_set_size_B0": dist(r["causal_set_size"] for r in lines),
            "n_causal_sets": dist(r["n_causal_sets"] for r in lines),
            "repair_size_S_star": dist(r["repair_size"] for r in lines),
            "n_tied_repairs": dist(r["n_tied_repairs"] for r in lines),
            "n_choice_invalid": dist(r["n_choice_invalid"] for r in lines),
            "n_static_invalid": dist(r["n_static_invalid"] for r in lines),
            "n_movable_objects": dist(len(r["spec"]["objects"]) for r in lines),
            "n_regions": dist(len(r["spec"]["regions"]) for r in lines),
            "n_repositioning_options": dist(len(r["spec"]["shifts"]) for r in lines),
        },
        "digest": ds.digest(lines), "generation_runtime_s": round(runtime, 2),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda v: v.item()))
    print(json.dumps({"rejection_sampling": rejection, "checks": checks, "distributions": summary["distributions"]},
                     indent=1))


if __name__ == "__main__":
    main()
