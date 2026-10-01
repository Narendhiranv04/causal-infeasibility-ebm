"""PoC-2 Stage 4: benchmark-inspired multi-task extension of the structural dataset (plan2.md section 20).

Run from the repository root (after Stage 2 wrote PoC-2/out/s2/scenes.jsonl):

    PYTHONPATH=PoC-1/src:PoC-2/src python PoC-2/scripts/s4_tasks.py

Writes PoC-2/out/s4/scenes.jsonl (the unchanged 40 make-space records + 3 x 20 new scenes),
summary.json and examples.png. Each added family has a pre-registered composition of 15
repairable + 5 feasible hard negatives; draws are accepted only by intent / validity / catalogue
criteria, never by interaction structure. No structure analysis here.
"""

import json
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from poc import envelope as ev
from poc.envelope import ENVELOPE_STEP
from poc2 import dataset as ds
from poc2 import oracle as orc
from poc2 import tasks
from poc2.oracle import Polyline

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s4"
N_DIRECT = 3     # first scenes of each added family checked subset-by-subset against direct do(S)


def as_json(obj) -> dict:
    return json.loads(json.dumps(obj, default=lambda v: v.item()))


def direct_check(spec) -> dict:
    scene, _, options = tasks.build(spec)
    T = orc.repair_table(scene, options)
    n, bad, worst = 0, 0, 0.0
    for x in np.flatnonzero(T.M == 1):
        a, v = orc.direct_row(scene, options, int(x))
        n, bad, worst = n + 1, bad + ((a.F != T.F[x]) or (v != T.V[x])), max(worst, abs(a.G - T.G[x]))
    return {"scene_id": spec.scene_id, "P": len(options), "n_subsets": n, "label_mismatches": int(bad), "max_abs_dG": worst}


def dist(values) -> dict:
    return {str(k): v for k, v in sorted(Counter(values).items())}


def family_summary(lines: list[dict]) -> dict:
    return {"n": len(lines), "intent": dist(r["intent"] for r in lines), "status": dist(r["status"] for r in lines),
            "P": dist(r["P"] for r in lines), "B0_size": dist(r["causal_set_size"] for r in lines),
            "repair_size": dist(r["repair_size"] for r in lines), "n_tied": dist(r["n_tied_repairs"] for r in lines),
            "structural_cause": sum(any(b in {"divider", "post", "wall", "jamb"} for b in r["B0"]) for r in lines),
            "n_repositioning_options": dist(len(r["spec"]["shifts"]) for r in lines),
            "n_movable_objects": dist(len(r["spec"]["objects"]) for r in lines)}


def plot_examples(specs: dict, path: Path) -> None:
    """Top/side projections of one repairable scene per family with sampled envelope poses (diagnostic)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    views = {"storage_insertion": (0, 2, "side view x-z"), "storage_extraction": (0, 2, "side view x-z"),
             "articulated_opening": (0, 1, "top view x-y")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#fcfcfb")
    for ax, (fam, spec) in zip(axes, specs.items()):
        i, j, title = views[fam]
        scene, _, _ = tasks.build(spec)
        a = orc.evaluate(scene)
        for ent, c in zip(scene.entities, a.c):
            pts = ent.boxes[0].corners()[:, [i, j]]
            color = "#e34948" if c > 0 else ("#cfcdc4" if not ent.entity.movable else "#1baf7a")
            lo, hi = pts.min(0), pts.max(0)
            ax.add_patch(Polygon([lo, (hi[0], lo[1]), hi, (lo[0], hi[1])], fc=color, ec="#52514e", lw=0.6, alpha=0.6))
        segs = scene.motion.segments if isinstance(scene.motion, Polyline) else (scene.motion,)
        for seg in segs:
            _, frames = ev.poses(seg, scene.moving, 0.02)
            for pos, quat in frames:
                for part in np.split(ev.world_corners(scene.moving, pos, quat), len(scene.moving.parts)):
                    p2 = part[:, [i, j]]
                    lo, hi = p2.min(0), p2.max(0)
                    ax.add_patch(Polygon([lo, (hi[0], lo[1]), hi, (lo[0], hi[1])], fc="#2a78d6", ec="none", alpha=0.05))
        ax.set_title(f"{fam}: {spec.scene_id} ({title})\nred = blocker (c > 0), grey = structure, green = movable, "
                     "blue = sampled envelope", loc="left", fontsize=8)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_xlabel(f"{'xyz'[i]} [m]")
        ax.set_ylabel(f"{'xyz'[j]} [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_space = ds.load_dataset(ROOT / "s2" / "scenes.jsonl")            # unchanged 30 + 10 make-space records
    slots = ds.intents(ds.N_REPAIRABLE_FAMILY, ds.N_NEGATIVE_FAMILY)     # pre-registered 15 + 5 per family
    t0, generated, logs = time.perf_counter(), {}, {}
    for fam in tasks.FAMILIES:
        generated[fam], logs[fam] = ds.generate(slots, ds.MAX_ATTEMPTS_FAMILY, fam)
    runtime = time.perf_counter() - t0
    new = [as_json(ds.to_record_json(*row)) for fam in tasks.FAMILIES for row in generated[fam]]
    lines = make_space + new
    (OUT / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))

    again = [as_json(ds.to_record_json(*row)) for fam in tasks.FAMILIES
             for row in ds.generate(slots, ds.MAX_ATTEMPTS_FAMILY, fam)[0]]
    labs = {r["scene_id"]: ds.label(ds.spec_from_json(r["spec"])) for r in lines}
    rebuilt = [as_json(ds.to_record_json(ds.spec_from_json(r["spec"]), r["intent"], ds.record_stream_attempt(r),
                                         labs[r["scene_id"]])) for r in lines]
    fine = {r["scene_id"]: ds.labels_of(ds.label(ds.spec_from_json(r["spec"]), ENVELOPE_STEP / 2))
            == ds.labels_of(labs[r["scene_id"]]) for r in lines}
    direct = [direct_check(spec) for fam in tasks.FAMILIES for spec, *_ in generated[fam][:N_DIRECT]]
    regression = {}
    for fam, (spec, eC, eS) in tasks.regression_cases().items():
        lab = ds.label(spec)
        regression[fam] = {"C_star_match": lab["cause"].minimal_causes == eC, "S_star_match": lab["repair"].minimal_repairs == eS,
                           "S_star": sorted(sorted(s) for s in lab["repair"].minimal_repairs)}
    by_family = {fam: [r for r in lines if r["family"] == fam] for fam in (ds.FAMILY,) + tasks.FAMILIES}
    checks = {
        "n_scenes": len(lines), "per_family": {f: len(v) for f, v in by_family.items()},
        "make_space_unchanged": ds.digest(make_space) == ds.STAGE2_DIGEST,
        "composition_met": all(dist(r["intent"] for r in by_family[f]) == {"negative": 5, "repairable": 15}
                               for f in tasks.FAMILIES),
        "reproducible_regeneration": ds.digest(again) == ds.digest(new),
        "all_reconstruct_from_records": all(a == b for a, b in zip(rebuilt, lines)),
        "original_scenes_statically_valid": all(labs[k]["table"].V[0] == 1 for k in labs),
        "hard_negatives_feasible": all(r["status"] == "FEASIBLE" for r in lines if r["intent"] == "negative"),
        "repairables_have_valid_repair": all(r["status"] == "REPAIRED" for r in lines if r["intent"] == "repairable"),
        "base_vs_fine_F_Cstar_Sstar_stable": all(fine.values()), "n_base_fine_checked": len(fine),
        "direct_vs_table_agree": all(d["label_mismatches"] == 0 and d["max_abs_dG"] <= 1e-12 for d in direct),
        "regression_cases_match": all(v["C_star_match"] and v["S_star_match"] for v in regression.values()),
        "C_star_equals_B0_everywhere": all(r["C_star"] == ([r["B0"]] if r["B0"] else []) for r in lines),
        "all_unit_cost": all(o["cost"] == 1 for r in lines for o in r["candidate_interventions"]),
        "same_oracle_definitions": "every record is labelled by poc2.dataset.label -> poc2.oracle (M, V, F, G, C*, S*)",
        "no_outcome_based_selection": "draws accepted only by intent / validity / catalogue criteria",
    }
    summary = {"stage": 4, "seed": ds.SEED, "mujoco_version": mujoco.__version__, "causal_note": ds.CAUSAL_NOTE,
               "cost_note": "unit repair cost for every intervention: K(S) = |S|; ties kept",
               "envelopes": {"make_space": "side grasp, straight horizontal insertion into a cupboard shelf",
                             "storage_insertion": "top grasp, transfer at height then vertical descent into a bin",
                             "storage_extraction": "side grasp, lift over a front rail then withdraw from a cubby",
                             "articulated_opening": "door rotating about a vertical hinge axis"},
               "rejection_sampling": logs, "checks": checks, "regression_cases": regression, "direct_vs_table": direct,
               "families": {f: family_summary(v) for f, v in by_family.items()}, "digest": ds.digest(lines),
               "new_family_digest": ds.digest(new), "generation_runtime_s": round(runtime, 2)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda v: v.item()))
    examples = {fam: next(spec for spec, intent, *_ in generated[fam] if intent == "repairable") for fam in tasks.FAMILIES}
    plot_examples(examples, OUT / "examples.png")
    print(json.dumps({"rejection_sampling": logs, "checks": checks, "regression_cases": regression}, indent=1))
    for fam in tasks.FAMILIES:
        print(fam, json.dumps(summary["families"][fam]))


if __name__ == "__main__":
    main()
