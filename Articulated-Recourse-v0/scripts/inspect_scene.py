"""Build one scene, run the oracle and print a compact structural summary.

    python scripts/inspect_scene.py V5            # canonical scene by variant
    python scripts/inspect_scene.py path/to/oracle.json   # re-inspect a saved scene spec
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artrecourse.interventions import SceneSpec, build_problem  # noqa: E402
from artrecourse.oracle import solve  # noqa: E402
from artrecourse.variants import check_variant, classify_all  # noqa: E402


def summarize(spec: SceneSpec, with_robot=True, verbose=True):
    t = time.time()
    pb = build_problem(spec, with_robot=with_robot)
    r = solve(pb)
    scene = {"objects": spec.objects, "candidate_interventions": pb.ivs}
    ok, why = check_variant(spec.variant_requested, r, scene) if spec.variant_requested else (None, [])
    print(f"== {spec.scene_id} [{spec.variant_requested}] {'PASS' if ok else 'FAIL'} {why} "
          f"({time.time() - t:.1f}s) problems={pb.problems}")
    if verbose:
        for iv in pb.ivs:
            if not iv.admissible:
                print("   INADMISSIBLE", iv.id, "|", iv.rejection[0][:160])
        print("   direct blockers", {o: (d["min_dist"], d["part_star"]) for o, d in r["direct_target_blocker_details"].items()})
        print("   executable@s0", r["executable_initially"])
        for iv in pb.ivs:
            if iv.admissible:
                ex, bl = pb.exec_info(iv, pb.s0)
                if not ex:
                    print("   blocked@s0", iv.id, [(o, round(p.min_dist, 4), p.phase_star, p.part_star) for o, p in bl if p])
        print("   optimal cost", r["optimal_cost"], "n sequences", r["n_optimal_sequences"])
        for s in r["optimal_sequences"][:4]:
            print("     ", " -> ".join(s))
        for p in r["sequence_proofs"][:1]:
            for e in p["edges"]:
                c = e["cause"]
                print(f"     edge {e['from']} -> {e['to']} | blocker {e['blocker']} | {c.get('type')} "
                      f"phase={c.get('phase_star')} d={c.get('min_dist')}")
        print("   irreducible plans", [(s["length"], s["interventions"]) for s in r["irreducible_solutions"]][:6])
        print("   compatibility", {k: len(v) for k, v in r["compatibility_edges"].items()},
              "enable@s0", len(r["enable_edges"]), "disable@s0", len(r["disable_edges"]))
        print("   satisfies", [k for k, v in classify_all(r, scene).items() if v],
              "| consulted ambiguities", r["consulted_ambiguities"])
    return pb, r, ok


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg.endswith(".json"):
        spec = SceneSpec.from_dict(json.loads(Path(arg).read_text())["spec"])
    else:
        from artrecourse.canonical import ALL

        spec = ALL[arg]()
    summarize(spec)
