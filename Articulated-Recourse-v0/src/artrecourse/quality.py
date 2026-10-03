"""Scene quality gates.

* coarse / fine agreement: the whole problem is rebuilt at FINE_SPACING (2.5 mm, 4x denser
  tau sampling) with the same grasps and every executability / feasibility status plus the
  oracle result must be identical; any disagreement rejects the scene;
* no consulted ambiguity: no table entry with |min distance| < 4 mm may be consulted by the
  oracle (a dependency must not hinge on numerical tolerance);
* static validity: initial objects resting, non-interpenetrating, inside their supports.
"""

from __future__ import annotations

from .interventions import RecourseProblem, build_problem
from .oracle import solve
from .primitives import FINE_SPACING


def coarse_fine_check(pb: RecourseProblem, res: dict) -> dict:
    override = {iv.id: (iv.grasp.name if iv.admissible else "__inadmissible__") for iv in pb.ivs}
    fine = build_problem(pb.spec, spacing=FINE_SPACING, with_robot=False, grasp_override=override)
    rf = solve(fine)
    dis = []
    for iv in pb.ivs:
        ivf = fine.iv_by_id[iv.id]
        if iv.admissible != ivf.admissible:
            dis.append(f"admissibility of {iv.id}")
            continue
        for o, profs in pb.tab[iv.id].items():
            for k, p in enumerate(profs):
                q = fine.tab[iv.id][o][k]
                if (p.status == "conflict") != (q.status == "conflict"):
                    dis.append(f"{iv.id} vs {o}@{k}: coarse {p.min_dist:+.4f} fine {q.min_dist:+.4f}")
    for o, profs in pb.ttab.items():
        for k, p in enumerate(profs):
            q = fine.ttab[o][k]
            if (p.status == "conflict") != (q.status == "conflict"):
                dis.append(f"TARGET vs {o}@{k}: coarse {p.min_dist:+.4f} fine {q.min_dist:+.4f}")
    for key in ("F_initial", "optimal_cost", "optimal_sequences", "no_recourse", "direct_target_blockers"):
        if res[key] != rf[key]:
            dis.append(f"oracle field {key} differs")
    worst_shift = max([abs(p.min_dist - fine.tab[iv.id][o][k].min_dist) for iv in pb.ivs
                       for o, profs in pb.tab[iv.id].items() for k, p in enumerate(profs)] + [0.0])
    return {"coarse_spacing_samples": int(sum(iv.traj.n for iv in pb.ivs) + pb.target.n),
            "fine_spacing_samples": int(sum(iv.traj.n for iv in fine.ivs) + fine.target.n),
            "disagreements": dis, "agree": not dis,
            "max_min_distance_shift_coarse_vs_fine_m": round(float(worst_shift), 5),
            "fine_consulted_ambiguities": rf["consulted_ambiguities"]}


def quality_report(pb: RecourseProblem, res: dict, fine: dict | None) -> dict:
    q = {
        "static_problems": pb.problems,
        "consulted_ambiguities": res["consulted_ambiguities"],
        "coarse_fine": fine,
        "target_robot_coverage": getattr(pb, "target_robot", None),
        "target_gripper_vs_fixed_env": pb.target_env.to_dict(),
        "distance_glitch_retries": pb.eng.glitches,
        "n_inadmissible_interventions": sum(not iv.admissible for iv in pb.ivs),
    }
    q["accepted"] = bool(not pb.problems and not res["consulted_ambiguities"]
                         and (fine is None or (fine["agree"] and not fine["fine_consulted_ambiguities"])))
    return q
