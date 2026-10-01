"""Stage 4 experiment: minimal 3D MuJoCo envelope oracle.

Run from the repository root:

    PYTHONPATH=src python scripts/s4_mujoco.py

Writes out/s4/metrics.json. No QUBO, no Hopfield: this stage only validates the
3D geometric intervention oracle. Deterministic (no random draws); SEED is
recorded for the reproducibility contract.
"""

import io
import json
import math
import time
import tokenize
from pathlib import Path

import mujoco
import numpy as np

from poc import cases as cs
from poc import envelope as ev
from poc import mj_scene as ms
from poc import oracle as orc
from poc.types import Entity, EntityRole

OUT = Path("out/s4")
SEED = 7
STEPS = {"base": ev.ENVELOPE_STEP, "fine": ev.ENVELOPE_STEP / 2, "coarse": ev.ENVELOPE_STEP * 2}
BOUNDARY_TOL = 1e-5  # [m] allowed |d - (-/+ DELTA)| at a near-boundary pair (hinge arc sampling)


def geometry_sanity() -> list[dict]:
    """Analytic box-box checks of the signed-distance wrapper (0.2 m cube at the origin)."""
    cube = ms.Entity3D(Entity("cube", EntityRole.STRUCTURAL), (ms.Box3D((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)),))
    q45 = (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
    corner45 = 0.1 + 0.1 * math.sqrt(2)
    checks = [("separation", (0.1, 0.1, 0.1), ms.IDENTITY, (0.5, 0, 0), 0.3),
              ("separation_diagonal", (0.1, 0.1, 0.1), ms.IDENTITY, (0.3, 0.3, 0), 0.1 * math.sqrt(2)),
              ("face_to_face_raw_zero", (0.1, 0.05, 0.05), ms.IDENTITY, (0.05, 0.25, 0.05), 0.1),
              ("touching", (0.1, 0.1, 0.1), ms.IDENTITY, (0.2, 0, 0), 0.0),
              ("near_touching_out", (0.1, 0.1, 0.1), ms.IDENTITY, (0.200001, 0, 0), 1e-6),
              ("near_touching_in", (0.1, 0.1, 0.1), ms.IDENTITY, (0.199999, 0, 0), -1e-6),
              ("shallow_penetration", (0.1, 0.1, 0.1), ms.IDENTITY, (0.199, 0, 0), -0.001),
              ("deeper_penetration", (0.1, 0.1, 0.1), ms.IDENTITY, (0.1, 0, 0), -0.1),
              ("coincident_centres", (0.1, 0.1, 0.1), ms.IDENTITY, (0.0, 0, 0), -0.2),
              ("rotated45_separation", (0.1, 0.1, 0.1), q45, (0.3, 0, 0), 0.3 - corner45),
              ("rotated45_penetration", (0.1, 0.1, 0.1), q45, (0.2, 0, 0), 0.2 - corner45)]
    rows = []
    for name, half, quat, pos, expected in checks:
        world = ms.GeomWorld((cube,), ms.Composite((("probe", ms.Box3D((0.0, 0.0, 0.0), half, quat)),)))
        world.set_pose(pos, ms.IDENTITY)
        gm, ge = world.moving_geoms[0], world.entity_geoms[0][0]
        got = world.signed_distance(gm, ge)
        rows.append({"check": name, "expected": expected, "raw_mujoco": world._raw(gm, ge, ms.DISTMAX),
                     "wrapped": got, "abs_error": abs(got - expected), "sign_ok": np.sign(got) == np.sign(expected)})
    return rows


def per_entity(a: orc.Assessment, sw: ev.Sweep) -> dict:
    return {i: {"d": a.d[k], "d_start": a.d_start[k], "d_goal": a.d_goal[k], "p": a.p[k], "c": a.c[k],
                "tau_argmin": float(sw.tau_argmin[k])} for k, i in enumerate(a.ids)}


def stability(case: cs.Case3D, base: orc.Assessment) -> dict:
    out = {}
    for label in ("fine", "coarse"):
        other, _ = cs.evaluate(case.scene, STEPS[label])
        out[f"base_vs_{label}"] = {
            "step": STEPS[label], "label_flip": other.F != base.F, "blockers_changed": other.blockers != base.blockers,
            "max_abs_dd": float(np.max(np.abs(np.subtract(other.d, base.d)))),
            "max_abs_dc": float(np.max(np.abs(np.subtract(other.c, base.c))))}
    return out


def case_metrics(case: cs.Case3D) -> dict:
    a, sw = cs.evaluate(case.scene)
    causes = orc.diagnostic_causes(a)
    row = {"name": case.name, "family": case.scene.family, "kind": case.kind, "boundary": case.boundary,
           "n_samples": len(sw.taus), "roles": {e.eid: e.entity.role.value for e in case.scene.entities},
           "entities": per_entity(a, sw), "c": list(a.c), "G": a.G, "F": a.F,
           "start_feasible": a.F_start == 0, "goal_feasible": a.F_goal == 0, "interior_only_failure": a.interior_only,
           "diagnostic_causes": list(causes), "expected_causes": list(case.cause), "causes_match": causes == case.cause,
           "diagnostic_exclusion_restores_feasibility": (orc.feasibility_excluding(a, causes) == 0) if causes else None,
           "stability": stability(case, a), "deterministic": cs.evaluate(case.scene)[0] == a}
    if case.repair:
        fixed, _ = cs.evaluate(cs.do(case.scene, case.repair))
        targets = sorted({iv.entity_id for iv in case.repair})
        row["executable_repair"] = {"interventions": [iv.intervention_id for iv in case.repair],
                                    "kinds": [iv.kind.value for iv in case.repair], "repair_targets": targets,
                                    "restores_feasibility": fixed.F == 0, "G_after": fixed.G, "min_d_after": min(fixed.d),
                                    "cause_differs_from_repair_target": set(targets).isdisjoint(causes)}
    return row


def phase_identifiers() -> list[str]:
    found = []
    for path in sorted(Path("src/poc").glob("*.py")):
        for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
            if t.type == tokenize.NAME and "phase" in t.string.lower():
                found.append(f"{path.name}:{t.string}")
    return found


def boundary_pairs(rows: list[dict]) -> list[dict]:
    out = []
    for fam in "ABC":
        lo, hi = (next(r for r in rows if r["name"].startswith(fam) and r["kind"] == k)
                  for k in ("near_boundary_out", "near_boundary_in"))
        eid = hi["expected_causes"][0]
        d_out, d_in = lo["entities"][eid]["d"], hi["entities"][eid]["d"]
        out.append({"family": lo["family"], "entity": eid, "boundary": lo["boundary"], "delta": cs.DELTA,
                    "d_out": d_out, "d_in": d_in, "F_out": lo["F"], "F_in": hi["F"],
                    "flips_at_boundary": (lo["F"], hi["F"]) == (0, 1)
                    and abs(d_out - cs.DELTA) <= BOUNDARY_TOL and abs(d_in + cs.DELTA) <= BOUNDARY_TOL})
    return out


def gate(rows, sanity, pairs, phases) -> dict:
    repaired = [r for r in rows if "executable_repair" in r]
    checks = {
        "deterministic": all(r["deterministic"] for r in rows),
        "geometry_sanity": all(s["abs_error"] <= 1e-8 and s["sign_ok"] for s in sanity),
        "near_boundary_pairs_flip": all(p["flips_at_boundary"] for p in pairs),
        "zero_base_vs_fine_label_flips": not any(r["stability"]["base_vs_fine"]["label_flip"]
                                                 or r["stability"]["base_vs_fine"]["blockers_changed"] for r in rows),
        "cupboard_endpoint_free_envelope_infeasible": any(r["interior_only_failure"] for r in rows
                                                          if r["family"] in ("insertion", "extraction")),
        "hinge_endpoint_free_envelope_infeasible": any(r["interior_only_failure"] for r in rows if r["family"] == "hinge"),
        "diagnostic_causes_match": all(r["causes_match"] for r in rows),
        "executable_repairs_restore_feasibility": all(r["executable_repair"]["restores_feasibility"] for r in repaired),
        "cause_differs_from_repair_target": all(r["executable_repair"]["cause_differs_from_repair_target"]
                                                for r in repaired if r["kind"] == "immovable_cause"),
        "feasible_implies_G_zero": all(r["G"] == 0.0 for r in rows if r["F"] == 0),
        "no_phase_identifiers": not phases,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = [case_metrics(c) for c in cs.canonical_cases()]
    sanity, phases = geometry_sanity(), phase_identifiers()
    pairs = boundary_pairs(rows)
    metrics = {
        "stage": 4, "seed": SEED, "mujoco_version": mujoco.__version__,
        "constants": {"CONTACT_TOL_3D": orc.CONTACT_TOL_3D, "ENVELOPE_STEP": ev.ENVELOPE_STEP, "steps": STEPS,
                      "sampling_criterion": "max displacement of any moving-composite vertex between adjacent samples",
                      "DISTMAX": ms.DISTMAX, "ZERO_PROBE": ms.ZERO_PROBE, "BISECTION_RES": ms.BISECTION_RES,
                      "COINCIDENT_NUDGE": ms.COINCIDENT_NUDGE, "DELTA": cs.DELTA, "HARD_GAP": cs.HARD_GAP},
        "conflict_definition": "d_i = min over tau, moving geoms, entity geoms of MuJoCo signed distance; "
                               "p_i = max(0, -d_i); c_i = max(0, p_i - CONTACT_TOL_3D); F = 1[any c_i > 0]; G = sum c_i",
        "gate": gate(rows, sanity, pairs, phases),
        "geometry_sanity": sanity, "near_boundary_pairs": pairs, "phase_identifiers": phases,
        "cases": rows, "runtime_s": round(time.perf_counter() - t0, 2),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({"mujoco": metrics["mujoco_version"], "gate": metrics["gate"], "runtime_s": metrics["runtime_s"]}, indent=1))
    for r in rows:
        rep = r.get("executable_repair", {})
        print(f'{r["name"]:20s} F={r["F"]} G={r["G"]:.4f} start_ok={r["start_feasible"]} goal_ok={r["goal_feasible"]} '
              f'interior_only={r["interior_only_failure"]} causes={r["diagnostic_causes"]} '
              f'repair={rep.get("interventions")} ok={rep.get("restores_feasibility")} '
              f'dd_fine={r["stability"]["base_vs_fine"]["max_abs_dd"]:.2e}')


if __name__ == "__main__":
    main()
