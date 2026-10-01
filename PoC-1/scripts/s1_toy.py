"""Stage 1 experiment: 2D envelope oracle, exhaustive intervention truth, G study.

Run from the repository root:

    PYTHONPATH=src python scripts/s1_toy.py

Writes out/s1/metrics.json and out/s1/cases.png. Stage 1 is fully
deterministic (no random draws); SEED is recorded for the reproducibility contract.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from poc import toy2d as t
from poc.types import InterventionKind

OUT = Path("out/s1")
SEED = 7
REFINE_FACTORS = (2.0, 0.5)  # sampling-step multipliers for the label-stability check
BOUNDARY_DEPTHS = (-1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-3)  # [m], > 0 intrudes
G_UNITS = {"l1": "m", "l2sq": "m^2", "max": "m"}
INDEPENDENT_FAMILIES = ("one_blocker", "two_independent", "five_independent")


def _sets(fs) -> list[list[str]]:
    return sorted(sorted(s) for s in fs)


def digest(tables: dict) -> str:
    rows = [(n, sorted(r.selected_ids), r.conflict, r.G, r.F, r.K, r.J) for n, rs in tables.items() for r in rs]
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def run_all(step: float) -> dict:
    return {case.name: t.enumerate_subsets(case, step) for case in t.all_cases()}


def refinement(case, base, factor: float) -> dict:
    other = t.enumerate_subsets(case, t.SAMPLE_STEP * factor)
    return {
        "step": t.SAMPLE_STEP * factor,
        "label_flips": sum(a.F != b.F for a, b in zip(base, other)),
        "minimal_repairs_equal": t.minimal_repairs(other) == t.minimal_repairs(base),
        "max_abs_dc": max(float(np.max(np.abs(np.subtract(a.conflict, b.conflict)), initial=0.0))
                          for a, b in zip(base, other)),
    }


def case_metrics(case, results) -> dict:
    s_star = t.minimal_repairs(results)
    causes = t.diagnostic_causes(case.scene)
    targets = sorted({iv.entity_id for iv in case.candidates if any(iv.intervention_id in s for s in s_star)})
    return {
        "name": case.name, "family": case.family, "action": case.scene.action.action_id,
        "P": len(case.candidates), "K_max": case.k_max, "B": case.k_max + 1,
        "c_empty": {b.eid: c for b, c in zip(case.scene.entities, results[0].conflict)},
        "expected_minimal_repairs": _sets(case.expected), "found_minimal_repairs": _sets(s_star),
        "match": s_star == case.expected,
        "minimal_cardinality": sorted({len(s) for s in s_star}),
        "n_feasible_subsets": sum(r.feasible for r in results),
        "diagnostic_causes": list(causes), "repair_targets": targets,
        "cause_differs_from_repair_target": bool(set(causes) - set(targets)),
        "refinement": [refinement(case, results, f) for f in REFINE_FACTORS],
    }


def boundary_sweep() -> list[dict]:
    rows = []
    for kind in t.BOUNDARY_KINDS:
        for depth in BOUNDARY_DEPTHS:
            margin = float(t.entity_margins(t.boundary_scene(kind, depth))[0])
            label = t.feasibility(np.maximum([margin], 0.0))
            rows.append({"kind": kind, "depth": depth, "margin": margin, "F": label,
                         "expected_F": int(depth > t.CONTACT_TOL), "correct": label == int(depth > t.CONTACT_TOL)})
    return rows


def g_study(cases: dict, tables: dict, boundary: list[dict]) -> dict:
    """Compare candidate g(c^S) on the exhaustive tables (plan.md section 9)."""
    study = {}
    for form, g in t.G_FORMS.items():
        strict, relocations, increases, pairs, sens = 0, 0, 0, 0, []
        g_feas, g_infeas = [], [g(np.array([r["margin"]])) for r in boundary if r["F"] == 1]
        for name, results in tables.items():
            case = cases[name]
            idx = {b.eid: i for i, b in enumerate(case.scene.entities)}
            G = [g(np.asarray(r.conflict)) for r in results]
            for r, gx in zip(results, G):
                (g_feas if r.feasible else g_infeas).append(gx)
            for x, r in enumerate(results):
                for p, iv in enumerate(case.candidates):
                    if x >> p & 1:
                        continue
                    pairs += 1
                    increases += G[x | 1 << p] > G[x]
                    if iv.kind is InterventionKind.RELOCATE and r.conflict[idx[iv.entity_id]] > t.CONTACT_TOL:
                        relocations += 1
                        strict += G[x | 1 << p] < G[x]
                        if x == 0 and case.family in INDEPENDENT_FAMILIES:
                            sens.append((G[0] - G[1 << p]) / G[0])
        positive = [v for v in g_infeas if v > 0]
        study[form] = {
            "units": G_UNITS[form],
            "strict_decrease_rate_relocating_a_conflicting_blocker": strict / relocations,
            "increase_events_fraction": increases / pairs,
            "independent_blocker_min_relative_drop": min(sens),
            "independent_blocker_mean_relative_drop": float(np.mean(sens)),
            "max_G_feasible": max(g_feas), "min_G_infeasible": min(positive),
            "dynamic_range_max_over_min_infeasible": max(g_infeas) / min(positive),
            "min_G_infeasible_over_contact_tol": min(positive) / t.CONTACT_TOL,
        }
    eligible = [f for f, m in study.items()
                if m["units"] == "m" and m["strict_decrease_rate_relocating_a_conflicting_blocker"] == 1.0
                and m["independent_blocker_min_relative_drop"] > 0.0]
    rule = ("linear physical units [m], strictly decreases whenever a conflicting blocker is relocated, "
            "every independent blocker changes G; then smallest dynamic range")
    chosen = min(eligible, key=lambda f: study[f]["dynamic_range_max_over_min_infeasible"]) if eligible else None
    return {"forms": study, "selection_rule": rule, "eligible": eligible, "chosen": chosen,
            "matches_toy2d_default": chosen == t.G_FORM}


def plot(cases: dict, tables: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, (name, case) in zip(axes.flat, cases.items()):
        env = t.envelope(case.scene)
        for poly in env[:: max(1, len(env) // 80)]:
            ax.fill(*poly.T, color="0.85", lw=0)
        s_star = sorted(t.minimal_repairs(tables[name]), key=sorted)
        shifted = [iv for iv in case.candidates if s_star and iv.intervention_id in s_star[-1]
                   and iv.kind is InterventionKind.SHIFT_TARGET]
        if shifted:
            for poly in t.envelope(t.do(case.scene, shifted))[:: max(1, len(env) // 40)]:
                ax.plot(*np.vstack([poly, poly[:1]]).T, color="tab:green", lw=0.3)
        for b, c in zip(case.scene.entities, tables[name][0].conflict):
            color = "tab:red" if c > t.CONTACT_TOL else ("k" if not b.entity.movable else "tab:blue")
            ax.fill(*b.polygon().T, color=color, alpha=0.6)
            ax.annotate(b.eid, np.mean(b.polygon(), axis=0), fontsize=7, ha="center")
        ax.set_title(f"{name}\nS* = {_sets(s_star)}", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(cases):]:
        ax.axis("off")
    fig.suptitle("Stage 1: grey = sampled envelope E, red = c_i > CONTACT_TOL, "
                 "black = structural, green = envelope after S* shift", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {c.name: c for c in t.all_cases()}
    t0 = time.perf_counter()
    tables = run_all(t.SAMPLE_STEP)
    runtime = time.perf_counter() - t0
    repeat = run_all(t.SAMPLE_STEP)
    per_case = [case_metrics(cases[n], rs) for n, rs in tables.items()]
    boundary = boundary_sweep()
    metrics = {
        "stage": 1, "seed": SEED,
        "constants": {"CONTACT_TOL": t.CONTACT_TOL, "SAMPLE_STEP": t.SAMPLE_STEP, "G_FORM": t.G_FORM,
                      "MAX_CANDIDATES": t.MAX_CANDIDATES},
        "phi": "max over sampled envelope poses of SAT minimum translation distance [m]",
        "cases": per_case,
        "boundary_sweep": boundary,
        "g_study": g_study(cases, tables, boundary),
        "determinism": {"digest": digest(tables), "repeat_digest": digest(repeat),
                        "identical": digest(tables) == digest(repeat)},
        "summary": {
            "all_minimal_repairs_match": all(c["match"] for c in per_case),
            "all_boundary_labels_correct": all(r["correct"] for r in boundary),
            "total_label_flips_under_refinement": sum(r["label_flips"] for c in per_case for r in c["refinement"]),
            "all_minimal_repairs_stable_under_refinement": all(r["minimal_repairs_equal"]
                                                              for c in per_case for r in c["refinement"]),
            "n_subsets_evaluated": sum(len(rs) for rs in tables.values()),
            "enumeration_runtime_s": round(runtime, 3),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot(cases, tables, OUT / "cases.png")
    print(json.dumps(metrics["summary"], indent=2))
    print(json.dumps(metrics["g_study"], indent=2))


if __name__ == "__main__":
    main()
