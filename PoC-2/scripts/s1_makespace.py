"""PoC-2 Stage 1: deterministic make-space recourse oracle (plan2.md section 17).

Run from the repository root:

    PYTHONPATH=PoC-1/src:PoC-2/src python PoC-2/scripts/s1_makespace.py

Writes PoC-2/out/s1/metrics.json. Exact enumeration only; no QUBO / Hopfield.
The cases are deterministic (no random draws); SEED is recorded for the contract.
"""

import json
import time
from itertools import combinations
from pathlib import Path

import mujoco

from poc.envelope import ENVELOPE_STEP
from poc.oracle import CONTACT_TOL_3D
from poc2 import oracle as orc
from poc2 import scenes
from poc2.types import SceneRecord

OUT = Path(__file__).resolve().parents[1] / "out" / "s1"
SEED = 7
FINE = ENVELOPE_STEP / 2
DIRECT_TOL = 1e-12  # [m] table vs direct do(S) agreement on G


def _sets(fs) -> list[list[str]]:
    return sorted(sorted(s) for s in fs)


def direct_check(case, table) -> dict:
    """Re-evaluate every choice-consistent subset by direct do(S) and compare with the table."""
    worst, mismatches, n = 0.0, 0, 0
    for x in range(len(table.M)):
        if table.M[x] == 0:
            continue
        a, v = orc.direct_row(case.scene, case.options, x)
        n += 1
        worst = max(worst, abs(a.G - table.G[x]))
        mismatches += (a.F != table.F[x]) or (v != table.V[x])
    return {"n_subsets": n, "label_mismatches": mismatches, "max_abs_dG": worst}


def mechanism_evidence(case, table) -> dict:
    """Pairwise sources kept separate: geometric beta^G, static-validity edges, choice-constraint edges."""
    ids = [o.option_id for o in case.options]
    beta = {f"{ids[p]}*{ids[q]}": orc.pair_effect(table, p, q) for p, q in combinations(range(len(ids)), 2)}
    groups = orc.choice_groups(case.options)
    return {
        "beta_G_nonzero": {k: v for k, v in beta.items() if v is not None and abs(v) > CONTACT_TOL_3D},
        "beta_G_undefined_choice_pairs": sorted(k for k, v in beta.items() if v is None),
        "validity_edges": sorted(f"{ids[p]}*{ids[q]}" for p, q in combinations(range(len(ids)), 2)
                                 if orc.validity_edge(table, p, q)),
        "choice_groups": {g.group_id: list(g.option_ids) for g in groups},
    }


def analyse(case) -> dict:
    t0 = time.perf_counter()
    table = orc.repair_table(case.scene, case.options)
    runtime = time.perf_counter() - t0
    fine = orc.repair_table(case.scene, case.options, FINE)
    cause, repair = orc.causes(table.original), orc.repair_result(table)
    cause_f, repair_f = orc.causes(fine.original), orc.repair_result(fine)
    record = SceneRecord(case.name, SEED, "make_space", case.regions, case.options, orc.choice_groups(case.options),
                         cause, repair)
    repaired_entities = sorted({o.intervention.entity_id for o in case.options
                                if any(o.option_id in S for S in repair.minimal_repairs)})
    a0 = table.original
    return {
        "name": case.name, "mechanism": case.mechanism, "P": len(case.options),
        "regions": [r.region_id for r in case.regions],
        "options": [{"id": o.option_id, "entity": o.intervention.entity_id, "kind": o.intervention.kind.value,
                     "region": o.region_id, "cost": o.cost} for o in case.options],
        "original": {"F0": a0.F, "G0": a0.G, "c0": dict(zip(a0.ids, a0.c)), "d0": dict(zip(a0.ids, a0.d))},
        "B0": sorted(cause.blockers), "C_star": _sets(cause.minimal_causes),
        "S_star": _sets(repair.minimal_repairs), "status": repair.status.value, "K_star": repair.cost,
        "n_tied": repair.n_tied, "n_choice_invalid": repair.n_choice_invalid,
        "n_static_invalid": repair.n_static_invalid,
        "C_star_matches": cause.minimal_causes == case.expected_causes,
        "S_star_matches": repair.minimal_repairs == case.expected_repairs,
        "repair_entities": repaired_entities,
        "cause_differs_from_repair_target": not set().union(*cause.minimal_causes) & set(repaired_entities),
        "mechanism_evidence": mechanism_evidence(case, table),
        "direct_vs_table": direct_check(case, table),
        "base_vs_fine": {"F0_equal": fine.original.F == a0.F, "C_star_equal": cause_f == cause,
                         "S_star_equal": repair_f.minimal_repairs == repair.minimal_repairs},
        "deterministic": orc.repair_result(orc.repair_table(case.scene, case.options)) == repair,
        "record_valid": isinstance(record, SceneRecord), "table_runtime_s": round(runtime, 3),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [analyse(c) for c in scenes.stage1_cases()]
    summary = {
        "all_C_star_match": all(r["C_star_matches"] for r in rows),
        "all_S_star_match": all(r["S_star_matches"] for r in rows),
        "all_direct_agree": all(r["direct_vs_table"]["label_mismatches"] == 0
                                and r["direct_vs_table"]["max_abs_dG"] <= DIRECT_TOL for r in rows),
        "all_base_fine_stable": all(all(r["base_vs_fine"].values()) for r in rows),
        "all_deterministic": all(r["deterministic"] for r in rows),
    }
    metrics = {"stage": 1, "seed": SEED, "family": "make_space", "mujoco_version": mujoco.__version__,
               "constants": {"CONTACT_TOL_3D": CONTACT_TOL_3D, "ENVELOPE_STEP": ENVELOPE_STEP, "FINE_STEP": FINE},
               "summary": summary, "cases": rows}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=lambda v: v.item()))  # numpy scalars
    print(json.dumps(summary, indent=1))
    for r in rows:
        ev = r["mechanism_evidence"]
        print(f'{r["name"]:26s} C*={r["C_star"]} S*={r["S_star"]} K*={r["K_star"]} Minv={r["n_choice_invalid"]} '
              f'Vinv={r["n_static_invalid"]} betaG={ev["beta_G_nonzero"]} Vedges={ev["validity_edges"]} '
              f'cause!=repair={r["cause_differs_from_repair_target"]}')


if __name__ == "__main__":
    main()
