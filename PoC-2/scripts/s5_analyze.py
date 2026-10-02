"""PoC-2 Stage 5: cross-family structural recourse analysis (plan2.md section 21).

Run from the repository root (after Stage 4 wrote PoC-2/out/s4/scenes.jsonl):

    PYTHONPATH=PoC-1/src:PoC-2/src python PoC-2/scripts/s5_analyze.py

Writes PoC-2/out/s5/analysis.json, decision_recovery.png, graph_structure.png. The fixed 100-scene
Stage-4 dataset is loaded and digest-verified (never regenerated); exact tables are recomputed.
Stage-3 rules (poc2/structure.py) are reused unchanged; Stage-5 graph definitions are
pre-registered there. No optimizer, QUBO or Hopfield.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from poc.envelope import ENVELOPE_STEP
from poc.types import InterventionKind
from poc2 import dataset as ds
from poc2 import structure as st

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s5"
FINE = ENVELOPE_STEP / 2
FAMILIES = ("make_space", "storage_insertion", "storage_extraction", "articulated_opening")


def _sets(table, xs) -> list[list[str]]:
    return sorted(sorted(table.subset(x)) for x in xs)


def analyse(record: dict) -> dict:
    spec = ds.spec_from_json(record["spec"])
    lab_b, lab_f = ds.label(spec), ds.label(spec, FINE)
    tb, tf, options = lab_b["table"], lab_f["table"], lab_b["options"]
    P = len(options)
    a_b, a_f = st.compatible_mobius(tb.G, tb.M), st.compatible_mobius(tf.G, tf.M)
    sig = st.significant(a_b, a_f)
    eps, exact = st.eps_repr(tb.G, tf.G, tb.M), st.exact_decision(tb)
    orders = {}
    for k in st.ORDERS:
        gk_b, gk_f = st.reconstruct(a_b, k), st.reconstruct(a_f, k)
        d_b = st.order_decision(tb, gk_b)
        orders[k] = {**st.representation_errors(tb.G, tb.M, gk_b), "correct": d_b == exact, "S_star_k": _sets(tb, d_b),
                     "base_fine_decision_equal": _sets(tb, d_b) == _sets(tf, st.order_decision(tf, gk_f))}
    edges = st.interaction_edges(tb, a_b, sig)
    hubs = {p for p, o in enumerate(options) if o.intervention.kind is InterventionKind.SHIFT_TARGET}
    geo = [(p, q) for p, q, _, _ in edges["geometric"]]
    phys = [(p, q) for p, q, _, kind in edges["geometric"] if kind == st.PHYSICAL]
    union = sorted(set(geo) | set(edges["validity"]) | set(edges["choice"]))
    in_repair = [kind for p, q, _, kind in edges["geometric"] if any(x >> p & 1 and x >> q & 1 for x in exact)]
    touching = [kind for p, q, _, kind in edges["geometric"] if any((x >> p & 1) or (x >> q & 1) for x in exact)]
    g = tb.G[tb.M == 1]
    size = np.array([bin(x).count("1") for x in range(len(tb.G))])
    return {
        "scene_id": record["scene_id"], "family": record["family"], "P": P, "B0_size": len(lab_b["cause"].blockers),
        "S_star": _sets(tb, exact), "repair_size": int(min(bin(x).count("1") for x in exact)), "n_tied": len(exact),
        "labels_match_stored": _sets(tb, exact) == record["S_star"],
        "eps_repr": eps, "representation_order": {"base": st.representation_order(tb.G, tb.M, a_b, eps),
                                                  "fine": st.representation_order(tf.G, tf.M, a_f, eps)},
        "max_significant_order": int(size[sig].max(initial=0)), "orders": orders,
        "edges": {"geometric": [[options[p].option_id, options[q].option_id, b, kind] for p, q, b, kind in edges["geometric"]],
                  "validity": [[options[p].option_id, options[q].option_id] for p, q in edges["validity"]],
                  "choice": [[options[p].option_id, options[q].option_id] for p, q in edges["choice"]]},
        "graphs": {"geometric": st.graph_metrics(P, geo), "physical": st.graph_metrics(P, phys),
                   "validity": st.graph_metrics(P, edges["validity"]), "choice": st.graph_metrics(P, edges["choice"]),
                   "combined": st.graph_metrics(P, union)},
        "geometric_class": st.geometric_class(P, geo, hubs),
        "n_macros_touched": len({p for e in geo for p in e if p in hubs}),
        "repositioning_offered": bool(hubs), "n_repositioning_options": len(hubs),
        "geometric_kinds_inside_exact_repair": in_repair, "geometric_kinds_touching_exact_repair": touching,
        "validity_binding": st.validity_binding(tb), "distractor_fraction": st.distractor_fraction(P, exact),
        "min_positive_G": float(g[g > 0].min(initial=np.inf)),
    }


def mean(rows, f) -> float:
    return float(np.mean([f(r) for r in rows])) if rows else float("nan")


def aggregate(rows: list[dict]) -> dict:
    ok = {k: (lambda r, k=k: r["orders"][k]["correct"]) for k in st.ORDERS}
    unary_fail = [r for r in rows if not ok[1](r)]
    pair_cases = [r for r in rows if not ok[1](r) and ok[2](r)]
    return {
        "n_scenes": len(rows),
        "R_pair": mean(rows, lambda r: not ok[1](r) and ok[2](r)), "R_unary": mean(rows, ok[1]),
        "R_pair_exact": mean(rows, ok[2]), "R_third_exact": mean(rows, ok[3]),
        "pairwise_failures": [r["scene_id"] for r in rows if not ok[2](r)],
        "higher_order_alters_decision": mean(rows, lambda r: r["orders"][3]["S_star_k"] != r["orders"][2]["S_star_k"]),
        "representation_order": dict(Counter(str(max(r["representation_order"].values())) for r in rows)),
        "representation_order_stable": all(len(set(r["representation_order"].values())) == 1 for r in rows),
        "max_significant_order": max((r["max_significant_order"] for r in rows), default=0),
        "max_abs_error": {k: {"mean": mean(rows, lambda r, k=k: r["orders"][k]["max_abs"]),
                              "max": max((r["orders"][k]["max_abs"] for r in rows), default=0.0)} for k in st.ORDERS},
        "third_order_improvement_mean": mean(rows, lambda r: r["orders"][2]["max_abs"] - r["orders"][3]["max_abs"]),
        "decisions_base_fine_stable": all(r["orders"][k]["base_fine_decision_equal"] for r in rows for k in st.ORDERS),
        "incidence": {"physical_beta": mean(rows, lambda r: r["graphs"]["physical"]["n_edges"] > 0),
                      "substitutable_beta": mean(rows, lambda r: any(e[3] == st.SUBSTITUTABLE for e in r["edges"]["geometric"])),
                      "validity_edges": mean(rows, lambda r: r["graphs"]["validity"]["n_edges"] > 0),
                      "choice_edges": mean(rows, lambda r: r["graphs"]["choice"]["n_edges"] > 0)},
        "mean_density": {s: mean(rows, lambda r, s=s: r["graphs"][s]["density"])
                         for s in ("geometric", "physical", "validity", "choice", "combined")},
        "mean_max_degree": {s: mean(rows, lambda r, s=s: r["graphs"][s]["max_degree"]) for s in ("geometric", "combined")},
        "mean_components_combined": mean(rows, lambda r: r["graphs"]["combined"]["n_components"]),
        "geometric_class": dict(Counter(r["geometric_class"] for r in rows)),
        "macros_touched_by_geometric_edges": dict(Counter(str(r["n_macros_touched"]) for r in rows)),
        "validity_binding_rate": mean(rows, lambda r: r["validity_binding"]),
        "distractor_fraction_mean": mean(rows, lambda r: r["distractor_fraction"]),
        "tied_repair_rate": mean(rows, lambda r: r["n_tied"] > 1),
        "unary_failures": {"n": len(unary_fail), "with_repositioning_option": sum(r["repositioning_offered"] for r in unary_fail),
                           "scenes": [r["scene_id"] for r in unary_fail]},
        "R_pair_mechanisms": {
            "n": len(pair_cases),
            "physical_pair_inside_exact_repair": sum(st.PHYSICAL in r["geometric_kinds_inside_exact_repair"] for r in pair_cases),
            "substitutable_pair_touching_exact_repair": sum(st.SUBSTITUTABLE in r["geometric_kinds_touching_exact_repair"]
                                                            for r in pair_cases),
            "validity_binding": sum(r["validity_binding"] for r in pair_cases),
            "choice_edges_present_not_physical_evidence": sum(r["graphs"]["choice"]["n_edges"] > 0 for r in pair_cases)},
        "min_positive_exact_G": min((r["min_positive_G"] for r in rows), default=float("inf")),
    }


def self_checks() -> dict:
    """Script-level checks of the Stage-5 definitions (plan2 authorizes no Stage-5 test file)."""
    star = st.geometric_class(6, [(0, 2), (0, 3), (0, 4)], {0})
    general = st.geometric_class(6, [(0, 2), (3, 4)], {0})
    dense = st.geometric_class(4, [(0, 1), (0, 2), (1, 2), (2, 3)], {0})
    gm = st.graph_metrics(5, [(0, 1), (1, 2)])
    return {"star": star == "macro_star", "general": general == "sparse_general", "dense": dense == "dense",
            "graph_metrics": gm == {"n_edges": 2, "density": 0.2, "max_degree": 2, "n_components": 3},
            "distractor_fraction": st.distractor_fraction(4, frozenset({0b0011, 0b0101})) == 0.25}


def plots(per_family: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, colors = "#fcfcfb", ("#2a78d6", "#eb6834", "#1baf7a", "#7d7a72")
    fams = list(per_family)
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=surface)
    for i, (key, label) in enumerate((("R_unary", "unary exact"), ("R_pair_exact", "pairwise exact"),
                                      ("R_third_exact", "third-order exact"), ("R_pair", r"$R_{pair}$"))):
        ax.bar(x + (i - 1.5) * 0.2, [per_family[f][key] for f in fams], 0.19, color=colors[i], label=label)
    ax.set_xticks(x, [f"{f}\n(n = {per_family[f]['n_scenes']})" for f in fams], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of repairable scenes [-]")
    ax.set_title("Exact-repair recovery by order of G (same exact M, V, unit K), per family", loc="left", fontsize=10)
    ax.legend(ncol=4, fontsize=8, loc="upper center", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "decision_recovery.png", dpi=200, facecolor=surface)
    plt.close(fig)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=surface)
    classes = ("independent", "macro_star", "sparse_general", "dense")
    bottom = np.zeros(len(fams))
    for c, color in zip(classes, colors):
        v = np.array([per_family[f]["geometric_class"].get(c, 0) for f in fams])
        a.bar(x, v, 0.6, bottom=bottom, color=color, label=c)
        bottom += v
    a.set_xticks(x, fams, fontsize=8)
    a.set_ylabel("repairable scenes [count]")
    a.set_title("Geometric (beta^G) interaction graph class", loc="left", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    for i, (src, color) in enumerate(zip(("geometric", "validity", "choice"), colors)):
        b.bar(x + (i - 1) * 0.25, [per_family[f]["mean_density"][src] for f in fams], 0.24, color=color, label=src)
    b.set_xticks(x, fams, fontsize=8)
    b.set_ylabel("mean edge density [-]")
    b.set_title("Edge density by source (G, V and M kept separate)", loc="left", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "graph_structure.png", dpi=200, facecolor=surface)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = ds.load_dataset(ROOT / "s4" / "scenes.jsonl", ds.STAGE4_DIGEST)
    rows = [analyse(r) for r in records if r["intent"] == "repairable"]
    per_family = {f: aggregate([r for r in rows if r["family"] == f]) for f in FAMILIES}
    overall = aggregate(rows)
    every_G = [analyse(r)["min_positive_G"] for r in records if r["intent"] == "negative"]
    stage3 = {}
    s3_path = ROOT / "s3" / "analysis.json"
    if s3_path.exists():
        s3 = json.load(open(s3_path))
        mine = {r["scene_id"]: r for r in rows if r["family"] == "make_space"}
        stage3 = {"compared": True, "R_pair_equal": s3["aggregate"]["R_pair"] == per_family["make_space"]["R_pair"],
                  "per_scene_decisions_equal": all(all(s["orders"][str(k)]["correct"] == mine[s["scene_id"]]["orders"][k]["correct"]
                                                       for k in st.ORDERS) for s in s3["scenes"])}
    out = {
        "stage": 5, "dataset_digest": ds.STAGE4_DIGEST, "n_records": len(records), "n_repairable_analysed": len(rows),
        "excluded": "feasible hard negatives (S* = {{}} trivially); no repairable scene removed",
        "rules": {"stage3_unchanged": st.__doc__.split("PRE-REGISTERED RULES")[1].strip(),
                  "stage5_graph_rules": {"DENSE_DENSITY": st.DENSE_DENSITY, "macro_star": "every geometric edge touches a "
                                         "repositioning option", "validity_binding": "S* changes when V is ignored",
                                         "distractor_fraction": "options in no tied minimal repair"}},
        "surrogate_threshold": {"G_FEAS_TOL": st.G_FEAS_TOL, "min_positive_exact_G_repairable": overall["min_positive_exact_G"],
                                "ratio_to_threshold": overall["min_positive_exact_G"] / st.G_FEAS_TOL,
                                "min_positive_exact_G_all_100": min([overall["min_positive_exact_G"]] + every_G)},
        "distributions_all_100": {k: dict(Counter(str(f(r)) for r in records)) for k, f in (
            ("family", lambda r: r["family"]), ("P", lambda r: r["P"]), ("B0_size", lambda r: r["causal_set_size"]),
            ("S_star_size", lambda r: r["repair_size"]), ("n_tied", lambda r: r["n_tied_repairs"]),
            ("status", lambda r: r["status"]))},
        "overall": overall, "per_family": per_family, "stage3_consistency": stage3, "self_checks": self_checks(),
        "labels_match_stored": all(r["labels_match_stored"] for r in rows), "scenes": rows,
    }
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=lambda v: v.item()))
    plots(per_family)
    brief = {k: overall[k] for k in ("R_pair", "R_unary", "R_pair_exact", "R_third_exact", "pairwise_failures",
                                     "higher_order_alters_decision", "geometric_class", "unary_failures", "R_pair_mechanisms")}
    print(json.dumps({"overall": brief, "surrogate_threshold": out["surrogate_threshold"], "stage3": stage3,
                      "self_checks": out["self_checks"], "labels_match_stored": out["labels_match_stored"]}, indent=1))
    for f in FAMILIES:
        a = per_family[f]
        print(f, {k: a[k] for k in ("n_scenes", "R_pair", "R_unary", "R_pair_exact", "R_third_exact", "geometric_class",
                                    "validity_binding_rate", "incidence")})


if __name__ == "__main__":
    main()
