"""PoC-2 Stage 3: interaction structure and pairwise-necessity analysis (plan2.md section 19).

Run from the repository root (after Stage 2 wrote PoC-2/out/s2/scenes.jsonl):

    PYTHONPATH=PoC-1/src:PoC-2/src python PoC-2/scripts/s3_structure.py

Writes PoC-2/out/s3/analysis.json, pairwise_need.png, interaction_sources.png. The fixed
Stage-2 dataset is loaded (digest-verified), never regenerated; tables are recomputed.
Rules are pre-registered in poc2/structure.py. No QUBO / Hopfield.
"""

from pathlib import Path

import json
import numpy as np

from poc import energy as en
from poc.envelope import ENVELOPE_STEP
from poc2 import dataset as ds
from poc2 import oracle as orc
from poc2 import structure as st
from poc2.scenes import build

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "s3"
FINE = ENVELOPE_STEP / 2


def _sets(table, xs) -> list[list[str]]:
    return sorted(sorted(table.subset(x)) for x in xs)


def analyse(record: dict) -> dict:
    spec = ds.spec_from_json(record["spec"])
    scene, _, options = build(spec)
    tb, tf = orc.repair_table(scene, options), orc.repair_table(scene, options, FINE)
    a_b, a_f = st.compatible_mobius(tb.G, tb.M), st.compatible_mobius(tf.G, tf.M)
    sig = st.significant(a_b, a_f)
    eps = st.eps_repr(tb.G, tf.G, tb.M)
    exact = st.exact_decision(tb)
    size = en.popcount(len(tb.G))
    orders, decisions = {}, {}
    for k in st.ORDERS:
        Gk_b, Gk_f = st.reconstruct(a_b, k), st.reconstruct(a_f, k)
        d_b, d_f = st.order_decision(tb, Gk_b), st.order_decision(tf, Gk_f)
        decisions[k] = d_b
        orders[k] = {**st.representation_errors(tb.G, tb.M, Gk_b), "S_star_k": _sets(tb, d_b),
                     "correct": d_b == exact, "only_correct_repairs": bool(d_b) and d_b <= exact,
                     "base_fine_decision_equal": _sets(tb, d_b) == _sets(tf, d_f)}
    edges = st.interaction_edges(tb, a_b, sig)
    P = len(options)
    geo = edges["geometric"]
    unstable = np.isfinite(a_b) & (np.maximum(np.abs(a_b), np.abs(np.nan_to_num(a_f))) > st.COEF_FLOOR) & ~sig
    unstable[0] = False
    return {
        "scene_id": record["scene_id"], "P": P, "S_star": _sets(tb, exact), "n_tied": len(exact),
        "labels_match_stored": _sets(tb, exact) == record["S_star"],
        "eps_repr": eps,
        "representation_order": {"base": st.representation_order(tb.G, tb.M, a_b, eps),
                                 "fine": st.representation_order(tf.G, tf.M, a_f, eps)},
        "orders": orders,
        "significant_by_order": {str(k): int(np.sum(sig & (size == k))) for k in range(1, int(size.max()) + 1)},
        "n_resolution_unstable": int(unstable.sum()),
        "edges": {"choice": [[options[p].option_id, options[q].option_id] for p, q in edges["choice"]],
                  "validity": [[options[p].option_id, options[q].option_id] for p, q in edges["validity"]],
                  "geometric": [{"pair": [options[p].option_id, options[q].option_id], "beta_G": b, "kind": kind}
                                for p, q, b, kind in geo]},
        "n_physical": sum(kind == st.PHYSICAL for *_, kind in geo),
        "n_substitutable": sum(kind == st.SUBSTITUTABLE for *_, kind in geo),
        "density": {"choice": st.density(len(edges["choice"]), P), "validity": st.density(len(edges["validity"]), P),
                    "geometric": st.density(len(geo), P),
                    "all": st.density(len({tuple(e) for e in edges["choice"] + edges["validity"]}
                                          | {(p, q) for p, q, *_ in geo}), P)},
    }


def rate(rows, key) -> float:
    return float(np.mean([key(r) for r in rows])) if rows else float("nan")


def aggregate(rows: list[dict]) -> dict:
    c = {k: (lambda r, k=k: r["orders"][k]["correct"]) for k in st.ORDERS}
    pair_cases = [r for r in rows if not c[1](r) and c[2](r)]
    return {
        "n_scenes": len(rows),
        "R_pair": rate(rows, lambda r: not c[1](r) and c[2](r)),
        "unary_exact_rate": rate(rows, c[1]), "pairwise_exact_rate": rate(rows, c[2]),
        "third_order_exact_rate": rate(rows, c[3]),
        "higher_order_decision_failure_rate": rate(rows, lambda r: not c[2](r)),
        "third_order_rescues_pairwise_failure": sum(not c[2](r) and c[3](r) for r in rows),
        "unary_only_correct_repairs_rate": rate(rows, lambda r: r["orders"][1]["only_correct_repairs"]),
        "significant_physical_pair_incidence": rate(rows, lambda r: r["n_physical"] > 0),
        "substitutable_pair_incidence": rate(rows, lambda r: r["n_substitutable"] > 0),
        "compatibility_edge_incidence": rate(rows, lambda r: len(r["edges"]["validity"]) > 0),
        "choice_edge_incidence": rate(rows, lambda r: len(r["edges"]["choice"]) > 0),
        "mean_density": {k: rate(rows, lambda r, k=k: r["density"][k]) for k in ("geometric", "validity", "choice", "all")},
        "tied_repair_rate": rate(rows, lambda r: r["n_tied"] > 1),
        "representation_order": {str(k): sum(max(r["representation_order"].values()) == k for r in rows)
                                 for k in range(0, 5)},
        "max_significant_order": max(max((int(o) for o, n in r["significant_by_order"].items() if n), default=0)
                                     for r in rows),
        "R_pair_cases": [{"scene_id": r["scene_id"], "geometric": r["edges"]["geometric"],
                          "validity_edges": len(r["edges"]["validity"])} for r in pair_cases],
        "pairwise_failures": [r["scene_id"] for r in rows if not c[2](r)],
        # descriptive (not a decision rule): which options carry the geometric pair structure
        "geometric_edges_involving_repositioning": rate(
            [g for r in rows for g in r["edges"]["geometric"]], lambda g: any(i.startswith("shift") for i in g["pair"])),
        "unary_failures_with_repositioning_offered": rate(
            [r for r in rows if not c[1](r)], lambda r: any(e for e in r["edges"]["geometric"])),
        "base_fine_decisions_stable": all(r["orders"][k]["base_fine_decision_equal"] for r in rows for k in st.ORDERS),
        "representation_order_stable": all(len(set(r["representation_order"].values())) == 1 for r in rows),
        "labels_match_stored": all(r["labels_match_stored"] for r in rows),
    }


def plots(agg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, surface, colors = "#141413", "#898781", "#fcfcfb", ("#2a78d6", "#eb6834", "#1baf7a", "#7d7a72")
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=surface)
    labels = ["unary $S^*_1$", "pairwise $S^*_2$", "third order $S^*_3$", r"$R_{pair}$"]
    vals = [agg["unary_exact_rate"], agg["pairwise_exact_rate"], agg["third_order_exact_rate"], agg["R_pair"]]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", color=ink)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of the 30 repairable scenes [-]")
    ax.set_title("Exact-repair recovery by approximation order of G (same exact M, V, K)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "pairwise_need.png", dpi=200, facecolor=surface)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=surface)
    names = ["physical $\\beta^G$", "substitutable $\\beta^G$", "static compatibility (V)", "choice constraint (M)"]
    vals = [agg["significant_physical_pair_incidence"], agg["substitutable_pair_incidence"],
            agg["compatibility_edge_incidence"], agg["choice_edge_incidence"]]
    bars = ax.bar(names, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", color=ink)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of scenes with >= 1 such edge [-]")
    ax.set_title("Pairwise interaction sources, reported separately", loc="left", fontsize=10)
    ax.tick_params(axis="x", labelsize=8, colors=muted)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "interaction_sources.png", dpi=200, facecolor=surface)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = ds.load_dataset(ROOT / "s2" / "scenes.jsonl")
    repairable = [r for r in records if r["intent"] == "repairable"]
    rows = [analyse(r) for r in repairable]
    agg = aggregate(rows)
    out = {"stage": 3, "dataset_digest": ds.STAGE2_DIGEST, "n_records": len(records),
           "n_repairable_analysed": len(rows), "excluded": "feasible hard negatives (S* = {{}} trivially); no "
           "repairable scene was removed", "preregistered": st.__doc__.split("PRE-REGISTERED RULES")[1].strip(),
           "aggregate": agg, "scenes": rows}
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=lambda v: v.item()))
    plots(agg)
    print(json.dumps({k: v for k, v in agg.items() if k not in ("R_pair_cases",)}, indent=1))
    print("R_pair cases:", json.dumps(agg["R_pair_cases"]))


if __name__ == "__main__":
    main()
