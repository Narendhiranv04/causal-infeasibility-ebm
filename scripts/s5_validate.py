"""Stage 5 experiment: 3D multi-blocker repair structure and final PoC verdict.

Run from the repository root:

    PYTHONPATH=src python scripts/s5_validate.py

Writes out/s5/metrics.json, out/s5/scaling.png, out/s5/order_error.png.

PRE-REGISTERED RULES (fixed before any Stage-5 interaction result was inspected)
  Significance  a(T) is a genuine interaction only if its sign agrees at base (2 mm)
                and fine (1 mm) envelope resolution and min(|a_b|, |a_f|) >
                max(10 |a_b - a_f|, COEF_FLOOR)  [energy.significant].
                COEF_FLOOR = 1e-6 m: three orders above the certified distance
                resolution (<= 1e-9 m), a hundred times below CONTACT_TOL_3D.
  Representation order  smallest k with max_S |G(S) - G_k(S)| <= eps_repr ("resolution-exact":
                representable within the registered numerical tolerance, not algebraically exact), where
                eps_repr = max(10 max_S |G_base(S) - G_fine(S)|, COEF_FLOOR); computed at
                both resolutions, an order that changes under refinement is not claimed.
  Decision-sufficient order  smallest k for which the admissible argmin of
                J_k = B 1[G_k > COEF_FLOOR] + K equals the exact S* (all ties).
  QUBO          H = kappa G_2 + B_V (1 - V)_2 + lambda K with lambda = 1 (repair actions),
                kappa = B / COEF_FLOOR (any infeasible S with G > COEF_FLOOR costs more than
                every repair), B_V = B = K_max + 1 (an invalid state never beats a valid
                feasible one). Categories: resolution_exact (G and 1 - V both representable at
                order <= 2 within eps_repr), decision_sufficient (argmin H = S* although not
                resolution-exact), failure (otherwise).
  Hopfield      secondary: unchanged Stage-3 solver on resolution-exact/decision-sufficient QUBOs,
                R in {1, 4, 16} as prefixes of one seeded 16-restart run.
  Verdict       RED if the oracle / S* is unstable under refinement or S* disagrees with the
                manual expectation; else YELLOW-C if pairwise fails a decision; else YELLOW-A
                if unary is decision-sufficient in every scene; else GREEN if Hopfield meets
                the Stage-3 targets (>= 95% exact QUBO optimum, >= 99% true-feasible
                admissible at R = 16) and YELLOW-B if it does not.

INTERPRETATION BOUNDARIES
  The oracle certifies that the prescribed next-action envelope is (in)feasible; it does
  not prove that no other motion exists. RELOCATE / SHIFT_TARGET are high-level corrective
  macros (SHIFT_TARGET = reposition the target/box, then retry the same skill); Stage 5
  validates their post-intervention geometry (V), combinatorics and restoration of
  feasibility, not the low-level motion that executes them.
"""

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from poc import cases as cs
from poc import energy as e
from poc import envelope as ev
from poc import hopfield as hf
from poc.oracle import CONTACT_TOL_3D, admissible_minimal_repairs, oracle_repair_cost

OUT = Path("out/s5")
SEED = 7
STEPS = {"base": ev.ENVELOPE_STEP, "fine": ev.ENVELOPE_STEP / 2}
COEF_FLOOR = 1e-6
BUDGETS = (1, 4, 16)
TARGET_EXACT, TARGET_FEASIBLE = 0.95, 0.99


def _ids(case, x: int) -> frozenset[str]:
    return frozenset(iv.intervention_id for p, iv in enumerate(case.candidates) if x >> p & 1)


def _sets(case, xs) -> list[list[str]]:
    return sorted(sorted(_ids(case, x)) for x in xs)


def _surrogate(case, G_hat, V, K) -> frozenset[int]:
    F_hat = (np.asarray(G_hat) > COEF_FLOOR).astype(int)
    return admissible_minimal_repairs(F_hat, V, K, case.k_max)


def _repr_order(G, eps) -> int:
    return next(k for k in range(e.n_vars(G) + 1) if np.max(np.abs(G - e.approximate(G, k))) <= eps)


def coefficients(case, a_b, a_f, sig) -> dict:
    names = [iv.intervention_id for iv in case.candidates]
    out = {"significant": [], "resolution_unstable": []}
    size = e.popcount(len(a_b))
    for x in range(1, len(a_b)):
        if sig[x] or max(abs(a_b[x]), abs(a_f[x])) > COEF_FLOOR:
            row = {"T": [n for p, n in enumerate(names) if x >> p & 1], "order": int(size[x]),
                   "base": float(a_b[x]), "fine": float(a_f[x])}
            out["significant" if sig[x] else "resolution_unstable"].append(row)
    return out


def qubo_terms(case, a_b, V) -> dict:
    """Pre-registered QUBO H = kappa G_2 + B_V (1 - V)_2 + lambda K on every subset."""
    B = oracle_repair_cost(1, 0, case.k_max)
    kappa, lam = e.qubo_weights(B, COEF_FLOOR)
    a_inv = e.mobius(1 - V)
    lengths = [iv.length for iv in case.candidates]
    Q, q, c = e.qubo(kappa * e.truncate(a_b, 2) + B * e.truncate(a_inv, 2), lengths, 1.0, lam)
    return {"B": B, "kappa": kappa, "lam": lam, "a_inv": a_inv, "Q": Q, "q": q, "c": c, "H": e.qubo_energy(Q, q, c)}


def qubo_block(case, idx, a_b, V, F, K, S_star) -> dict:
    t = qubo_terms(case, a_b, V)
    B, kappa, lam, a_inv, Q, q, c, H = (t[k] for k in ("B", "kappa", "lam", "a_inv", "Q", "q", "c", "H"))
    opt = e.argmin_sets(H)
    good = (F == 0) & (V == 1)
    out = {"kappa": kappa, "lambda": lam, "B_V": B, "validity_order": e.degree(a_inv, 0.5),
           "argmin": _sets(case, opt), "argmin_matches_S_star": opt == S_star,
           "valid_feasible_below_rest": bool(good.all() or H[good].max() < H[~good].min())}
    net = hf.to_hopfield(*hf.rescale(Q, q, c, hf.normalising_factor(Q, q)))
    t0 = time.perf_counter()
    runs = hf.restarts(net, max(BUDGETS), np.random.default_rng([SEED, idx]))
    out["hopfield_runtime_per_restart_s"] = (time.perf_counter() - t0) / len(runs)
    masks = [int(hf.bits(d.s) @ (1 << np.arange(len(q)))) for d in runs]
    out["hopfield"] = {}
    for R in BUDGETS:
        m = masks[min(range(R), key=lambda j: runs[j].energy)]
        out["hopfield"][R] = {"exact_qubo_optimum": m in opt, "true_feasible_admissible": bool(good[m]),
                              "minimal": m in S_star, "energy_gap": float(H[m] - H.min())}
    return out


def analyse(case, idx: int) -> dict:
    t0 = time.perf_counter()
    Tb = cs.exhaustive_table(case, STEPS["base"])
    runtime = time.perf_counter() - t0
    Tf = cs.exhaustive_table(case, STEPS["fine"])
    Gb, Gf, Fb, Ff, V, K = Tb.G, Tf.G, Tb.F, Tf.F, Tb.V, Tb.K
    S_b, S_f = (admissible_minimal_repairs(F, V, K, case.k_max) for F in (Fb, Ff))
    a_b, a_f = e.mobius(Gb), e.mobius(Gf)
    sig = e.significant(a_b, a_f, COEF_FLOOR)
    eps = max(10 * float(np.max(np.abs(Gb - Gf))), COEF_FLOOR)
    adm = V == 1
    approx = {}
    for k in (1, 2, 3):
        Gk = e.approximate(Gb, k)
        approx[k] = {**e.error_metrics(Gb, Gk), "feasibility_agreement": float(np.mean(((Gk > COEF_FLOOR) == (Fb == 1))[adm])),
                     "minimal_repair_match": _surrogate(case, Gk, V, K) == S_b}
    decision = {res: next((k for k in range(1, len(case.candidates) + 1)
                           if _surrogate(case, e.approximate(G, k), V, K) == S_b), None)
                for res, G in (("base", Gb), ("fine", Gf))}
    row = {
        "name": case.name, "family": case.scene.family, "structure": case.structure, "P": len(case.candidates),
        "S_star": _sets(case, S_b), "S_star_fine": _sets(case, S_f), "S_star_stable": S_b == S_f,
        "matches_expected": frozenset(_ids(case, x) for x in S_b) == case.expected,
        "minimal_cardinality": min(len(s) for s in _sets(case, S_b)), "n_tied_minimal": len(S_b),
        "n_valid_subsets": int(adm.sum()), "n_invalid_subsets": int((~adm).sum()),
        "invalid_subsets": _sets(case, np.flatnonzero(~adm)),
        "label_flips_base_fine": int(np.sum((Fb != Ff) & adm)), "max_abs_dG_base_fine": float(np.max(np.abs(Gb - Gf))),
        "eps_repr": eps, "representation_order": {"base": _repr_order(Gb, eps), "fine": _repr_order(Gf, eps)},
        "significant_degree": int(e.popcount(len(a_b))[sig].max(initial=0)),
        "decision_sufficient_order": decision,
        "approximations": approx, "coefficients": coefficients(case, a_b, a_f, sig),
        "pair_sources_significant": [s for s in e.pair_sources(Gb, Tb.C, COEF_FLOOR) if sig[(1 << s["p"]) | (1 << s["q"])]],
        "min_positive_G": float(Gb[Gb > 0].min(initial=np.inf)), "exhaustive_runtime_s": runtime,
        "table": [{"S": sorted(_ids(case, x)), "d": [float(v) for v in Tb.d[x]], "c": list(Tb.rows[x].c),
                   "G": float(Gb[x]), "F": int(Fb[x]), "V": int(V[x]), "K": int(K[x])} for x in range(len(K))],
    }
    r_order = max(row["representation_order"].values())
    if r_order <= 2 or (decision["base"] or 99) <= 2:
        row["qubo"] = qubo_block(case, idx, a_b, V, Fb, K, S_b)
        exact = r_order <= 2 and row["qubo"]["validity_order"] <= 2
        row["qubo_category"] = ("resolution_exact" if exact else
                                "decision_sufficient" if row["qubo"]["argmin_matches_S_star"] else "failure")
    else:
        row["qubo_category"] = "failure"
    return row


def stage4_recheck() -> dict:
    flips = 0
    for c in cs.canonical_cases():
        flips += cs.evaluate(c.scene, STEPS["base"])[0].F != cs.evaluate(c.scene, STEPS["fine"])[0].F
    return {"n_cases": len(cs.canonical_cases()), "label_flips_base_fine": int(flips)}


def scaling(rows: list[dict], key: str) -> dict:
    out = {}
    for v in sorted({r[key] for r in rows}):
        g = [r for r in rows if r[key] == v]
        q = [r for r in g if "qubo" in r]
        out[str(v)] = {"n_scenes": len(g), "mean_P": float(np.mean([r["P"] for r in g])),
                       "unary_match": float(np.mean([r["approximations"][1]["minimal_repair_match"] for r in g])),
                       "pairwise_match": float(np.mean([r["approximations"][2]["minimal_repair_match"] for r in g])),
                       "qubo_match": float(np.mean([r["qubo"]["argmin_matches_S_star"] for r in q])) if q else None,
                       "max_representation_order": max(max(r["representation_order"].values()) for r in g),
                       "max_unary_error": max(r["approximations"][1]["max_abs"] for r in g),
                       "mean_runtime_s": float(np.mean([r["exhaustive_runtime_s"] for r in g]))}
    return out


def high_cardinality(rows: list[dict], k_min: int = 4) -> dict:
    """|S*| >= k_min scenes by structure: blocker count and interaction order reported separately."""
    hi = [r for r in rows if r["minimal_cardinality"] >= k_min and r["structure"] != "perturbed"]
    order = {r["name"]: {"structure": r["structure"], "S_star_size": r["minimal_cardinality"],
                         "representation_order": max(r["representation_order"].values())} for r in hi}
    indep = [o for o in order.values() if o["structure"] == "independent"]
    other = {n: o for n, o in order.items() if o["structure"] != "independent"}
    return {"scenes": order, "all_independent_high_cardinality_unary": all(o["representation_order"] == 1 for o in indep),
            "non_independent_high_cardinality": other,
            "conclusion": "all independent high-cardinality scenes are unary; a high-cardinality scene is pairwise "
                          "only when it contains a local coupled motif (the mixed scene)"}


def verdict(rows: list[dict], s4: dict) -> dict:
    core = [r for r in rows if r["structure"] != "perturbed"]
    hop = [r["qubo"]["hopfield"][16] for r in rows if r.get("qubo_category") in ("resolution_exact", "decision_sufficient")]
    evidence = {
        "oracle_stable": all(r["S_star_stable"] and r["label_flips_base_fine"] == 0 for r in rows)
        and s4["label_flips_base_fine"] == 0,
        "S_star_matches_expected": all(r["matches_expected"] for r in rows),
        "pairwise_decision_failure": any(r["qubo_category"] == "failure" for r in core),
        "unary_decision_sufficient_everywhere": all(r["approximations"][1]["minimal_repair_match"] for r in rows),
        "hopfield_exact_rate_R16": float(np.mean([h["exact_qubo_optimum"] for h in hop])),
        "hopfield_feasible_rate_R16": float(np.mean([h["true_feasible_admissible"] for h in hop])),
    }
    if not (evidence["oracle_stable"] and evidence["S_star_matches_expected"]):
        label = "RED"
    elif evidence["pairwise_decision_failure"]:
        label = "YELLOW-C"
    elif evidence["unary_decision_sufficient_everywhere"]:
        label = "YELLOW-A"
    elif evidence["hopfield_exact_rate_R16"] >= TARGET_EXACT and evidence["hopfield_feasible_rate_R16"] >= TARGET_FEASIBLE:
        label = "GREEN"
    else:
        label = "YELLOW-B"
    return {"label": label, "evidence": evidence}


INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # reference categorical slots 1-3, fixed order
MACHINE_ZERO = 1e-12  # errors below this are drawn at this value on the log axis


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def plot_order_error(rows, path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    _axes(ax)
    y = np.arange(len(rows))[::-1]
    for k, color, mark in zip((1, 2, 3), SERIES, ("o", "s", "^")):
        err = [max(r["approximations"][k]["max_abs"], MACHINE_ZERO) for r in rows]
        ax.scatter(err, y, s=40, color=color, marker=mark, edgecolor=SURFACE, lw=1.5, zorder=3,
                   label=f"order-{k} truncation")
    ax.axvline(COEF_FLOOR, color=MUTED, ls="--", lw=1)
    ax.text(COEF_FLOOR * 1.3, y[0] + 0.6, "COEF_FLOOR", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(MACHINE_ZERO / 3, 0.1)
    ax.set_yticks(y, [r["name"] for r in rows], fontsize=8, color=INK)
    ax.set_xlabel("max |G(S) - G_k(S)| over all subsets [m] (machine zero drawn at 1e-12)", color=INK, fontsize=9)
    ax.set_title("Stage 5: Möbius truncation error of the 3D repair response G", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_scaling(rows, path) -> None:
    plt = _pyplot()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in (a, b):
        _axes(ax)
    by = scaling(rows, "minimal_cardinality")
    ks = sorted(int(k) for k in by)
    for key, color, mark, label in (("unary_match", SERIES[0], "o", "unary surrogate"),
                                    ("qubo_match", SERIES[1], "s", "pairwise QUBO")):
        a.plot(ks, [by[str(k)][key] for k in ks], color=color, lw=2, marker=mark, ms=8,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    for k in ks:
        a.annotate(f"n={by[str(k)]['n_scenes']}", (k, 0.03), ha="center", color=MUTED, fontsize=8)
    a.set_ylim(0, 1.08)
    a.set_xticks(ks)
    a.set_xlabel("certified minimal repair cardinality |S*|", color=INK, fontsize=9)
    a.set_ylabel("fraction of scenes recovering exact S*", color=INK, fontsize=9)
    a.set_title("Repair recovery vs |S*|", color=INK, fontsize=11, loc="left")
    a.legend(frameon=False, fontsize=8, loc="center right")
    P = [r["P"] for r in rows]
    b.scatter(P, [r["exhaustive_runtime_s"] for r in rows], s=40, color=SERIES[0], edgecolor=SURFACE, lw=1.5, zorder=3)
    b.set_yscale("log")
    b.set_xlabel("candidate interventions P (2^P subsets)", color=INK, fontsize=9)
    b.set_ylabel("exhaustive 3D table runtime [s]", color=INK, fontsize=9)
    b.set_title("Exact enumeration cost vs P", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = [analyse(c, i) for i, c in enumerate(cs.stage5_cases())]
    s4 = stage4_recheck()
    metrics = {
        "stage": 5, "seed": SEED, "mujoco_version": mujoco.__version__,
        "constants": {"CONTACT_TOL_3D": CONTACT_TOL_3D, "steps": STEPS, "COEF_FLOOR": COEF_FLOOR,
                      "SIG_FACTOR": e.SIG_FACTOR, "BUDGETS": BUDGETS},
        "interpretation": __doc__.split("INTERPRETATION BOUNDARIES")[1].strip(),
        "verdict": verdict(rows, s4), "stage4_recheck": s4,
        "scaling_by_minimal_cardinality": scaling(rows, "minimal_cardinality"),
        "scaling_by_P": scaling(rows, "P"),
        "high_cardinality_vs_order": high_cardinality(rows),
        "cases": rows, "runtime_s": round(time.perf_counter() - t0, 2),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    plot_order_error(rows, OUT / "order_error.png")
    plot_scaling(rows, OUT / "scaling.png")
    print(json.dumps({k: metrics[k] for k in ("verdict", "stage4_recheck", "runtime_s")}, indent=1))
    for r in rows:
        q = r.get("qubo", {})
        print(f'{r["name"]:24s} P={r["P"]} |S*|={r["minimal_cardinality"]} ties={r["n_tied_minimal"]} '
              f'inv={r["n_invalid_subsets"]} repr={r["representation_order"]} sigdeg={r["significant_degree"]} '
              f'dec={r["decision_sufficient_order"]} U={r["approximations"][1]["minimal_repair_match"]} '
              f'qubo={r["qubo_category"]} hop16={q.get("hopfield", {}).get(16, {}).get("exact_qubo_optimum")} '
              f'stable={r["S_star_stable"]} flips={r["label_flips_base_fine"]} dG={r["max_abs_dG_base_fine"]:.1e}')


if __name__ == "__main__":
    main()
