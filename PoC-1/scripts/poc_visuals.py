"""PoC freeze: visual diagnostics of the validated Stage 1-5 pipeline (visualization only).

Run from the repository root:

    PYTHONPATH=src python scripts/poc_visuals.py

Writes PNGs to out/visuals/. Every number is produced by the existing APIs
(cases, envelope, energy, hopfield) and by the Stage-5 analysis itself
(scripts/s5_validate.analyse / qubo_terms); no geometry or oracle logic lives here.
Views are orthographic projections of the axis-aligned box scenes: top view (x-y)
for the cupboard, side view (x-z) for the hinged lid.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s5_validate as s5  # noqa: E402

from poc import cases as cs  # noqa: E402
from poc import energy as e  # noqa: E402
from poc import envelope as ev  # noqa: E402
from poc import hopfield as hf  # noqa: E402
from poc.oracle import CONTACT_TOL_3D, admissible_minimal_repairs  # noqa: E402

OUT = Path("out/visuals")
DPI = 220
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"
STRUCT = "#cfcdc4"   # structural bodies
STRUCTURE_COLOR = {"independent": BLUE, "coupled": ORANGE, "substitutable": AQUA, "mixed": MAGENTA, "validity": VIOLET}
FAMILY_MARKER = {"insertion": "o", "extraction": "s", "hinge": "^"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": MUTED,
                     "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points (rendering only: outline of a projected box)."""
    pts = sorted(set(map(tuple, np.round(points, 12))))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


# ------------------------------------------------------------ shared data

CASES = {c.name: c for c in cs.stage5_cases()}
ROWS = {c.name: s5.analyse(c, i) for i, c in enumerate(CASES.values())}


def s_star(name: str) -> list:
    """First certified minimal repair (as interventions) from the Stage-5 analysis."""
    ids = ROWS[name]["S_star"][0]
    return [iv for iv in CASES[name].candidates if iv.intervention_id in ids]


# ------------------------------------------------------ 1. scene envelopes

def _draw_scene(ax, scene, sweep, a, axes_xy, roles, show_samples, highlight=None) -> None:
    i, j = axes_xy
    for ent in scene.entities:
        kind = roles.get(ent.eid, "structural" if not ent.entity.movable else "other")
        face, edge, ls = {"structural": (STRUCT, MUTED, "-"), "cause": (RED, RED, "-"), "coupled": (ORANGE, ORANGE, "-"),
                          "former": (STRUCT if not ent.entity.movable else "none", RED, "--"),
                          "distractor": ("none", INK2, "--"), "other": ("none", MUTED, ":"),
                          "moved": (AQUA, AQUA, "-")}[kind]
        for box in ent.boxes:
            ax.add_patch(Polygon(hull(box.corners()[:, [i, j]]), fc=face, ec=edge, ls=ls, lw=1.2 if kind == "former" else 1.0,
                                 alpha=(0.3 if face == STRUCT else 0.55) if face != "none" else 1.0))
    for box in scene.fixture:
        ax.add_patch(Polygon(hull(box.corners()[:, [i, j]]), fc="none", ec=INK, lw=1.2, hatch="////", alpha=0.6))
    taus, frames = ev.poses(scene.motion, scene.moving, ev.ENVELOPE_STEP)
    hit = (sweep.distances < -CONTACT_TOL_3D).any(axis=1)
    picks = np.unique(np.r_[np.linspace(0, len(frames) - 1, 14).astype(int), np.flatnonzero(hit)[::6]])
    for k in (picks if show_samples else [0, len(frames) - 1]):
        corners = ev.world_corners(scene.moving, *frames[k])
        for part in np.split(corners, len(scene.moving.parts)):
            color = RED if hit[k] else BLUE
            ax.add_patch(Polygon(hull(part[:, [i, j]]), fc=color, ec=color, lw=0.5, alpha=0.10 if show_samples else 0.0))
            if not show_samples or k in (0, len(frames) - 1) or (highlight is not None and k == highlight):
                ax.add_patch(Polygon(hull(part[:, [i, j]]), fc="none", ec=RED if hit[k] else BLUE,
                                     lw=2.4 if k == highlight or k in (0, len(frames) - 1) else 1.0))
    ax.set_aspect("equal")
    ax.autoscale_view()


def fig_scene_envelopes() -> None:
    specs = [("S5_ins_coupled", (0, 1), "top view"), ("S5_ext_k4", (0, 1), "top view"), ("S5_hinge_coupled", (0, 2), "side view")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for r, (name, xy, view) in enumerate(specs):
        case, scene = CASES[name], CASES[name].scene
        a0, sw0 = cs.evaluate(scene)
        fixed = cs.do(scene, s_star(name))
        a1, sw1 = cs.evaluate(fixed)
        repaired_ids = {iv.entity_id for iv in s_star(name)}
        cand_ids = {iv.entity_id for iv in case.candidates}
        roles = {i: "cause" for i in a0.blockers}
        roles |= {i: "coupled" for i in repaired_ids if i not in a0.blockers and i != scene.action.target_id}
        roles |= {i: "distractor" for i in cand_ids - repaired_ids - {scene.action.target_id}}
        k_hit = int(np.argmin(sw0.distances.min(axis=1)))
        _draw_scene(axes[r, 0], scene, sw0, a0, xy, roles, False)
        _draw_scene(axes[r, 1], scene, sw0, a0, xy, roles, True, highlight=k_hit)
        after = {i: "former" for i in a0.blockers} | {i: "moved" for i in repaired_ids} | {
            i: "distractor" for i in cand_ids - repaired_ids - {scene.action.target_id}}
        _draw_scene(axes[r, 2], fixed, sw1, a1, xy, after, True)
        lbl = ("x [m]", "y [m]" if xy[1] == 1 else "z [m]")
        for c, title in enumerate(("initial scene: start and goal poses", "action envelope (sampled poses)",
                                   f"after certified repair {sorted(iv.intervention_id for iv in s_star(name))}")):
            axes[r, c].set_title(f"{name} ({view}) - {title}", loc="left")
            axes[r, c].set_xlabel(lbl[0])
            axes[r, c].set_ylabel(lbl[1])
        if name == "S5_hinge_coupled":
            t_hit = sw0.taus[k_hit]
            axes[r, 1].set_title(f"{name} (side view) - action envelope\n"
                                 f"tau = 0, 1 clear ({1e3 * sw0.distances[0].min():.0f} / {1e3 * sw0.distances[-1].min():.0f} mm); "
                                 f"tau = {t_hit:.2f} penetrates shelf {-1e3 * sw0.distances[k_hit].min():.1f} mm", loc="left")
    handles = [Patch(fc=STRUCT, ec=MUTED, label="structural"), Patch(fc=RED, alpha=0.55, label="causal blocker (c > 0)"),
               Patch(fc=ORANGE, alpha=0.55, label="entity that blocks only after the repositioning macro"),
               Patch(fc="none", ec=INK2, ls="--", label="distractor candidate"), Patch(fc=AQUA, alpha=0.55, label="relocated by repair"),
               Patch(fc="none", ec=RED, ls="--", label="original cause, now clear"),
               Patch(fc="none", ec=INK, hatch="////", label="box body (target fixture)"),
               Patch(fc=BLUE, alpha=0.3, label="moving composite, clear sample"), Patch(fc=RED, alpha=0.3, label="colliding sample")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    save(fig, "scene_envelopes.png")


# ------------------------------------------------------ 2. conflict vs tau

def fig_conflict_vs_tau() -> None:
    specs = [("S5_ins_coupled", ("jamb", "nb", "d1")), ("S5_hinge_coupled", ("shelf", "bt", "d1"))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    for r, (name, ents) in enumerate(specs):
        case, macro = CASES[name], s_star(name)[0]
        states = [("initial scene", case.scene), (f"after macro only [{macro.intervention_id}]", cs.do(case.scene, [macro])),
                  (f"after certified repair {sorted(iv.intervention_id for iv in s_star(name))}", cs.do(case.scene, s_star(name)))]
        for c, (title, scene) in enumerate(states):
            ax = axes[r, c]
            a, sw = cs.evaluate(scene)
            for color, eid in zip((BLUE, ORANGE, AQUA), ents):
                k = a.ids.index(eid)
                d = 1e3 * sw.distances[:, k]
                ax.plot(sw.taus, d, color=color, lw=2, label=f"{eid} (min {d.min():.1f} mm)")
                m = int(np.argmin(d))
                ax.plot(sw.taus[m], d[m], "o", ms=7, color=color, mec=SURFACE, mew=1.5)
            ax.axhline(0.0, color=INK2, lw=1)
            ax.axhline(-1e3 * CONTACT_TOL_3D, color=RED, lw=1, ls="--")
            ax.set_ylim(-40, 60)
            ax.set_title(f"{name}: {title}  (F = {a.F})", loc="left")
            ax.set_xlabel("tau (normalised action progress) [-]")
            ax.legend(loc="lower left", title="entity (dot = min over tau)", fontsize=8, title_fontsize=8)
            if c == 0:
                ax.set_ylabel("signed distance d_i(tau) [mm]  (view clipped to [-40, 60])")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.01, "Reference lines: grey solid d = 0 (touching); red dashed d = -CONTACT_TOL_3D = -0.1 mm "
             "(conflict threshold; the two coincide at this scale). Endpoints tau = 0 and tau = 1 are the start/goal poses.",
             color=INK2, fontsize=8)
    save(fig, "conflict_vs_tau.png")


# ------------------------------------------------------ 3. interaction graphs

def fig_interaction_graphs() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, name in zip(axes, ("S5_ins_coupled", "S5_hinge_coupled", "S5_ins_mixed")):
        row, case = ROWS[name], CASES[name]
        names = [iv.intervention_id for iv in case.candidates]
        in_star = set(row["S_star"][0])
        pairs = [c["T"] for c in row["coefficients"]["significant"] if c["order"] == 2]
        layout = [n for n in names if not any(n == p[1] for p in pairs)]  # partners placed opposite their hub
        for p, q in pairs:
            layout.insert(len(names) // 2, q)
        ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(names), endpoint=False)
        pos = {n: (np.cos(t), np.sin(t)) for n, t in zip(layout, ang)}
        alpha = {c["T"][0]: c["base"] for c in row["coefficients"]["significant"] if c["order"] == 1}
        kinds = {frozenset((names[s["p"]], names[s["q"]])): {x["kind"] for x in s["entities"]}
                 for s in row["pair_sources_significant"]}
        for coef in (c for c in row["coefficients"]["significant"] if c["order"] == 2):
            (p, q), kind = coef["T"], kinds.get(frozenset(coef["T"]), set())
            color, ls = (ORANGE, "-") if "physical" in kind else (AQUA, "--")
            ax.plot(*zip(pos[p], pos[q]), color=color, ls=ls, lw=3, zorder=1)
            mid = np.mean([pos[p], pos[q]], axis=0)
            ax.annotate(f"beta = {1e3 * coef['base']:+.1f} mm", mid + np.array([0.0, 0.12]), ha="center", va="center", color=color,
                        bbox={"fc": SURFACE, "ec": "none", "pad": 1})
        for n, (x, y) in pos.items():
            ax.scatter([x], [y], s=900, color=BLUE if n in in_star else SURFACE, ec=BLUE, lw=1.5, zorder=2)
            a_txt = f"alpha = {1e3 * alpha[n]:+.1f} mm" if n in alpha else "alpha ~ 0 (not significant)"
            ax.annotate(f"{n}\n{a_txt}", (x * 1.62, y * 1.62), ha="center", va="center", fontsize=8, color=INK)
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-1.95, 1.95)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{name}: P = {len(names)}, |S*| = {row['minimal_cardinality']}, "
                     f"{sum(c['order'] == 2 for c in row['coefficients']['significant'])} significant pair term(s)", loc="left")
    handles = [Line2D([], [], color=ORANGE, lw=3, label="significant beta, pair_sources() label 'physical' (diagnostic heuristic)"),
               Line2D([], [], color=AQUA, lw=3, ls="--", label="significant beta, label 'substitutable'"),
               Line2D([], [], marker="o", ls="", ms=12, color=BLUE, label="intervention in certified S*"),
               Line2D([], [], marker="o", ls="", ms=12, mfc=SURFACE, mec=BLUE, label="candidate not in S*")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Möbius interaction graphs (edges only for coefficients passing the pre-registered base/fine significance rule; "
                 "alpha, beta in metres of summed conflict G)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "interaction_graphs.png")


# ------------------------------------------------------ 4. cardinality vs order

def fig_cardinality_vs_order() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    core = [r for r in ROWS.values() if r["structure"] != "perturbed"]
    for r in core:
        dx = {"insertion": -0.14, "extraction": 0.0, "hinge": 0.14}[r["family"]]
        dy = -0.12 if r["structure"] == "validity" else 0.0
        y = max(r["representation_order"].values())
        ax.scatter(r["minimal_cardinality"] + dx, y + dy, s=90, marker=FAMILY_MARKER[r["family"]],
                   color=STRUCTURE_COLOR[r["structure"]], ec=SURFACE, lw=1.2, zorder=3)
    mixed = ROWS["S5_ins_mixed"]
    ax.annotate("S5_ins_mixed: |S*| = 5 but order 2\n(one local coupled motif inside a 5-action repair)",
                (5 - 0.14, 2), (3.1, 2.6), arrowprops={"arrowstyle": "->", "color": INK2}, color=INK2)
    ax.axhspan(0.5, 1.5, color=BLUE, alpha=0.04)
    ax.set_xticks(range(1, 7))
    ax.set_yticks([0, 1, 2, 3], ["0", "1 (unary)", "2 (pairwise)", "3"])
    ax.set_ylim(-0.3, 3.2)
    ax.set_xlabel("certified minimal repair cardinality |S*| [interventions]")
    ax.set_ylabel("representation order (resolution-exact) [-]")
    ax.set_title("Blocker cardinality vs interaction order (4 perturbation copies of S5_ins_k3 excluded)", loc="left")
    handles = [Line2D([], [], marker=m, ls="", color=INK2, label=f) for f, m in FAMILY_MARKER.items()]
    handles += [Line2D([], [], marker="o", ls="", color=c, label=s + (" (drawn 0.12 below its order)" if s == "validity" else ""))
                for s, c in STRUCTURE_COLOR.items()]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=8)
    assert mixed["minimal_cardinality"] == 5
    save(fig, "cardinality_vs_order.png")


# ------------------------------------------------------ 5. coefficient stability

def fig_coefficient_stability() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7))
    inset = ax.inset_axes([0.58, 0.07, 0.38, 0.38])
    for axis in (ax, inset):
        for r in ROWS.values():
            for key, filled in (("significant", True), ("resolution_unstable", False)):
                for c in r["coefficients"][key]:
                    m = {1: "o", 2: "s"}.get(c["order"], "^")
                    axis.scatter(1e3 * c["base"], 1e3 * c["fine"], marker=m, s=46 if filled else 70,
                                 color=BLUE if filled else "none", ec=BLUE if filled else RED,
                                 alpha=0.85 if filled else 1.0, lw=1.4, zorder=3)
    lim = 40
    for axis, lo_hi in ((ax, (-lim, lim)), (inset, (-1.5, 1.5))):
        axis.plot(lo_hi, lo_hi, color=MUTED, lw=1, ls="--", zorder=1)
        axis.set_xlim(*lo_hi)
        axis.set_ylim(*lo_hi)
    inset.set_title("zoom |a| < 1.5 mm", fontsize=8)
    inset.tick_params(labelsize=7)
    ax.indicate_inset_zoom(inset, edgecolor=MUTED)
    ax.set_xlabel("Möbius coefficient at base resolution (2 mm) [mm]")
    ax.set_ylabel("same coefficient at fine resolution (1 mm) [mm]")
    ax.set_title("Coefficient stability under envelope refinement (all 22 Stage-5 scenes)", loc="left")
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, label="unary, significant"),
               Line2D([], [], marker="s", ls="", color=BLUE, label="pairwise, significant"),
               Line2D([], [], marker="^", ls="", color=BLUE, label="order >= 3, significant (none)"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=RED, label="rejected: resolution-unstable"),
               Line2D([], [], color=MUTED, ls="--", label="y = x")]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    save(fig, "coefficient_stability.png")


# ------------------------------------------------------ 6. repair landscape

def _landscape(ax, name: str) -> None:
    case = CASES[name]
    T = cs.exhaustive_table(case)
    a_b = e.mobius(T.G)
    q = s5.qubo_terms(case, a_b, T.V)
    net = hf.to_hopfield(*hf.rescale(q["Q"], q["q"], q["c"], hf.normalising_factor(q["Q"], q["q"])))
    star = admissible_minimal_repairs(T.F, T.V, T.K, case.k_max)
    order = np.argsort(q["H"], kind="stable")
    cols = ["subset S", "G [mm]", "F", "V", "K", "H [repair actions]", "Hopfield\nfixed point"]
    widths = [3.4, 1.2, 0.6, 0.6, 0.6, 1.8, 1.2]
    xs = np.r_[0, np.cumsum(widths)]
    for j, c in enumerate(cols):
        ax.text(xs[j] + widths[j] / 2, -0.95, c, ha="center", va="center", fontweight="bold", color=INK)
    for i, x in enumerate(order):
        S = [iv.intervention_id for p, iv in enumerate(case.candidates) if x >> p & 1]
        fixed = hf.is_fixed_point(net, hf.spins([x >> p & 1 for p in range(len(case.candidates))]))
        vals = ["{" + ", ".join(S) + "}", f"{1e3 * T.G[x]:.1f}", str(T.F[x]), str(T.V[x]), str(T.K[x]),
                f"{q['H'][x]:.4g}", "yes" if fixed else ""]
        for j, v in enumerate(vals):
            face = SURFACE
            if (j == 2 and T.F[x]) or (j == 1 and T.G[x] > 0):
                face = "#f6d5d4"
            ax.add_patch(Rectangle((xs[j], i - 0.5), widths[j], 1, fc=face, ec=GRID,
                                   hatch="xxx" if (j == 3 and T.V[x] == 0) else None))
            ax.text(xs[j] + widths[j] / 2, i, v, ha="center", va="center", fontsize=8, color=INK)
        if x in star:
            ax.add_patch(Rectangle((0, i - 0.5), xs[-1], 1, fc="none", ec=BLUE, lw=2.5))
    ax.set_xlim(-0.1, xs[-1] + 0.1)
    ax.set_ylim(len(order) - 0.4, -1.5)
    ax.axis("off")
    ax.set_title(f"{name}: all 2^{len(case.candidates)} subsets sorted by QUBO energy H", loc="left")


def fig_repair_landscape() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 4.8))
    _landscape(axes[0], "S5_ins_coupled")
    _landscape(axes[1], "S5_hinge_validity")
    handles = [Patch(fc=SURFACE, ec=BLUE, lw=2.5, label="certified S* (exact enumeration)"),
               Patch(fc="#f6d5d4", ec=GRID, label="infeasible: F = 1 / G > 0"),
               Patch(fc=SURFACE, ec=GRID, hatch="xxx", label="invalid post-intervention state: V = 0")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "repair_landscape.png")


# ------------------------------------------------------ 7. scaling

def fig_scaling() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    core = [r for r in ROWS.values() if r["structure"] != "perturbed"]
    by = s5.scaling(core, "minimal_cardinality")
    ks = sorted(int(k) for k in by)
    for key, color, mark, label in (("unary_match", BLUE, "o", "unary surrogate"), ("qubo_match", ORANGE, "s", "pairwise QUBO")):
        a.plot(ks, [by[str(k)][key] for k in ks], color=color, lw=2, marker=mark, ms=8, mec=SURFACE, mew=1.5, label=label)
    for k in ks:
        a.annotate(f"n = {by[str(k)]['n_scenes']}", (k, 0.04), ha="center", color=MUTED, fontsize=8)
    a.set_ylim(0, 1.1)
    a.set_xticks(ks)
    a.set_xlabel("certified minimal repair cardinality |S*| [interventions]")
    a.set_ylabel("fraction of core scenes recovering exact S* [-]")
    a.set_title("Exact repair recovery vs |S*| (core scenes only)", loc="left")
    a.legend(loc="center right")
    for group, filled in ((core, True), ([r for r in ROWS.values() if r["structure"] == "perturbed"], False)):
        b.scatter([r["P"] for r in group], [1e3 * r["exhaustive_runtime_s"] for r in group], s=50, color=BLUE if filled else "none",
                  ec=BLUE, lw=1.4, label="core scene" if filled else "perturbation copy of S5_ins_k3", zorder=3)
    Ps = sorted({r["P"] for r in ROWS.values()})
    b.set_xticks(Ps, [f"{P}\n({2 ** P})" for P in Ps])
    b.set_yscale("log")
    b.set_xlabel("candidate interventions P  (subsets enumerated 2^P) [-]")
    b.set_ylabel("exhaustive 3D table runtime [ms]")
    b.set_title("Exact enumeration cost vs P", loc="left")
    b.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "scaling.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fig in (fig_scene_envelopes, fig_conflict_vs_tau, fig_interaction_graphs, fig_cardinality_vs_order,
                fig_coefficient_stability, fig_repair_landscape, fig_scaling):
        fig()
        print("wrote", fig.__name__)


if __name__ == "__main__":
    main()
