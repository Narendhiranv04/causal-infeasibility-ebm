"""Generate the human-readable scene / infeasibility explainer (read-only explanatory tooling).

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/make_visual_explainer.py

Writes PoC-3/out/visualization/{unary_example, pairwise_envelope_coupling, pairwise_story, cause_vs_repair,
family_<family>}.png, pairwise_envelope_coupling.gif and scene_manifest.json. Uses only the frozen scene builders,
oracle and the digest-verified Stage-2 TRAIN scenes; trains nothing, never touches the test split or seed 61. Every
panel's FEASIBLE / INFEASIBLE label and red highlight is checked against the oracle before it is drawn.
"""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import viz_utils as vz  # noqa: E402
from poc.oracle import CONTACT_TOL_3D  # noqa: E402
from poc2 import dataset as pd  # noqa: E402
from poc2 import oracle as orc  # noqa: E402
from poc2 import scenes  # noqa: E402
from poc2 import structure as st  # noqa: E402
from poc3 import dataset as ds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "out"
OUT = ROOT / "visualization"
M4_STATES = {"A. ORIGINAL": 0, "B. MOVE nb ONLY": 0b10, "C. SHIFT ONLY": 0b01, "D. SHIFT + MOVE nb": 0b11}
M4_EXPECT = {0: 1, 0b10: 1, 0b01: 1, 0b11: 0}


def panel(ax, base, base_info, scene, info, plane, regions=(), dest=None, shifted=False, title=None):
    vz.draw_scene(ax, scene, plane, blockers=info["a"].blockers, regions=regions, dest_region=dest)
    if scene is not base:
        vz.relocation_arrows(ax, base, scene, plane)
    if shifted:
        vz.draw_motion(ax, base, base_info, plane, faint=True, ls="--", show_hit=False)
    vz.draw_motion(ax, scene, info, plane, path_color=vz.C["shift"] if shifted else vz.C["trajectory"])
    vz.status(ax, info["a"].F, plane)
    if title:
        ax.set_title(title, fontsize=10, weight="bold")


def why(info) -> str:
    a = info["a"]
    if not a.F:
        return "nothing intersects the swept motion"
    def when(t):
        return "the start pose" if t <= 0 else ("the goal pose" if t >= 1 else f"{t:.0%} of the motion")
    parts = [f"the moving gripper/object sweeps {-a.d[a.ids.index(i)] * 1000:.0f} mm into '{i}' at "
             f"{when(float(info['sweep'].taus[k]))}" for i, k in info["tau_idx"].items()]
    extra = " (start and goal poses are both collision-free: the collision happens mid-motion)" if a.interior_only else ""
    return "; ".join(parts) + extra


def state_scene(scene, options, x):
    s = vz.apply_state(scene, options, x)
    return s, vz.analyse(s)


def before_repair_after(path: Path, title: str, scene, regions, options, x: int, lines: list[str], side: bool) -> dict:
    """Three columns: BEFORE (oracle F = 1) / the intervention itself / AFTER (oracle F = 0, same action)."""
    base_info = vz.analyse(scene)
    vz.verify(base_info, 1, f"{title} BEFORE")
    after, info = state_scene(scene, options, x)
    vz.verify(info, 0, f"{title} AFTER")
    chosen = vz.bits(x, len(options))
    shifted = any(options[p].intervention.kind.value == "shift_target" for p in chosen)
    dests = [options[p].region_id for p in chosen if options[p].region_id]
    planes = ("3d", "xy", "xz") if side else ("3d", "xy")
    fig = plt.figure(figsize=(16, 11.5 if side else 9))
    gs = GridSpec(len(planes), 3, figure=fig, height_ratios=[0.8, 1.25, 0.8][:len(planes)], hspace=0.12, wspace=0.1,
                  bottom=0.25 if side else 0.3, top=0.9)
    heads = ["1  BEFORE: the prescribed action is blocked",
             "2  REPAIR: " + "; ".join(vz.option_text(options[p]) for p in chosen), "3  AFTER: retry the SAME action"]
    for c in range(3):
        for r, plane in enumerate(planes):
            ax = fig.add_subplot(gs[r, c], projection="3d" if plane == "3d" else None)
            if c == 0:
                panel(ax, scene, base_info, scene, base_info, plane, regions)
            elif c == 1:  # the intervention only: old -> new, original path faint, shifted path (if any) in purple
                vz.draw_scene(ax, after, plane, regions=regions, dest_region=dests[0] if dests else None)
                vz.relocation_arrows(ax, scene, after, plane)
                vz.draw_motion(ax, scene, base_info, plane, faint=True, ls="--", show_hit=False)
                if shifted:
                    vz.draw_motion(ax, after, info, plane, faint=True, path_color=vz.C["shift"], show_hit=False)
                vz.status(ax, "INTERVENTION", plane)
            else:
                vz.draw_scene(ax, after, plane, regions=regions)
                vz.draw_motion(ax, after, info, plane, path_color=vz.C["shift"] if shifted else vz.C["trajectory"])
                vz.status(ax, info["a"].F, plane)
            vz.frame_axes(ax, [(scene, base_info), (after, info)], plane)
            ax.set_title(textwrap.fill(heads[c], 48) if r == 0 else {"xy": "top-down view", "xz": "side view"}[plane],
                         fontsize=10.5 if r == 0 else 8, weight="bold" if r == 0 else "normal")
    fig.suptitle(title, fontsize=15, weight="bold")
    fig.text(0.05, 0.075, "\n".join(textwrap.fill(l, 125, subsequent_indent=" " * 21) for l in lines), fontsize=10,
             family="monospace", va="bottom")
    vz.legend(fig, y=0.005)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"before": {"F": base_info["a"].F, "G": base_info["a"].G, "blockers": list(base_info["a"].blockers)},
            "after": {"F": info["a"].F, "G": info["a"].G}, "state": x, "options": [options[p].option_id for p in chosen]}


def dataset_scene(records, arrays, j):
    spec = pd.spec_from_json(records[j]["spec"])
    scene, regions, options = ds.build(spec)
    return scene, regions, options, ds.unpack_states(arrays["optimal"][j], int(arrays["P"][j]))


def find_unary(records, arrays) -> int:
    rows = sorted((int(arrays["P"][j]), j) for j, r in enumerate(records) if r["split"] == "train" and r["intent"] == "repairable")
    for _, j in rows:
        scene, _, options, opt = dataset_scene(records, arrays, j)
        a = orc.evaluate(scene)
        if a.F != 1 or len(a.blockers) != 1:
            continue
        b = a.blockers[0]
        movable = any(e.eid == b and e.entity.movable for e in scene.entities)
        if movable and any((1 << p) in opt and o.intervention.entity_id == b for p, o in enumerate(options)):
            return j
    raise RuntimeError("no simple unary train scene found")


def m4_figures(manifest: dict) -> dict:
    case = scenes.envelope_coupling()
    sc0, regions, options = case.scene, case.regions, case.options
    infos = {x: state_scene(sc0, options, x) for x in M4_STATES.values()}
    for x, F in M4_EXPECT.items():
        vz.verify(infos[x][1], F, f"M4 state {x}")
    if infos[0][1]["a"].blockers != ("jamb",) or infos[1][1]["a"].blockers != ("nb",):
        raise AssertionError("M4 blockers differ from the designed mechanism; refusing to draw")
    T = orc.repair_table(sc0, options)
    if st.exact_decision(T) != {0b11}:
        raise AssertionError("M4 S* is not {SHIFT, nb->r1}")
    G = {x: float(T.G[x]) for x in M4_STATES.values()}
    beta = G[3] - G[1] - G[2] + G[0]
    base_info = infos[0][1]
    # ---- 2x2 counterfactual figure
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[3, 2, 3, 2], hspace=0.16, wspace=0.06, bottom=0.24, top=0.92)
    notes = {"A. ORIGINAL": "jamb blocks the action", "B. MOVE nb ONLY": "jamb still blocks",
             "C. SHIFT ONLY": "jamb cleared, but the shifted path now hits nb", "D. SHIFT + MOVE nb": "both cleared: same action succeeds"}
    for k, (name, x) in enumerate(M4_STATES.items()):
        sc, inf = infos[x]
        shifted = bool(x & 1)
        for c, plane in enumerate(("xy", "3d")):
            ax = fig.add_subplot(gs[k // 2, 2 * (k % 2) + c], projection="3d" if plane == "3d" else None)
            panel(ax, sc0, base_info, sc, inf, plane, regions, "r1" if x & 2 else None, shifted,
                  f"{name}: {notes[name]}" if c == 0 else None)
            vz.frame_axes(ax, [(sc0, base_info), (infos[3][0], infos[3][1])], plane)
            if c == 0:
                ax.text(0.98, 0.04, f"F = {inf['a'].F}", transform=ax.transAxes, ha="right", fontsize=12, weight="bold")
    fig.suptitle("Why is this repair PAIRWISE?  (M4 envelope coupling, exact oracle geometry)", fontsize=15, weight="bold")
    fig.text(0.05, 0.075, "Why is this pairwise?\n\n"
             "  Moving nb alone does not solve the original jamb collision.\n"
             "  Shifting alone avoids the jamb but creates a new collision with nb.\n"
             "  Only SHIFT + MOVE(nb) makes the prescribed action feasible.\n\n"
             f"  Oracle total conflict G:  G(none) = {G[0] * 1000:.1f} mm,  G(MOVE) = {G[2] * 1000:.1f} mm,  "
             f"G(SHIFT) = {G[1] * 1000:.1f} mm,  G(SHIFT+MOVE) = {G[3] * 1000:.1f} mm\n"
             f"  Pair interaction  beta(SHIFT, MOVE) = G(S+M) - G(S) - G(M) + G(none) = {beta * 1000:+.1f} mm  (non-zero => pairwise)",
             fontsize=10.5, family="monospace", va="bottom")
    vz.legend(fig, y=0.005)
    fig.savefig(OUT / "pairwise_envelope_coupling.png", dpi=130)
    plt.close(fig)
    # ---- compact story
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw=dict(wspace=0.32))
    story = [(0, "ORIGINAL\njamb is the blocker"), (1, "SHIFTED ACTION\njamb leaves the path, but nb enters it"),
             (3, "REPAIRED (SHIFT + MOVE nb)\npath clear")]
    for ax, (x, ttl) in zip(axs, story):
        sc, inf = infos[x]
        panel(ax, sc0, base_info, sc, inf, "xy", regions, "r1" if x & 2 else None, bool(x & 1), ttl)
        vz.frame_axes(ax, [(sc0, base_info), (infos[3][0], infos[3][1])], "xy")
    fig.suptitle("SHIFT changes the geometry, so it changes which other intervention becomes necessary", fontsize=13,
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=6)
    for k, lab in enumerate(("SHIFT", "MOVE nb")):  # arrows drawn in the gaps between the panels
        x0, x1 = axs[k].get_position().x1, axs[k + 1].get_position().x0
        col = vz.C["shift"] if k == 0 else vz.C["relocation"]
        fig.patches.append(matplotlib.patches.FancyArrow((x0 + 0.004), 0.5, (x1 - x0) - 0.012, 0, width=0.012, head_width=0.04,
                                                         head_length=0.012, color=col, transform=fig.transFigure))
        fig.text((x0 + x1) / 2, 0.57, lab, ha="center", fontsize=12, weight="bold", color=col)
    fig.savefig(OUT / "pairwise_story.png", dpi=130)
    plt.close(fig)
    # ---- cause vs repair
    cause = orc.causes(base_info["a"])
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    panel(axs[0], sc0, base_info, sc0, base_info, "xy", regions, title="CAUSE: what blocks the original action")
    axs[0].text(0.02, 0.04, f"direct blocker B0 = {sorted(cause.blockers)}\nminimal causal set C* = "
                f"{[sorted(c) for c in cause.minimal_causes]}\njamb is STRUCTURAL: it cannot be moved", transform=axs[0].transAxes,
                fontsize=10, bbox=dict(fc="white", ec=vz.C["blocker"]))
    panel(axs[1], sc0, base_info, infos[3][0], infos[3][1], "xy", regions, "r1", True,
          "REPAIR: what the robot can actually do")
    axs[1].text(0.02, 0.04, "minimal executable repair S* = {SHIFT_TARGET, move nb -> r1}\n(the jamb is never touched)",
                transform=axs[1].transAxes, fontsize=10, bbox=dict(fc="white", ec=vz.C["ok"]))
    for ax in axs:
        vz.frame_axes(ax, [(sc0, base_info), (infos[3][0], infos[3][1])], "xy")
    fig.suptitle("CAUSE  !=  REPAIR", fontsize=20, weight="bold", color=vz.C["blocker"])
    fig.text(0.05, 0.02, "The jamb causes the original failure, but it is structural and cannot simply be moved. The executable "
             "repair changes the\naction path (SHIFT), which makes nb relevant, so nb must also be moved. Blocker attribution "
             "(B0 / C*) answers 'what is in the\nway'; repair selection (S*) answers 'what is the cheapest set of allowed actions "
             "that makes the same motion succeed'.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(OUT / "cause_vs_repair.png", dpi=130)
    plt.close(fig)
    m4_gif(sc0, infos)
    manifest["M4_envelope_coupling"] = {"source": "poc2.scenes.envelope_coupling()", "options": [o.option_id for o in options],
                                        "states": {n: {"bitmask": x, "F": infos[x][1]["a"].F, "G": G[x],
                                                       "blockers": list(infos[x][1]["a"].blockers)} for n, x in M4_STATES.items()},
                                        "beta_shift_move": beta, "S_star": [3], "B0": sorted(cause.blockers)}
    return {"G": G, "beta": beta, "cause": sorted(cause.blockers)}


def m4_gif(sc0, infos) -> None:
    (s0, i0), (s3, i3) = infos[0], infos[3]
    jamb = i0["a"].ids.index("jamb")
    hit = int(np.flatnonzero(i0["sweep"].distances[:, jamb] < -CONTACT_TOL_3D)[0])
    seq = [("before", k, False) for k in np.linspace(0, hit, 22).astype(int)] + [("before", hit, f % 2 == 0) for f in range(10)]
    seq += [("swap", 0, False)] * 8 + [("after", k, False) for k in np.linspace(0, len(i3["poses"]) - 1, 32).astype(int)]
    seq += [("after", len(i3["poses"]) - 1, False)] * 10
    fig, ax = plt.subplots(figsize=(9, 6.5))

    def update(f):
        mode, k, flash = seq[f]
        ax.clear()
        sc, inf = (s0, i0) if mode == "before" else (s3, i3)
        vz.draw_scene(ax, sc, "xy", blockers=("jamb",) if mode == "before" and (flash or k == hit) else ())
        if mode != "before":
            vz.relocation_arrows(ax, s0, s3, "xy")
        cent = np.array([np.mean(np.concatenate([c for _, c in vz.parts_at(sc.moving, p, q)]), axis=0) for p, q in inf["poses"]])
        if mode != "before":  # original path faint for comparison with the shifted one
            c0 = np.array([np.mean(np.concatenate([c for _, c in vz.parts_at(s0.moving, p, q)]), axis=0) for p, q in i0["poses"]])
            ax.plot(c0[:, 0], c0[:, 1], color=vz.C["trajectory"], lw=1.2, ls="--", alpha=0.45)
        ax.plot(cent[:, 0], cent[:, 1], color=vz.C["shift"] if mode != "before" else vz.C["trajectory"], lw=2, alpha=0.5)
        if mode != "swap":
            for n, pc in vz.parts_at(sc.moving, *inf["poses"][k]):
                vz.poly2d(ax, pc, "xy", facecolor=vz.C["target"] if n == "object" else vz.C["gripper"], alpha=0.8,
                          edgecolor="k", lw=0.6)
        msg = {"before": "BEFORE REPAIR: prescribed insertion" + ("  -  COLLISION WITH jamb" if k == hit else ""),
               "swap": "REPAIR: SHIFT the action sideways + MOVE nb to r1", "after": "AFTER SHIFT + RELOCATE nb: same action replayed"}
        ax.set_title(msg[mode], fontsize=12, weight="bold", color=vz.C["blocker"] if k == hit and mode == "before" else "k")
        if mode == "after" and k == len(i3["poses"]) - 1:
            vz.status(ax, 0)
        vz.frame_axes(ax, [(s0, i0), (s3, i3)], "xy")

    FuncAnimation(fig, update, frames=len(seq)).save(OUT / "pairwise_envelope_coupling.gif", writer=PillowWriter(fps=5))
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, arrays = ds.load(ROOT / "s2")  # digest-verified; only TRAIN rows are used below
    manifest, report = {}, {}
    j = find_unary(records, arrays)
    scene, regions, options, opt = dataset_scene(records, arrays, j)
    x = min(s for s in opt if s.bit_count() == 1)
    info0 = vz.analyse(scene)
    p = vz.bits(x, len(options))[0]
    lines = [f"Scene: {records[j]['scene_id']}  ({records[j]['family']})",
             f"Intended action:     {vz.ACTION_TEXT[records[j]['family']]}",
             f"Direct blocker:      {info0['a'].blockers[0]} (a movable box)",
             f"Why infeasible:      {why(info0)}",
             f"Selected repair:     {vz.option_text(options[p])}  (minimal, cost {options[p].cost})",
             "Result after repair: the same prescribed motion is collision-free (F = 0)"]
    manifest["unary_example"] = {"scene_id": records[j]["scene_id"], "spec": records[j]["spec"]} | before_repair_after(
        OUT / "unary_example.png", "A simple infeasible action and its one-step repair", scene, regions, options, x, lines,
        side=False)
    report["unary"] = {"scene": records[j]["scene_id"], "blocker": info0["a"].blockers[0], "repair": vz.option_text(options[p])}
    report["m4"] = m4_figures(manifest)
    report["families"] = {}
    for fam in ds.FAMILIES:
        rows = sorted((int(arrays["P"][k]), k) for k, r in enumerate(records)
                      if r["split"] == "train" and r["intent"] == "repairable" and r["family"] == fam)
        k = rows[0][1]
        scene, regions, options, opt = dataset_scene(records, arrays, k)
        x = min(opt)
        a = orc.evaluate(scene)
        lines = [f"scene_id: {records[k]['scene_id']}    family: {fam}", f"action:   {vz.ACTION_TEXT[fam]}",
                 f"direct blockers B0: {list(a.blockers)}", f"why infeasible: {why(vz.analyse(scene))}",
                 "S* (all tied optimal repairs): " + "; ".join("{" + ", ".join(vz.option_text(options[q]) for q in vz.bits(s, len(options)))
                                                             + "}" for s in sorted(opt)),
                 f"shown repair (lowest state index): state {x}, cost {sum(options[q].cost for q in vz.bits(x, len(options)))}"]
        manifest[f"family_{fam}"] = {"scene_id": records[k]["scene_id"], "spec": records[k]["spec"],
                                     "S_star": sorted(opt)} | before_repair_after(
            OUT / f"family_{fam}.png", f"{fam}: before / optimal repair / after", scene, regions, options, x, lines,
            side=fam.startswith("storage"))
        report["families"][fam] = (records[k]["scene_id"], list(a.blockers), manifest[f"family_{fam}"]["options"])
    (OUT / "scene_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    u, m = report["unary"], report["m4"]
    print("VISUAL EXPLAINER COMPLETE\n\n1. Simple unary example\n"
          f"   - scene {u['scene']}\n   - blocker {u['blocker']}\n   - repair {u['repair']}\n\n2. Pairwise example (M4)\n"
          f"   - original blocker: jamb (G = {m['G'][0] * 1000:.1f} mm)\n"
          f"   - why MOVE alone fails: nb is not in the original path; the jamb still blocks (G = {m['G'][2] * 1000:.1f} mm)\n"
          f"   - why SHIFT alone fails: the shifted path clears the jamb but runs into nb (G = {m['G'][1] * 1000:.1f} mm)\n"
          f"   - why the combination succeeds: shifted path + nb moved away -> G = {m['G'][3] * 1000:.1f}, F = 0 "
          f"(beta = {m['beta'] * 1000:+.1f} mm)\n\n3. Cause vs repair\n   - cause: {m['cause']} (structural)\n"
          "   - executable repair: SHIFT_TARGET + move nb -> r1\n\n4. Family examples")
    for fam, (sid, b0, opts) in report["families"].items():
        print(f"   - {fam}: {sid}, blockers {b0}, repair {opts}")
    print("\nOpen first:\n  PoC-3/out/visualization/pairwise_envelope_coupling.png\n"
          "  PoC-3/out/visualization/pairwise_envelope_coupling.gif")


if __name__ == "__main__":
    main()
