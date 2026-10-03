"""Inspect one stored TRAIN or VALIDATION scene and render a chosen intervention state (read-only tooling).

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/visualize_scene.py --scene-id train-ms-000
    ... --state 5            (bitmask over the candidate list)
    ... --options 0,3        (same, as option indices)
    ... --debug              (per-entity min signed distance, penetration, conflict c_i, tau of minimum distance)

Prints the scene, its prescribed action, the oracle verdict, direct blockers, the candidate interventions and the
stored optimal repair set S*, then saves PoC-3/out/visualization/scene_<id>[_state<x>].png: original vs requested
state (default: the lowest-index optimal repair). Test-split scenes are refused.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import viz_utils as vz  # noqa: E402
from poc2 import dataset as pd  # noqa: E402
from poc3 import dataset as ds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "out"


def parse_state(state: int | None, options: str | None, P: int) -> int | None:
    if state is not None and options is not None:
        raise SystemExit("use either --state or --options, not both")
    if options is not None:
        idx = sorted({int(v) for v in options.split(",") if v.strip()})
        state = sum(1 << p for p in idx)
    if state is not None and not 0 <= state < 2 ** P:
        raise SystemExit(f"state must be in 0 .. {2 ** P - 1} for P = {P}")
    return state


def load_scene(scene_id: str):
    records, arrays = ds.load(ROOT / "s2")
    j = next((k for k, r in enumerate(records) if r["scene_id"] == scene_id), None)
    if j is None:
        raise SystemExit(f"unknown scene id {scene_id}")
    if records[j]["split"] not in ("train", "val"):
        raise SystemExit("test-split scenes are never inspected")
    scene, regions, options = ds.build(pd.spec_from_json(records[j]["spec"]))
    return records[j], scene, regions, options, ds.unpack_states(arrays["optimal"][j], int(arrays["P"][j]))


def describe(rec, scene, options, opt, info) -> str:
    a = info["a"]
    lines = [f"Scene: {rec['scene_id']}", f"Family: {rec['family']}", "", "Proposed action:",
             f"  {vz.ACTION_TEXT[rec['family']]}", "", "Original:", f"  {'INFEASIBLE' if a.F else 'FEASIBLE'}", "",
             "Direct blockers:", "  " + (", ".join(a.blockers) or "none"), "", "Candidate interventions:"]
    lines += [f"  [{p}] {vz.option_text(o)}  (cost {o.cost})" for p, o in enumerate(options)]
    lines += ["", "Optimal repair(s):"] + ["  {" + ", ".join(str(p) for p in vz.bits(x, len(options))) + "}" for x in sorted(opt)]
    return "\n".join(lines)


def debug_table(info) -> str:
    a, sw = info["a"], info["sweep"]
    rows = ["entity            min_d[mm]  penetration[mm]  c_i[mm]  tau_of_min_d"]
    for k, i in enumerate(a.ids):
        rows.append(f"{i:<16} {a.d[k] * 1000:10.2f} {a.p[k] * 1000:16.2f} {a.c[k] * 1000:8.2f} "
                    f"{float(sw.taus[sw.distances[:, k].argmin()]):13.3f}")
    return "\n".join(rows)


def render(path: Path, rec, scene, regions, options, x: int) -> dict:
    base = vz.analyse(scene)
    after = vz.apply_state(scene, options, x)
    info = vz.analyse(after)
    for inf, lab in ((base, "original"), (info, "requested")):
        vz.verify(inf, None, lab)
    shifted = any(options[p].intervention.kind.value == "shift_target" for p in vz.bits(x, len(options)))
    fig = plt.figure(figsize=(15, 9))
    for c, (sc, inf, ttl) in enumerate(((scene, base, "original state"),
                                        (after, info, "requested state: " + ("; ".join(vz.option_text(options[p]) for p in
                                                                                       vz.bits(x, len(options))) or "no intervention")))):
        for r, plane in enumerate(("3d", "xy")):
            ax = fig.add_subplot(2, 2, 1 + c + 2 * r, projection="3d" if plane == "3d" else None)
            vz.draw_scene(ax, sc, plane, blockers=inf["a"].blockers, regions=regions)
            if c == 1:
                vz.relocation_arrows(ax, scene, after, plane)
                if shifted:
                    vz.draw_motion(ax, scene, base, plane, faint=True, ls="--", show_hit=False)
            vz.draw_motion(ax, sc, inf, plane, path_color=vz.C["shift"] if c == 1 and shifted else vz.C["trajectory"])
            vz.status(ax, inf["a"].F, plane)
            vz.frame_axes(ax, [(scene, base), (after, info)], plane)
            if r == 0:
                ax.set_title(ttl, fontsize=10, weight="bold")
    fig.suptitle(f"{rec['scene_id']} ({rec['family']}): resulting action feasibility F = {info['a'].F}", fontsize=13,
                 weight="bold")
    vz.legend(fig, y=0.0)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"F": info["a"].F, "blockers": list(info["a"].blockers)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--state", type=int)
    ap.add_argument("--options")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "visualization"))
    args = ap.parse_args(argv)
    rec, scene, regions, options, opt = load_scene(args.scene_id)
    x = parse_state(args.state, args.options, len(options))
    base = vz.analyse(scene)
    print(describe(rec, scene, options, opt, base))
    if args.debug:
        print("\n" + debug_table(base))
    shown = min(opt) if x is None else x
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"scene_{rec['scene_id']}" + ("" if x is None else f"_state{x}") + ".png")
    res = render(path, rec, scene, regions, options, shown)
    print(f"\nRequested state {shown} = options {vz.bits(shown, len(options))}: "
          f"{'INFEASIBLE' if res['F'] else 'FEASIBLE'}" + (f" (blockers: {', '.join(res['blockers'])})" if res["blockers"] else ""))
    print(f"Saved {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
