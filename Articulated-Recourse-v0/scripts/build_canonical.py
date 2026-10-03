"""Build, verify and render the eight canonical scenes (V0-V7).

    python scripts/build_canonical.py            # all
    python scripts/build_canonical.py V3 V5      # subset

Per scene (canonical/<V>/):
  oracle.json            record (inputs / labels separated) + verification + quality
  inputs.npz, labels.npz numerical arrays (trajectories; distance profiles = labels)
  scene.png              initial state (blockers red) + interventional dependency graph
  dependency_graph.png   the dependency graph alone (edges: blocker, cause, phase, min distance)
  sweep_target.png       swept volume of CLOSE_DISHWASHER in the initial state, tau* pose, blockers
  sweep_step<k>.png      swept volume of each optimal repair step in the state where it runs,
                         plus the counterfactual blocker that made it necessary
  obs_front.png, obs_oblique.png, depth_front.png (uint16 mm), mask_front.png (object index)
  failed_action.mp4      the target action in the initial state, stopped at the first collision
  repair.mp4             blockers, each repair of the first optimal sequence, successful target
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from artrecourse import ROOT  # noqa: E402
from artrecourse.canonical import ALL  # noqa: E402
from artrecourse.interventions import build_problem  # noqa: E402
from artrecourse.oracle import solve, state_after  # noqa: E402
from artrecourse.quality import coarse_fine_check, quality_report  # noqa: E402
from artrecourse.records import build_record, write_record  # noqa: E402
from artrecourse.render import (GREEN, RED, SceneRenderer, VideoWriter, caption, dependency_panel,  # noqa: E402
                                hstack, save_png)
from artrecourse.sweep import PEN_TOL  # noqa: E402
from artrecourse.variants import check_variant, classify_all  # noqa: E402

OUT = ROOT / "canonical"
W, H = 640, 480


def first_conflict_index(pb, state):
    best = None
    for i, o in enumerate(pb.obj_keys):
        d = pb.ttab[o][state[i]].dist
        hit = np.where(d <= -PEN_TOL)[0]
        if len(hit) and (best is None or hit[0] < best[0]):
            best = (int(hit[0]), o)
    return best


def render_scene(pb, res, outdir: Path, variant: str):
    rr = SceneRenderer(pb, W, H)
    s0 = pb.s0
    blockers = res["direct_target_blockers"]
    hl = {b: RED for b in blockers}
    # observations (unannotated)
    rr.set_state(s0)
    save_png(outdir / "obs_front.png", rr.render("front"))
    save_png(outdir / "obs_oblique.png", rr.render("oblique"))
    dep = rr.depth("front")
    save_png(outdir / "depth_front.png", np.clip(dep * 1000, 0, 65535).astype(np.uint16))
    save_png(outdir / "mask_front.png", rr.instance_mask("front"))
    # scene.png: oblique + closeup with blockers + dependency graph
    ob = caption(rr.render("oblique", highlight=hl), [f"{variant}: initial state", "red = direct blockers of CLOSE_DISHWASHER"])
    cu = caption(rr.render("closeup", highlight=hl), [pb.spec.scene_id])
    panel = dependency_panel(res, size=(560, H))
    save_png(outdir / "scene.png", hstack(ob, cu, panel))
    save_png(outdir / "dependency_graph.png",
             dependency_panel(res, size=(820, 560), title=f"{variant}: interventional dependency graph (optimal sequence 1)"))
    # swept-volume debug renders
    t = pb.target
    det = res["direct_target_blocker_details"]
    ts = min((d["tau_star"] for d in det.values()), default=None)
    save_png(outdir / "sweep_target.png",
             rr.sweep_image(t, s0, None, blockers, ts, cam="closeup", k=12,
                            title="CLOSE_DISHWASHER swept volume, initial state"))
    if res["sequence_proofs"]:
        proof = res["sequence_proofs"][0]
        seq = proof["sequence"]
        for k, ivid in enumerate(seq):
            iv = pb.iv_by_id[ivid]
            st = state_after(pb, seq[:k])
            preds = [e for e in proof["edges"] if e["to"] == ivid]
            if preds:   # show the counterfactual: the blocker that made this step wait
                e = preds[0]
                cf = state_after(pb, [x for x in seq[:k] if x != e["from"]])
                img = rr.sweep_image(iv.traj, cf, iv, [e["blocker"]], e["cause"]["tau_star"],
                                     title=f"step {k + 1} {ivid}: blocked by {e['blocker']} "
                                           f"({e['cause']['type']}) unless {e['from']} runs first")
            else:
                img = rr.sweep_image(iv.traj, st, iv, [], None, title=f"step {k + 1} {ivid}: executable")
            save_png(outdir / f"sweep_step{k + 1}.png", img)
    # failed_action.mp4
    vw = VideoWriter(outdir / "failed_action.mp4", W, H)
    rr.set_state(s0)
    fc = first_conflict_index(pb, s0)
    stop = fc[0] if fc else t.n - 1
    frames = rr.action_frames(t, None, "oblique", hl, n_frames=max(20, int(60 * stop / t.n)), stop_index=stop,
                              title="TARGET: CLOSE_DISHWASHER (push upper rack, close door)")
    for f in frames:
        vw.add(f)
    if fc:
        last = caption(rr.last_raw, [f"COLLISION at tau={t.tau[stop]:.2f}: {fc[1]} "
                                    f"({det[fc[1]]['part_star']}, {det[fc[1]]['min_dist']:+.3f} m)",
                                    f"F(s0, close) = 1   blockers: {', '.join(blockers)}"], color=(255, 120, 120))
        for _ in range(30):
            vw.add(last)
    vw.close()
    # repair.mp4
    vw = VideoWriter(outdir / "repair.mp4", W, H)
    rr.set_state(s0)
    intro = caption(rr.render("oblique", highlight=hl), [f"{variant}: blockers (red): {', '.join(blockers)}",
                                                         f"minimal recourse: {res['optimal_cost']} intervention(s)"])
    for _ in range(25):
        vw.add(intro)
    state = s0
    if res["optimal_sequences"]:
        seq = res["optimal_sequences"][0]
        for k, ivid in enumerate(seq):
            iv = pb.iv_by_id[ivid]
            rr.set_state(state)
            for f in rr.action_frames(iv.traj, iv, "oblique", {iv.obj: GREEN}, n_frames=55,
                                      title=f"repair {k + 1}/{len(seq)}: {ivid}"):
                vw.add(f)
            state = pb.apply(state, iv)
        rr.set_state(state)
        tf = pb._target_traj(state)
        for f in rr.action_frames(tf, None, "oblique", None, n_frames=60,
                                  title="TARGET after repair: CLOSE_DISHWASHER"):
            vw.add(f)
        done = caption(rr.last_raw, [f"SUCCESS: F(s, close) = {pb.F(state)} after {len(seq)} repair(s)"], color=(140, 255, 140))
        for _ in range(25):
            vw.add(done)
    vw.close()
    rr.set_state(s0)
    rr.close()


def build_one(v: str) -> dict:
    t0 = time.time()
    spec = ALL[v]()
    outdir = OUT / v
    outdir.mkdir(parents=True, exist_ok=True)
    pb = build_problem(spec)
    res = solve(pb)
    scene = {"objects": spec.objects, "candidate_interventions": pb.ivs}
    ok, why = check_variant(v, res, scene)
    fine = coarse_fine_check(pb, res)
    q = quality_report(pb, res, fine)
    rec, ai, al = build_record(pb, res, v, ok, q)
    rec["verification"] = {"requested_variant": v, "satisfied": ok, "unmet": why,
                           "all_variant_definitions_satisfied": [k for k, x in classify_all(res, scene).items() if x],
                           "factorisation_spot_check": spot_check(pb, res)}
    write_record(outdir, rec, ai, al)
    t1 = time.time()
    render_scene(pb, res, outdir, v)
    print(f"{v}: variant {'PASS' if ok else 'FAIL'} {why} | quality accepted={q['accepted']} "
          f"coarse/fine agree={fine['agree']} | L={res['optimal_cost']} | oracle {t1 - t0:.0f}s render {time.time() - t1:.0f}s",
          flush=True)
    return {"variant": v, "pass": ok, "accepted": q["accepted"], "L": res["optimal_cost"],
            "blockers": res["direct_target_blockers"], "sequences": res["optimal_sequences"]}


def spot_check(pb, res) -> dict:
    """Brute-force re-sweep in explicit states (initial + every prefix of the first optimal
    sequence) must agree with the factorised tables."""
    n, bad = 0, []
    states = [pb.s0]
    if res["optimal_sequences"]:
        seq = res["optimal_sequences"][0]
        states += [state_after(pb, seq[:k]) for k in range(1, len(seq) + 1)]
    for s in states:
        if pb.full_check_target(s) != pb.F(s):
            bad.append(f"F mismatch in {s}")
        n += 1
        for iv in pb.ivs:
            if pb.full_check_intervention(iv, s) != pb.executable(iv, s):
                bad.append(f"{iv.id} mismatch in {s}")
            n += 1
    return {"checks": n, "mismatches": bad}


if __name__ == "__main__":
    vs = sys.argv[1:] or list(ALL)
    summary = [build_one(v) for v in vs]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
