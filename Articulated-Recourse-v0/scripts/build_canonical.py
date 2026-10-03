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
                                hstack, label_slots, save_png, slot_ghosts)
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


def target_name(pb):
    t = pb.target_spec
    return f"{t['skill']}({'upper rack' if t.get('rack') == 'rack1' else t.get('rack', 'door')})" \
        if t["skill"] == "PUSH_RACK" else "CLOSE_DOOR"


def slot_status(pb, state, slot, exclude):
    occ = [pb.slot_occupant(state, sl, exclude=exclude) for sl in (slot,) + tuple(pb.topo[slot].overlaps)]
    occ = [o for o in occ if o]
    return f"destination {slot} OCCUPIED by {occ[0]}" if occ else f"destination {slot} EMPTY"


def render_scene(pb, res, outdir: Path, variant: str):
    rr = SceneRenderer(pb, W, H)
    m, d = pb.scene.model, pb.scene.data
    s0 = pb.s0
    tn = target_name(pb)
    blockers = res["direct_target_blockers"]
    hl = {b: RED for b in blockers}
    st_target = (getattr(pb, "target_robot", None) or {}).get("station")
    # observations (unannotated)
    rr.set_state(s0)
    save_png(outdir / "obs_front.png", rr.render("front"))
    save_png(outdir / "obs_oblique.png", rr.render("oblique"))
    dep = rr.depth("front")
    save_png(outdir / "depth_front.png", np.clip(dep * 1000, 0, 65535).astype(np.uint16))
    save_png(outdir / "mask_front.png", rr.instance_mask("front"))
    # placement_map.png: slot footprints + labels, oblique + top-down (robot hidden)
    rr.set_state(s0)
    imgs = []
    for cam in ("oblique", "topo"):
        img = rr.render(cam, ghosts=slot_ghosts(pb.topo), robot=False)
        img = label_slots(img, m, d, cam, pb.topo, W, H)
        imgs.append(caption(img, [f"{variant}: placement topology ({cam})",
                                  "blue U1-U5 upper rack | green B tray (B1-B4 front, B5-B8 back)",
                                  "lower rack L1-L4: inside the tub, never a destination (not drawn)"]))
    save_png(outdir / "placement_map.png", hstack(*imgs))
    # scene.png
    ob = caption(rr.render("oblique", highlight=hl), [f"{variant}: initial state, target {tn}",
                                                       "red = direct blockers"])
    top = rr.render("topo", highlight=hl, ghosts=slot_ghosts(pb.topo), robot=False)
    top = caption(label_slots(top, m, d, "topo", pb.topo, W, H), [pb.spec.scene_id])
    panel = dependency_panel(res, size=(560, H))
    save_png(outdir / "scene.png", hstack(ob, top, panel))
    save_png(outdir / "dependency_graph.png",
             dependency_panel(res, size=(820, 560), title=f"{variant}: interventional dependency graph (optimal sequence 1)"))
    # swept-volume debug renders
    t = pb.target
    det = res["direct_target_blocker_details"]
    ts = min((dd["tau_star"] for dd in det.values()), default=None)
    save_png(outdir / "sweep_target.png",
             rr.sweep_image(t, s0, None, blockers, ts, cam="closeup", k=6, title=f"{tn} swept volume, initial state"))
    if res["sequence_proofs"]:
        proof = res["sequence_proofs"][0]
        seq = proof["sequence"]
        for k, ivid in enumerate(seq):
            iv = pb.iv_by_id[ivid]
            st = state_after(pb, seq[:k])
            preds = [e for e in proof["edges"] if e["to"] == ivid]
            dslot = iv.placement.slot
            cam = "topo" if pb.topo[dslot].semantic == "temporary_buffer" or iv.src.slot.startswith("B") else "oblique"
            if preds:   # counterfactual: the state without the prerequisite step
                e = preds[0]
                cf = state_after(pb, [x for x in seq[:k] if x != e["from"]])
                img = rr.sweep_image(iv.traj, cf, iv, [e["blocker"]], e["cause"]["tau_star"], cam=cam, dest_slot=dslot,
                                     title=f"step {k + 1} {ivid}: blocked by {e['blocker']} unless {e['from']} runs first",
                                     extra_lines=[f"{slot_status(pb, cf, dslot, iv.obj)} | cause {e['cause']['type']} "
                                                  f"({e['cause']['mechanism']}, {e['cause']['phase_star']}, "
                                                  f"d={e['cause']['min_dist']:+.3f} m)"])
            else:
                img = rr.sweep_image(iv.traj, st, iv, [], None, cam=cam, dest_slot=dslot,
                                     title=f"step {k + 1} {ivid}: executable",
                                     extra_lines=[slot_status(pb, st, dslot, iv.obj)])
            save_png(outdir / f"sweep_step{k + 1}.png", img)
    # failed_action.mp4
    vw = VideoWriter(outdir / "failed_action.mp4", W, H)
    rr.set_state(s0)
    fc = first_conflict_index(pb, s0)
    stop = fc[0] if fc else t.n - 1
    frames = rr.action_frames(t, None, "closeup", hl, n_frames=max(25, int(60 * stop / t.n)), stop_index=stop,
                              title=f"TARGET (next action): {tn}", station=st_target)
    for f in frames:
        vw.add(f)
    if fc:
        last = caption(rr.last_raw, [f"COLLISION at tau={t.tau[stop]:.2f}: {fc[1]} "
                                    f"({det[fc[1]]['part_star']}, {det[fc[1]]['min_dist']:+.3f} m)",
                                    f"F(s0, {tn}) = 1   blockers: {', '.join(blockers)}"], color=(255, 120, 120))
        for _ in range(30):
            vw.add(last)
    vw.close()
    # repair.mp4
    vw = VideoWriter(outdir / "repair.mp4", W, H)
    rr.set_state(s0)
    intro = caption(rr.render("oblique", highlight=hl), [f"{variant}: target {tn}; blockers (red): {', '.join(blockers)}",
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
        for f in rr.action_frames(tf, None, "closeup", None, n_frames=60, title=f"TARGET after repair: {tn}",
                                  station=st_target):
            vw.add(f)
        done = caption(rr.last_raw, [f"SUCCESS: F(s, {tn}) = {pb.F(state)} after {len(seq)} repair(s)"], color=(140, 255, 140))
        for _ in range(25):
            vw.add(done)
    vw.close()
    rr.set_state(s0)
    save_png(outdir / "_thumb.png", caption(rr.render("oblique", highlight=hl, robot=False), []))
    rr.close()


def compose_comparison():
    import cv2

    tiles = []
    for v in sorted(ALL):
        f = OUT / v / "oracle.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        lab = rec["labels"]
        n_edges = len({(e["from"], e["to"]) for p in lab["sequence_proofs"] for e in p["edges"] if e["to"] != "TARGET"})
        img = cv2.cvtColor(cv2.imread(str(OUT / v / "_thumb.png")), cv2.COLOR_BGR2RGB)
        img = caption(img, [f"{v}  target: {rec['inputs']['target_action']['name']}",
                            f"blockers: {', '.join(lab['direct_target_blockers'])}",
                            f"dependency edges: {n_edges}   optimal repair length: {lab['optimal_cost']}"], scale=0.5)
        tiles.append(img)
    rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    save_png(OUT / "comparison.png", np.concatenate(rows, 0))


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
    args = sys.argv[1:]
    if args == ["--compose"]:
        compose_comparison()
        summary = []
        for v in sorted(ALL):
            f = OUT / v / "oracle.json"
            if f.exists():
                r = json.loads(f.read_text())
                summary.append({"variant": v, "pass": r["verification"]["satisfied"], "accepted": r["quality"]["accepted"],
                                "L": r["labels"]["optimal_cost"], "blockers": r["labels"]["direct_target_blockers"],
                                "sequences": r["labels"]["optimal_sequences"]})
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    else:
        for v in args or list(ALL):
            build_one(v)
