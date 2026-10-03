"""Scene record schema.

Every record has two strictly separated blocks:

  inputs : what a future model MAY consume (object identities / shapes / exact poses,
           articulation state, the target action, placement candidates, candidate
           interventions with their deterministic trajectories).
  labels : SUPERVISION / EVALUATION ONLY. Feasibility F, direct blockers, minimum
           signed distances, tau*, conflict magnitudes, executability, enable/disable
           matrices, compatibility edges, optimal sequences, proofs. These must never be
           fed to a model as input.

Numerical arrays are stored next to the JSON: `inputs.npz` (trajectories) and
`labels.npz` (per-(action, entity, pose) distance profiles).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geom import pose7, quat_from_yaw
from .interventions import RecourseProblem
from .primitives import APPROACH_H, COARSE_SPACING, FINE_SPACING, LIFT_CLEAR, RETREAT_H
from .sweep import CAP, CLEAR_TOL, PEN_TOL

SCHEMA = "artrecourse.scene.v0"
INPUT_KEYS = ("objects", "initial_poses", "articulation", "target_action", "placement_candidates",
              "candidate_interventions")
LABEL_KEYS = ("F_initial", "direct_target_blockers", "target_clearance", "per_action", "enable_edges",
              "disable_edges", "compatibility_edges", "optimal_sequences", "optimal_cost", "sequence_length",
              "no_recourse", "sequence_proofs", "irreducible_solutions")


def _pl_pose(pl):
    return pose7(pl.pos, quat_from_yaw(pl.yaw))


def build_record(pb: RecourseProblem, res: dict, variant: str | None, variant_ok: bool | None,
                 quality: dict) -> tuple[dict, dict, dict]:
    sc = pb.scene
    art = sc.articulation_targets()
    objects = []
    for o in pb.spec.objects:
        a = sc.objects[o.key].asset
        objects.append({"key": o.key, "asset_id": o.asset_id, "category": a.category, "scale": a.scale,
                        "extent_canonical": [round(x, 4) for x in a.meta["extent_canonical"]],
                        "mass": round(a.meta["mass"], 4)})
    cands = []
    for o in pb.obj_keys:
        for k, pl in enumerate(pb.poses[o][1:], start=1):
            cands.append({"object": o, "pose_index": k, "slot": pl.slot, "support": pl.support, "pose": _pl_pose(pl)})
    ivs = []
    for iv in pb.ivs:
        ivs.append({
            "id": iv.id, "object": iv.obj, "primitive": "RELOCATE", "destination_slot": iv.placement.slot,
            "source_pose": _pl_pose(iv.src), "destination_pose": _pl_pose(iv.placement),
            "grasp_template": iv.grasp.to_dict(), "admissible": iv.admissible, "rejection_reasons": iv.rejection,
            "robot": {"stations": iv.robot.get("stations")} if iv.robot else None,
            "grasp_pose": None,
            "trajectory_ref": f"inputs.npz:{iv.id}",
            "n_samples": int(iv.traj.n),
            "moving_bodies": ["gripper(panda hand + fingers)", f"carried:{iv.obj}"],
        })
        from .geom import mat_to_quat

        p, R = iv.traj.waypoints["grasp"]
        ivs[-1]["grasp_pose"] = pose7(p, mat_to_quat(R))
    t = pb.target
    inputs = {
        "objects": objects,
        "initial_poses": {o: _pl_pose(pb.poses[o][0]) for o in pb.obj_keys},
        "articulation": {"door_joint": round(art["door"], 6), "rack1_joint_upper": round(art["rack1"], 6),
                         "rack0_joint_lower": round(art["rack0"], 6)},
        "target_action": {
            "name": "CLOSE_DISHWASHER", "primitive": "PUSH_RACK(rack1_joint 0.40->0) then CLOSE_DOOR(door_joint "
                                                     f"{art['door']:.4f}->0)",
            "phases": t.meta["phases"], "moving_bodies": ["upper rack (rack1)", "door", "gripper",
                                                          "objects resting on the upper rack"],
            "trajectory_ref": "inputs.npz:TARGET", "n_samples": int(t.n),
        },
        "placement_candidates": cands,
        "candidate_interventions": ivs,
    }
    per_action = {}
    for iv in pb.ivs:
        ex, bl = pb.exec_info(iv, pb.s0)
        per_action[iv.id] = {
            "executable_initially": ex,
            "blockers_initially": sorted({o for o, p in bl if p is not None}),
            "min_clearance_per_entity_initially": {o: pb.tab[iv.id][o][0].to_dict() for o in pb.tab[iv.id]},
            "fixed_environment": iv.env_profile.to_dict() if iv.env_profile else None,
        }
    per_action["TARGET"] = {"F_initial": res["F_initial"],
                            "min_clearance_per_entity_initially": {o: pb.ttab[o][0].to_dict() for o in pb.obj_keys}}
    labels = {k: res[k] for k in LABEL_KEYS if k in res}
    labels["per_action"] = per_action
    labels["direct_target_blocker_details"] = res["direct_target_blocker_details"]
    rec = {
        "schema": SCHEMA,
        "scene_id": pb.spec.scene_id, "split": pb.spec.split, "variant": variant if variant_ok else None,
        "variant_requested": pb.spec.variant_requested, "variant_satisfied": variant_ok,
        "fixture_id": "RoboCasa:Dishwasher054",
        "story": pb.spec.story,
        "spec": pb.spec.to_dict(),
        "INPUT_KEYS_ALLOWED_FOR_MODELS": list(INPUT_KEYS),
        "LABEL_KEYS_SUPERVISION_ONLY": list(LABEL_KEYS),
        "inputs": inputs,
        "labels": labels,
        "quality": quality,
        "constants": {"distance_cap_m": CAP, "conflict_if_min_dist_le": -PEN_TOL, "clear_if_min_dist_ge": CLEAR_TOL,
                      "lift_clearance_m": LIFT_CLEAR, "approach_m": APPROACH_H, "retreat_m": RETREAT_H,
                      "coarse_spacing_m": COARSE_SPACING, "fine_spacing_m": FINE_SPACING, "max_repair_depth": 4},
    }
    arr_in = {}
    for iv in [*pb.ivs]:
        tr = iv.traj
        pos, yaw = tr.carried[iv.obj]
        arr_in[f"{iv.id}/tau"] = tr.tau
        arr_in[f"{iv.id}/gripper_pos"] = tr.grip_pos
        arr_in[f"{iv.id}/gripper_quat_wxyz"] = tr.grip_quat
        arr_in[f"{iv.id}/finger_opening"] = tr.opening
        arr_in[f"{iv.id}/object_pos"] = pos
        arr_in[f"{iv.id}/object_yaw"] = yaw
        arr_in[f"{iv.id}/carried_mask"] = tr.carry_mask[iv.obj]
    arr_in["TARGET/tau"] = t.tau
    arr_in["TARGET/gripper_pos"] = t.grip_pos
    arr_in["TARGET/gripper_quat_wxyz"] = t.grip_quat
    arr_in["TARGET/rack1_joint"] = t.joints["rack1"]
    arr_in["TARGET/door_joint"] = t.joints["door"]
    arr_lab = {}
    for iv in pb.ivs:
        for o, profs in pb.tab[iv.id].items():
            for k, p in enumerate(profs):
                arr_lab[f"{iv.id}/{o}@{k}"] = p.dist
    for o, profs in pb.ttab.items():
        for k, p in enumerate(profs):
            arr_lab[f"TARGET/{o}@{k}"] = p.dist
    return rec, arr_in, arr_lab


def write_record(outdir: Path, rec: dict, arr_in: dict, arr_lab: dict, name="oracle.json"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / name).write_text(json.dumps(rec, indent=1, default=_np))
    np.savez_compressed(outdir / "inputs.npz", **{k: np.asarray(v) for k, v in arr_in.items()})
    np.savez_compressed(outdir / "labels.npz", **{k: np.asarray(v) for k, v in arr_lab.items()})


def _np(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))
