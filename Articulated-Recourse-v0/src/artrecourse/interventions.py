"""Scene specification, candidate interventions and the interventional conflict tables (v0.1).

A scene has ONE articulation state and ONE prescribed atomic target action (PUSH_RACK or
CLOSE_DOOR); the oracle query is F(s, a_target). Every object pose is an exact placement on a
named semantic slot with a named orientation template (topology.py); there are no free offsets.

State s = pose index of every movable object (0 = initial, k = destination of the k-th
intervention on that object). An intervention I_p = (object, exact destination pose, RELOCATE).
Because every sweep is evaluated against each static entity separately, executability and
F factorise exactly into per-(action, entity, entity-pose) tables:

    exec(p, s) = admissible(p) and s[obj_p] == 0 and for all o != obj_p: sweep(p) vs o@s[o] clear
    F(s)       = exists o: target vs o@s[o] not clear
                 or exists (o carried by the target, o' static): carried o vs o' not clear

`full_check_*` recompute the same quantities by brute force in explicit states (tests).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from .assets import geom_local_points, selected_assets
from .placements import Placement, check_placement, footprint, measure_supports
from .primitives import COARSE_SPACING, grasp_candidates, relocate, target_trajectory, transfer_bottom
from .scene import ObjectInstance, Scene
from .sweep import CAP, ENV_TOL, PEN_TOL, Profile, SweepEngine, status_of
from .topology import build_topology, lanes, orientation_pose

RACK_SUPPORTS = {"rack1": ("rack_left_bay", "rack_right_bay", "rack_tines"), "rack0": ()}
DEFAULT_TARGET = {"skill": "PUSH_RACK", "rack": "rack1"}
DEFAULT_ARTICULATION = {"door": "max", "rack1": 0.39, "rack0": 0.0}


@dataclass
class ObjSpec:
    key: str
    asset_id: str
    slot: str
    orient: str = "upright"
    dxy: tuple = (0.0, 0.0)          # must stay (0, 0) for canonical scenes (gate)


@dataclass
class InterventionSpec:
    obj: str
    slot: str
    orient: str = "upright"
    dxy: tuple = (0.0, 0.0)
    name: str = ""

    @property
    def id(self) -> str:
        if self.name:
            return self.name
        o = "" if self.orient in ("upright", "handle_back") else f"_{self.orient}"
        return f"move_{self.obj}_to_{self.slot}{o}"


@dataclass
class SceneSpec:
    scene_id: str
    objects: list
    interventions: list
    split: str = "canonical"
    variant_requested: str = ""
    story: str = ""
    seed: int | None = None
    target: dict = field(default_factory=lambda: dict(DEFAULT_TARGET))
    articulation: dict = field(default_factory=lambda: dict(DEFAULT_ARTICULATION))

    def to_dict(self):
        return {"scene_id": self.scene_id, "split": self.split, "variant_requested": self.variant_requested,
                "story": self.story, "seed": self.seed, "target": self.target, "articulation": self.articulation,
                "objects": [asdict(o) for o in self.objects],
                "interventions": [asdict(i) | {"id": i.id} for i in self.interventions]}

    @staticmethod
    def from_dict(d):
        return SceneSpec(d["scene_id"], [ObjSpec(**o) for o in d["objects"]],
                         [InterventionSpec(**{k: v for k, v in i.items() if k != "id"}) for i in d["interventions"]],
                         d.get("split", ""), d.get("variant_requested", ""), d.get("story", ""), d.get("seed"),
                         d.get("target", dict(DEFAULT_TARGET)), d.get("articulation", dict(DEFAULT_ARTICULATION)))


@dataclass
class Intervention:
    id: str
    obj: str
    pose_idx: int
    placement: Placement
    src: Placement
    grasp: object
    traj: object = None
    admissible: bool = True
    rejection: list = field(default_factory=list)
    env_profile: Profile | None = None
    robot: dict = field(default_factory=dict)
    grasp_candidates: list = field(default_factory=list)
    grasp_trials: list = field(default_factory=list)
    lips_crossed: list = field(default_factory=list)


def mechanism_of(part: str) -> str:
    """Moving part responsible for a conflict: gripper | carried_object | articulated_fixture."""
    p = part.split("|")[0]
    if p.startswith("gripper"):
        return "gripper"
    if p.startswith("carried:"):
        return "carried_object"
    if p.startswith("object:"):
        return "rack_borne_object"        # object carried by the articulated rack (target only)
    if p in ("rack0", "rack1", "door"):
        return "articulated_fixture"
    return p or "unknown"


class RecourseProblem:
    def __init__(self, spec: SceneSpec, spacing: float = COARSE_SPACING, with_robot: bool = True,
                 scene: Scene | None = None, grasp_override: dict | None = None):
        self.spec = spec
        self.spacing = spacing
        self.with_robot = with_robot
        self.grasp_override = grasp_override or {}
        assets = selected_assets()
        self.obj_keys = [o.key for o in spec.objects]
        self.scene = scene or Scene([ObjectInstance(o.key, assets[o.asset_id]) for o in spec.objects],
                                    with_robot=with_robot)
        self.scene.set_scene_articulation(spec.articulation)
        self.target_spec = spec.target
        self.manip = spec.target.get("rack") if spec.target["skill"] == "PUSH_RACK" else "door"
        self.sup = measure_supports(self.scene)
        self.topo = build_topology(self.scene)
        self.lanes = lanes(self.scene)
        self.eng = SweepEngine(self.scene)
        self.problems: list[str] = []
        self.gate_violations: list[str] = []
        self.track: set | None = None
        self.poses: dict[str, list[Placement]] = {}
        for o in spec.objects:
            self.poses[o.key] = [self._placement(o.key, o.slot, o.orient, o.dxy)]
        self.ivs: list[Intervention] = []
        seen = set()
        for s in spec.interventions:
            pl = self._placement(s.obj, s.slot, s.orient, s.dxy)
            sig = (s.obj, pl.key())
            if sig in seen:
                self.problems.append(f"duplicate intervention {s.id}")
                continue
            seen.add(sig)
            self.poses[s.obj].append(pl)
            meta = self.scene.objects[s.obj].asset.meta
            cands = grasp_candidates(self.scene.objects[s.obj].asset.category, meta, self._canon_verts(s.obj))
            iv = Intervention(s.id, s.obj, len(self.poses[s.obj]) - 1, pl, self.poses[s.obj][0], cands[0])
            iv.grasp_candidates = cands
            self.ivs.append(iv)
        self.iv_by_id = {iv.id: iv for iv in self.ivs}

    # ------------------------------------------------------------------ placements
    def _placement(self, key, slot_id, orient, dxy=(0.0, 0.0)) -> Placement:
        a = self.scene.objects[key].asset
        if slot_id not in self.topo:
            raise KeyError(f"{key}: unknown slot {slot_id}")
        slot = self.topo[slot_id]
        if a.category not in slot.categories:
            self.gate_violations.append(f"category {a.category} not allowed on slot {slot_id}")
        if orient not in slot.orientations.get(a.category, []):
            self.gate_violations.append(f"orientation {orient} not a template of {slot_id} for {a.category}")
        if not slot.reachable and any(i.obj == key and i.slot == slot_id for i in self.spec.interventions):
            self.gate_violations.append(f"{slot_id} is not reachable and cannot be a destination")
        if tuple(dxy) != (0.0, 0.0):
            self.gate_violations.append(f"ad-hoc offset {dxy} on {key}@{slot_id}")
        pos, yaw = orientation_pose(slot, orient, a.meta, self.lanes)
        pos = pos + np.array([dxy[0], dxy[1], 0.0])
        return Placement(slot_id, slot.support_id, pos, yaw, orient)

    def _canon_verts(self, key):
        from .assets import object_collision_vertices
        from .geom import rot_z

        a = self.scene.objects[key].asset
        v, _ = object_collision_vertices(a)
        return v @ rot_z(a.meta["canonical_yaw"]).T - np.array(a.meta["anchor"])

    def pose_tuple(self, key, k):
        p = self.poses[key][k]
        return (p.pos, p.yaw)

    def carried_by_target(self, key, k) -> bool:
        return self.manip in RACK_SUPPORTS and self.poses[key][k].support in RACK_SUPPORTS[self.manip]

    on_rack = carried_by_target

    @property
    def s0(self):
        return tuple(0 for _ in self.obj_keys)

    # ------------------------------------------------------------------ static validation
    def validate_static(self) -> list[str]:
        errs = []
        for key in self.obj_keys:
            ok, r, _ = check_placement(self.scene, self.sup, key, self.poses[key][0])
            if not ok:
                errs.append(f"initial pose of {key} invalid: {r}")
        occupied = {}
        for key in self.obj_keys:
            sl = self.poses[key][0].slot
            if sl in occupied:
                self.gate_violations.append(f"capacity: {key} and {occupied[sl]} both on {sl}")
            occupied[sl] = key
        for a, b in itertools.combinations(self.obj_keys, 2):
            dist = self.eng.static_distance(a, self.pose_tuple(a, 0), b, self.pose_tuple(b, 0))
            if dist < CLEAR_STATIC:
                errs.append(f"initial objects {a} and {b} interpenetrate / touch ({dist:+.4f} m)")
        for iv in self.ivs:
            ok, r, _ = check_placement(self.scene, self.sup, iv.obj, iv.placement)
            if not ok:
                iv.admissible = False
                iv.rejection.append(f"destination invalid: {r}")
            if iv.placement.slot == iv.src.slot and iv.placement.orient == iv.src.orient:
                iv.admissible = False
                iv.rejection.append("destination equals source")
        self.problems += errs
        for k in self.obj_keys:
            self.scene.hide_object(k)
        return errs

    def slot_occupant(self, state, slot_id, exclude=None):
        for i, o in enumerate(self.obj_keys):
            if o != exclude and self.poses[o][state[i]].slot == slot_id:
                return o
        return None

    # ------------------------------------------------------------------ trajectories
    def _lips(self):
        """Support-boundary geometry a carried object must be lifted over: the pulled rack's walls
        and tine plates, the staging tray's rims, the countertop edge. (Contact with any other
        fixed geometry -- dishwasher body, door, cabinets -- is a fixed-environment collision and
        makes the action inadmissible instead.)"""
        m = self.scene.model
        ix = self.scene.idx
        lip_geoms = set(ix.articulated[self.manip] if self.manip in ("rack0", "rack1") else ix.articulated["rack1"])
        lip_geoms |= set(ix.env_groups.get("tray", [])) | set(ix.env_groups.get("countertop", []))
        out = []
        for g, p, R in zip(*self._env_static):
            if g not in lip_geoms:
                continue
            pts = p + geom_local_points(m, g) @ R.T
            lo, hi = pts.min(0), pts.max(0)
            out.append((self._geom_part(g), lo[0], hi[0], lo[1], hi[1], hi[2]))
        return out

    def build_trajectories(self):
        """Trajectories with the derived transfer height + deterministic grasp selection
        (first candidate clear of the fixed environment and robot-admissible)."""
        self._env_static = self.eng.static_geoms(self.scene.idx.env_geoms + [
            g for k in ("door", "rack0", "rack1") for g in self.scene.idx.articulated[k]])
        self._env_parts = [self._geom_part(g) for g in self._env_static[0]]
        lips = self._lips()
        for iv in self.ivs:
            meta = self.scene.objects[iv.obj].asset.meta
            cat = self.scene.objects[iv.obj].asset.category
            radius = float(np.max(np.abs(np.r_[meta["canonical_xy_min"], meta["canonical_xy_max"]])))
            zs, zd = self.sup[iv.src.support].z, self.sup[iv.placement.support].z
            carry, iv.lips_crossed = transfer_bottom(iv.src.pos[:2], iv.placement.pos[:2], zs, zd, radius, lips)
            chosen = None
            cands = iv.grasp_candidates
            if iv.id in self.grasp_override:
                cands = [g for g in cands if g.name == self.grasp_override[iv.id]]
            for g in (cands if iv.admissible else []):
                traj = relocate(iv.obj, meta, cat, iv.src.pos, iv.src.yaw, iv.placement.pos, iv.placement.yaw,
                                zs, zd, g, spacing=self.spacing, carry_bottom=carry)
                envp = self._env_profile(iv.obj, traj)
                trial = {"grasp": g.name, "env_min_dist": round(envp.min_dist, 4), "env_part": envp.part_star}
                iv.grasp_trials.append(trial)
                if envp.min_dist < -ENV_TOL:
                    trial["result"] = "fixed-environment collision"
                    continue
                iv.grasp, iv.traj, iv.env_profile = g, traj, envp
                if self.with_robot and iv.id not in self.grasp_override:
                    rob = self._robot_for(traj)
                    if rob is None:
                        trial["result"] = "robot inadmissible (IK)"
                        continue
                    iv.robot = rob
                trial["result"] = "selected"
                chosen = g
                break
            if chosen is None:
                if iv.admissible:
                    iv.rejection.append("no grasp candidate is clear of the fixed environment and robot-admissible: "
                                        + "; ".join(f"{t['grasp']}: {t['result']}" for t in iv.grasp_trials))
                iv.admissible = False
                iv.grasp = iv.grasp_candidates[0]
                iv.traj = relocate(iv.obj, meta, cat, iv.src.pos, iv.src.yaw, iv.placement.pos, iv.placement.yaw,
                                   zs, zd, iv.grasp, spacing=self.spacing, carry_bottom=carry)
                iv.env_profile = self._env_profile(iv.obj, iv.traj)
        self.target = self._target_traj(self.s0)

    def _env_profile(self, obj, traj):
        ms = self.eng.moving_set(traj)
        mask = np.ones((traj.n, len(ms.geoms)), bool)
        for j, part in enumerate(ms.parts):
            if part.startswith("carried:"):
                mask[:, j] = traj.carry_mask[obj]
        return self.eng.profile(ms, *self._env_static, part_mask=mask, s_parts=self._env_parts)

    def _target_traj(self, state):
        carried = {k: self.pose_tuple(k, state[i]) for i, k in enumerate(self.obj_keys)
                   if self.carried_by_target(k, state[i])}
        return target_trajectory(self.scene, self.target_spec, carried, spacing=self.spacing)

    # ------------------------------------------------------------------ robot admissibility
    def _solve_group(self, names, traj, extra_env=(), stations=None, joints=False):
        sc = self.scene
        for st in (stations or sc.world["robot_stations"]):
            sc.set_station(st)
            q, ok, sols = None, True, {}
            for n in names:
                if joints:
                    i = int(n.split("_")[1])
                    sc.set_articulation(**{k: v[i] for k, v in traj.joints.items()})
                p, R = traj.waypoints[n]
                r = sc.kin.solve(p, R, q0=q, seeds=8, iters=250)
                if not r.ok or sc.kin.self_collision(r.q):
                    ok = False
                    break
                hit, _ = sc.kin.env_collision(r.q, list(sc.idx.env_geoms) + list(extra_env), skip_hand=True)
                if hit:
                    ok = False
                    break
                q = r.q
                sols[n] = r.q.round(4).tolist()
            if joints:
                sc.set_articulation()
            if ok:
                return st, sols
        return None, {}

    def _robot_for(self, traj):
        """Relocation admissibility: every waypoint needs a collision-free IK solution
        (self-collision + arm vs fixed environment; the hand is part of the swept volume).
        (pregrasp, grasp) share a base station and so do (place, retreat)."""
        sc = self.scene
        sc.set_articulation()
        art = [g for k in ("door", "rack0", "rack1") for g in sc.idx.articulated[k]]
        out = {"stations": {}, "ik": {}}
        for gname, names in [("pick", ["pregrasp", "grasp"]), ("lift", ["lift"]), ("transfer", ["transfer_mid"]),
                             ("above_place", ["above_place"]), ("place", ["place", "retreat"])]:
            st, q = self._solve_group(names, traj, art)
            if st is None:
                return None
            out["stations"][gname] = st
            out["ik"].update(q)
        return out

    def check_robot(self):
        """ACCEPTANCE GATE: the atomic target skill must be fully IK-admissible (every waypoint,
        ONE base station for the whole skill, arm vs fixed environment except the manipulated
        body). It is the same robot motion in every state, so one check covers 'after repair'."""
        sc = self.scene
        if sc.kin is None:
            return
        t = self.target
        names = sorted(t.waypoints, key=lambda n: int(n.split("_")[1]))
        extra = [g for k in ("door", "rack0", "rack1") if k != self.manip for g in sc.idx.articulated[k]]
        st, sols = self._solve_group(names, t, extra, joints=True)
        sc.set_station("dishwasher_right")
        self.target_robot = {"skill": t.meta["skill"], "station": st, "n_waypoints": len(names),
                             "fully_admissible": st is not None, "ik": sols}
        if st is None:
            self.problems.append("target action is not fully robot-admissible")

    # ------------------------------------------------------------------ conflict tables
    def compute_tables(self):
        eng = self.eng
        self.tab = {}
        for iv in self.ivs:
            ms = eng.moving_set(iv.traj)
            mask = np.ones((iv.traj.n, len(ms.geoms)), bool)
            for j, part in enumerate(ms.parts):
                if part.startswith("carried:"):
                    mask[:, j] = iv.traj.carry_mask[iv.obj]
            self.tab[iv.id] = {}
            for o in self.obj_keys:
                if o == iv.obj:
                    continue
                self.tab[iv.id][o] = []
                for k in range(len(self.poses[o])):
                    g, p, R = eng.object_geoms_at(o, *self.pose_tuple(o, k))
                    self.tab[iv.id][o].append(eng.profile(ms, g, p, R, part_mask=mask))
        self._target_tables()

    def _geom_part(self, g):
        ix = self.scene.idx
        for k, v in ix.env_groups.items():
            if g in v:
                return f"env:{k}"
        for k, v in ix.articulated.items():
            if g in v:
                return f"fixture:{k}"
        return "env"

    def _offsets(self, t):
        if self.manip in ("rack0", "rack1"):
            r = t.joints[self.manip]
            return np.stack([np.zeros(t.n), r[0] - r, np.zeros(t.n)], 1)
        return np.zeros((t.n, 3))

    def _target_tables(self):
        sc, eng = self.scene, self.eng
        env_static, env_parts = self._env_static, self._env_parts
        t = self.target
        offsets = self._offsets(t)
        ms_all = eng.moving_set(t, include_gripper=True, include_articulated=(self.manip,), carried=[])
        ms_g = eng.moving_set(t, include_gripper=True, carried=[])
        keep = [i for i, g in enumerate(env_static[0]) if g not in sc.idx.articulated[self.manip]]
        env_c = (env_static[0][keep], env_static[1][keep], env_static[2][keep])
        env_c_parts = [env_parts[i] for i in keep]
        self.ttab = {}
        for o in self.obj_keys:
            self.ttab[o] = []
            for k in range(len(self.poses[o])):
                if self.carried_by_target(o, k):
                    A = eng.carried_object_set(t, o, *self.pose_tuple(o, k), offsets)
                    p1 = eng.profile(A, *env_c, s_parts=env_c_parts)
                    p2 = eng.profile(A, ms_g.geoms, ms_g.pos, ms_g.mat, s_parts=ms_g.parts)
                    prof = p1 if p1.min_dist <= p2.min_dist else p2
                    prof.dist = np.minimum(p1.dist, p2.dist)
                    prof.phases_in_conflict = sorted(set(p1.phases_in_conflict) | set(p2.phases_in_conflict))
                    prof.parts_in_conflict = sorted(set(p1.parts_in_conflict) | set(p2.parts_in_conflict))
                else:
                    g, p, R = eng.object_geoms_at(o, *self.pose_tuple(o, k))
                    prof = eng.profile(ms_all, g, p, R)
                self.ttab[o].append(prof)
        self.tpair = {}
        for o, o2 in itertools.permutations(self.obj_keys, 2):
            for k in range(len(self.poses[o])):
                if not self.carried_by_target(o, k):
                    continue
                A = eng.carried_object_set(t, o, *self.pose_tuple(o, k), offsets)
                alo, ahi = (A.pos - A.rbound[None, :, None]).min((0, 1)), (A.pos + A.rbound[None, :, None]).max((0, 1))
                for k2 in range(len(self.poses[o2])):
                    if self.carried_by_target(o2, k2):
                        continue
                    g, p, R = eng.object_geoms_at(o2, *self.pose_tuple(o2, k2))
                    r = self.scene.model.geom_rbound[g]
                    blo, bhi = (p - r[:, None]).min(0), (p + r[:, None]).max(0)
                    if np.any(alo > bhi + CAP) or np.any(blo > ahi + CAP):
                        continue
                    self.tpair[(o, k, o2, k2)] = eng.profile(A, g, p, R)
        moving_fx = set(sc.idx.articulated[self.manip])
        keep = [i for i, g in enumerate(env_static[0]) if g not in moving_fx]
        self.target_env = eng.profile(ms_g, env_static[0][keep], env_static[1][keep], env_static[2][keep],
                                      s_parts=[env_parts[i] for i in keep])
        if self.target_env.min_dist < -ENV_TOL:
            self.problems.append(f"target gripper collides with fixed environment ({self.target_env.min_dist:+.4f} m)")

    # ------------------------------------------------------------------ queries
    def obj_index(self, key):
        return self.obj_keys.index(key)

    def exec_info(self, iv: Intervention, state) -> tuple[bool, list]:
        if not iv.admissible:
            return False, [("inadmissible", None)]
        if state[self.obj_index(iv.obj)] != 0:
            return False, [("already_moved", None)]
        blockers = []
        for o, profs in self.tab[iv.id].items():
            k = state[self.obj_index(o)]
            pr = profs[k]
            if self.track is not None:
                self.track.add(("iv", iv.id, o, k))
            if pr.status != "clear":
                blockers.append((o, pr))
        return not blockers, blockers

    def executable(self, iv, state) -> bool:
        return self.exec_info(iv, state)[0]

    def target_info(self, state) -> tuple[bool, list]:
        bl = []
        for i, o in enumerate(self.obj_keys):
            pr = self.ttab[o][state[i]]
            if self.track is not None:
                self.track.add(("target", o, state[i]))
            if pr.status != "clear":
                bl.append((o, pr))
        for (o, k, o2, k2), pr in self.tpair.items():
            if state[self.obj_index(o)] == k and state[self.obj_index(o2)] == k2:
                if self.track is not None:
                    self.track.add(("pair", o, k, o2, k2))
                if pr.status != "clear":
                    bl += [(o, pr), (o2, pr)]
        return bool(bl), bl

    def F(self, state) -> int:
        return int(self.target_info(state)[0])

    def apply(self, state, iv: Intervention):
        s = list(state)
        s[self.obj_index(iv.obj)] = iv.pose_idx
        return tuple(s)

    def consulted_ambiguities(self) -> list:
        out = []
        for e in sorted(self.track or []):
            if e[0] == "iv":
                pr = self.tab[e[1]][e[2]][e[3]]
            elif e[0] == "target":
                pr = self.ttab[e[1]][e[2]]
            else:
                pr = self.tpair[e[1:]]
            if pr.status == "ambiguous":
                out.append({"entry": list(e), "min_dist": round(pr.min_dist, 5)})
        return out

    # ------------------------------------------------------------------ brute force (tests)
    def place_state(self, state):
        for i, o in enumerate(self.obj_keys):
            p = self.poses[o][state[i]]
            self.scene.set_object_canonical(o, p.pos, p.yaw)

    def full_check_intervention(self, iv: Intervention, state) -> bool:
        eng = self.eng
        ms = eng.moving_set(iv.traj)
        mask = np.ones((iv.traj.n, len(ms.geoms)), bool)
        for j, part in enumerate(ms.parts):
            if part.startswith("carried:"):
                mask[:, j] = iv.traj.carry_mask[iv.obj]
        geoms, P, R = [], [], []
        for i, o in enumerate(self.obj_keys):
            if o == iv.obj:
                continue
            g, p, r = eng.object_geoms_at(o, *self.pose_tuple(o, state[i]))
            geoms.append(g), P.append(p), R.append(r)
        pr = eng.profile(ms, np.concatenate(geoms), np.concatenate(P), np.concatenate(R), part_mask=mask)
        return iv.admissible and state[self.obj_index(iv.obj)] == 0 and pr.status == "clear"

    def full_check_target(self, state) -> int:
        sc, eng = self.scene, self.eng
        t = self._target_traj(state)
        offsets = self._offsets(t)
        env_static = eng.static_geoms(sc.idx.env_geoms + [g for k in ("door", "rack0", "rack1") if k != self.manip
                                                          for g in sc.idx.articulated[k]])
        ms_g = eng.moving_set(t, include_gripper=True, carried=[])
        ms_all = eng.moving_set(t, include_gripper=True, include_articulated=(self.manip,), carried=[])
        worst = CAP
        for i, o in enumerate(self.obj_keys):
            k = state[i]
            if self.carried_by_target(o, k):
                A = eng.carried_object_set(t, o, *self.pose_tuple(o, k), offsets)
                worst = min(worst, eng.profile(A, *env_static).min_dist,
                            eng.profile(A, ms_g.geoms, ms_g.pos, ms_g.mat).min_dist)
                for j, o2 in enumerate(self.obj_keys):
                    if o2 != o and not self.carried_by_target(o2, state[j]):
                        g, p, R = eng.object_geoms_at(o2, *self.pose_tuple(o2, state[j]))
                        worst = min(worst, eng.profile(A, g, p, R).min_dist)
            else:
                g, p, R = eng.object_geoms_at(o, *self.pose_tuple(o, k))
                worst = min(worst, eng.profile(ms_all, g, p, R).min_dist)
        return int(status_of(worst) != "clear")


CLEAR_STATIC = 0.002   # initial objects must not touch (>= 2 mm apart)


def build_problem(spec: SceneSpec, spacing=COARSE_SPACING, with_robot=True, scene=None,
                  grasp_override=None) -> RecourseProblem:
    pb = RecourseProblem(spec, spacing=spacing, with_robot=with_robot, scene=scene, grasp_override=grasp_override)
    pb.validate_static()
    pb.build_trajectories()
    if with_robot:
        pb.check_robot()
    pb.compute_tables()
    return pb


__all__ = ["ObjSpec", "InterventionSpec", "SceneSpec", "RecourseProblem", "build_problem", "PEN_TOL",
           "mechanism_of", "footprint"]
