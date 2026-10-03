"""Scene specification, candidate interventions and the interventional conflict tables.

State s = the pose index of every movable object (0 = initial pose, k = destination of the
k-th intervention on that object) plus the (fixed) articulation state. An intervention
I_p = (object, exact destination pose, RELOCATE primitive). Because every sweep is
evaluated against each static entity separately, executability factorises exactly:

    exec(p, s) = admissible(p)  and  s[obj_p] == 0  and  for all o != obj_p:
                 status(sweep(p) vs o at pose s[o]) == clear
    F(s, target) = exists o: target-vs-o(s[o]) not clear
                   or exists (o on rack, o' off rack): carried o vs o' not clear

The tables hold one swept-volume profile per (action, entity, entity-pose). They are
computed once per scene with MuJoCo; `full_check_*` recompute the same quantities by
brute force in an explicit state (used by tests to verify the factorisation).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from .assets import selected_assets
from .placements import Placement, check_placement, make_placement, measure_supports
from .primitives import COARSE_SPACING, close_dishwasher, grasp_candidates, relocate
from .scene import ObjectInstance, Scene
from .sweep import CAP, ENV_TOL, PEN_TOL, Profile, SweepEngine, status_of

RACK_SUPPORTS = ("rack_left_bay", "rack_right_bay", "rack_tines")


@dataclass
class ObjSpec:
    key: str
    asset_id: str
    slot: str
    yaw_deg: float = 0.0
    dxy: tuple = (0.0, 0.0)


@dataclass
class InterventionSpec:
    obj: str
    slot: str
    yaw_deg: float = 0.0
    dxy: tuple = (0.0, 0.0)
    name: str = ""

    @property
    def id(self) -> str:
        return self.name or f"move_{self.obj}_to_{self.slot}"


@dataclass
class SceneSpec:
    scene_id: str
    objects: list
    interventions: list
    split: str = "canonical"
    variant_requested: str = ""
    story: str = ""
    seed: int | None = None

    def to_dict(self):
        return {"scene_id": self.scene_id, "split": self.split, "variant_requested": self.variant_requested,
                "story": self.story, "seed": self.seed,
                "objects": [asdict(o) for o in self.objects],
                "interventions": [asdict(i) | {"id": i.id} for i in self.interventions]}

    @staticmethod
    def from_dict(d):
        return SceneSpec(d["scene_id"], [ObjSpec(**o) for o in d["objects"]],
                         [InterventionSpec(**{k: v for k, v in i.items() if k != "id"}) for i in d["interventions"]],
                         d.get("split", ""), d.get("variant_requested", ""), d.get("story", ""), d.get("seed"))


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


class RecourseProblem:
    def __init__(self, spec: SceneSpec, spacing: float = COARSE_SPACING, with_robot: bool = True,
                 scene: Scene | None = None, grasp_override: dict | None = None):
        self.spec = spec
        self.spacing = spacing
        self.with_robot = with_robot
        self.grasp_override = grasp_override or {}
        assets = selected_assets()
        self.obj_keys = [o.key for o in spec.objects]
        self.scene = scene or Scene([ObjectInstance(o.key, assets[o.asset_id]) for o in spec.objects], with_robot=with_robot)
        self.sup = measure_supports(self.scene)
        self.eng = SweepEngine(self.scene)
        self.problems: list[str] = []
        self.track: set | None = None   # consulted table entries (set by the oracle)
        # poses: obj -> [Placement]; index 0 = initial
        self.poses: dict[str, list[Placement]] = {}
        for o in spec.objects:
            self.poses[o.key] = [make_placement(self.scene, self.sup, o.key, o.slot, o.yaw_deg, o.dxy)]
        self.ivs: list[Intervention] = []
        seen, ids = set(), set()
        for s in spec.interventions:
            pl = make_placement(self.scene, self.sup, s.obj, s.slot, s.yaw_deg, s.dxy)
            sig = (s.obj, pl.key(), tuple(np.round(pl.pos, 4)))
            if sig in seen:
                self.problems.append(f"duplicate intervention {s.id}")
                continue
            seen.add(sig)
            self.poses[s.obj].append(pl)
            meta = self.scene.objects[s.obj].asset.meta
            cands = grasp_candidates(self.scene.objects[s.obj].asset.category, meta, self._canon_verts(s.obj))
            iid = s.id if s.id not in ids else f"{s.id}_yaw{int(round(s.yaw_deg))}"
            ids.add(iid)
            iv = Intervention(iid, s.obj, len(self.poses[s.obj]) - 1, pl, self.poses[s.obj][0], cands[0])
            iv.grasp_candidates = cands
            self.ivs.append(iv)
        self.iv_by_id = {iv.id: iv for iv in self.ivs}

    # ------------------------------------------------------------------ helpers
    def _canon_verts(self, key):
        from .assets import object_collision_vertices
        from .geom import rot_z

        a = self.scene.objects[key].asset
        v, _ = object_collision_vertices(a)
        meta = a.meta
        v = v @ rot_z(meta["canonical_yaw"]).T - np.array(meta["anchor"])
        return v

    def pose_tuple(self, key, k):
        p = self.poses[key][k]
        return (p.pos, p.yaw)

    def on_rack(self, key, k) -> bool:
        return self.poses[key][k].support in RACK_SUPPORTS

    @property
    def s0(self):
        return tuple(0 for _ in self.obj_keys)

    # ------------------------------------------------------------------ static validation
    def validate_static(self) -> list[str]:
        """Initial placements valid and non-interpenetrating; destination placements valid;
        source and destination differ."""
        errs = []
        for key in self.obj_keys:
            ok, r, _ = check_placement(self.scene, self.sup, key, self.poses[key][0])
            if not ok:
                errs.append(f"initial pose of {key} invalid: {r}")
        for a, b in itertools.combinations(self.obj_keys, 2):
            dist = self.eng.static_distance(a, self.pose_tuple(a, 0), b, self.pose_tuple(b, 0))
            if dist < CLEAR_STATIC:
                errs.append(f"initial objects {a} and {b} interpenetrate / touch ({dist:+.4f} m)")
        for iv in self.ivs:
            ok, r, info = check_placement(self.scene, self.sup, iv.obj, iv.placement)
            if not ok:
                iv.admissible = False
                iv.rejection.append(f"destination invalid: {r}")
            dp = np.linalg.norm(iv.placement.pos - iv.src.pos)
            dyaw = abs((iv.placement.yaw - iv.src.yaw + np.pi) % (2 * np.pi) - np.pi)
            if dp < 0.01 and dyaw < np.radians(5):
                iv.admissible = False
                iv.rejection.append("destination equals source")
        for key in self.obj_keys:  # floating/unsupported objects scene-level
            for k, p in enumerate(self.poses[key]):
                if p.support not in self.sup:
                    errs.append(f"{key} pose {k} on unknown support")
        self.problems += errs
        for iv in self.ivs:
            self.scene.hide_object(iv.obj)
        return errs

    # ------------------------------------------------------------------ trajectories
    def build_trajectories(self):
        """Trajectories + deterministic grasp selection (first candidate clear of the fixed
        environment and robot-admissible)."""
        self._env_static = self.eng.static_geoms(self.scene.idx.env_geoms + [
            g for k in ("door", "rack0", "rack1") for g in self.scene.idx.articulated[k]])
        self._env_parts = [self._geom_part(g) for g in self._env_static[0]]
        for iv in self.ivs:
            meta = self.scene.objects[iv.obj].asset.meta
            cat = self.scene.objects[iv.obj].asset.category
            chosen = None
            cands = iv.grasp_candidates
            if iv.id in self.grasp_override:   # re-evaluation at another resolution: same grasp, no IK
                cands = [g for g in cands if g.name == self.grasp_override[iv.id]]
            for g in (cands if iv.admissible else []):
                traj = relocate(iv.obj, meta, cat, iv.src.pos, iv.src.yaw, iv.placement.pos, iv.placement.yaw,
                                self.sup[iv.src.support].z, self.sup[iv.placement.support].z, g, spacing=self.spacing)
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
                                   self.sup[iv.src.support].z, self.sup[iv.placement.support].z, iv.grasp,
                                   spacing=self.spacing)
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
        rack = {k: self.pose_tuple(k, state[i]) for i, k in enumerate(self.obj_keys) if self.on_rack(k, state[i])}
        return close_dishwasher(self.scene, rack, spacing=self.spacing)

    # ------------------------------------------------------------------ robot admissibility
    def _solve_group(self, names, traj, extra_env=()):
        sc = self.scene
        for st in sc.world["robot_stations"]:
            sc.set_station(st)
            q, ok, sols = None, True, {}
            for n in names:
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
            if ok:
                return st, sols
        return None, {}

    def _robot_for(self, traj):
        """Relocation admissibility: every waypoint needs a collision-free IK solution
        (self-collision + arm-vs-fixed-environment; the hand itself is part of the swept
        volume). (pregrasp, grasp) share a base station and so do (place, retreat); the
        mobile base may reposition while the object is held at transfer height."""
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
        """Target articulation robot coverage. CLOSE_DISHWASHER is identical in every scene,
        so this is reported (scene independent) but is not a per-scene filter."""
        sc = self.scene
        if sc.kin is None:
            return
        t = self.target
        cover = {}
        for n in t.waypoints:
            i = int(n.split("_")[1])
            sc.set_articulation(door=t.joints["door"][i], rack1=t.joints["rack1"][i])
            manip = "rack1" if n.startswith("push") else "door"
            extra = [g for k in ("door", "rack0", "rack1") if k != manip for g in sc.idx.articulated[k]]
            st, _ = self._solve_group([n], t, extra)
            if st is None:
                ik_only = False
                for s_ in sc.world["robot_stations"]:
                    sc.set_station(s_)
                    ik_only = ik_only or sc.kin.solve(*t.waypoints[n], seeds=8, iters=250).ok
                cover[n] = "ik_ok_arm_contacts_fixture" if ik_only else "no_ik"
            else:
                cover[n] = f"ok@{st}"
        sc.set_articulation()
        sc.set_station("dishwasher_right")
        self.target_robot = {"waypoints": cover, "fully_admissible": all(v.startswith("ok") for v in cover.values())}

    # ------------------------------------------------------------------ conflict tables
    def compute_tables(self):
        sc, eng = self.scene, self.eng
        env_static, env_parts = self._env_static, self._env_parts
        self.tab = {}            # iv.id -> obj -> [Profile per pose idx]
        for iv in self.ivs:
            ms = eng.moving_set(iv.traj)
            mask = np.ones((iv.traj.n, len(ms.geoms)), bool)
            cm = iv.traj.carry_mask[iv.obj]
            for j, part in enumerate(ms.parts):
                if part.startswith("carried:"):
                    mask[:, j] = cm
            self.tab[iv.id] = {}
            for o in self.obj_keys:
                if o == iv.obj:
                    continue
                self.tab[iv.id][o] = []
                for k in range(len(self.poses[o])):
                    g, p, R = eng.object_geoms_at(o, *self.pose_tuple(o, k))
                    self.tab[iv.id][o].append(eng.profile(ms, g, p, R, part_mask=mask))
        self._target_tables(env_static, env_parts)

    def _geom_part(self, g):
        ix = self.scene.idx
        for k, v in ix.env_groups.items():
            if g in v:
                return f"env:{k}"
        for k, v in ix.articulated.items():
            if g in v:
                return f"fixture:{k}"
        return "env"

    def _target_tables(self, env_static, env_parts):
        sc, eng = self.scene, self.eng
        t = self.target
        r0 = sc.articulation_targets()["rack1"]
        offsets = np.stack([np.zeros(t.n), r0 - t.joints["rack1"], np.zeros(t.n)], 1)
        ms_all = eng.moving_set(t, include_gripper=True, include_articulated=("rack1", "door"), carried=[])
        ms_dg = eng.moving_set(t, include_gripper=True, include_articulated=("door",), carried=[])
        # environment for rack-carried objects: everything static except the rack they rest on
        keep = [i for i, g in enumerate(env_static[0]) if g not in sc.idx.articulated["rack1"]
                and g not in sc.idx.articulated["door"]]
        env_c = (env_static[0][keep], env_static[1][keep], env_static[2][keep])
        env_c_parts = [env_parts[i] for i in keep]
        self.ttab = {}
        for o in self.obj_keys:
            self.ttab[o] = []
            for k in range(len(self.poses[o])):
                if self.on_rack(o, k):
                    A = eng.carried_object_set(t, o, *self.pose_tuple(o, k), offsets)
                    p1 = eng.profile(A, *env_c, s_parts=env_c_parts)
                    p2 = eng.profile(A, ms_dg.geoms, ms_dg.pos, ms_dg.mat, s_parts=ms_dg.parts)
                    prof = p1 if p1.min_dist <= p2.min_dist else p2
                    prof.dist = np.minimum(p1.dist, p2.dist)
                    prof.phases_in_conflict = sorted(set(p1.phases_in_conflict) | set(p2.phases_in_conflict))
                    prof.parts_in_conflict = sorted(set(p1.parts_in_conflict) | set(p2.parts_in_conflict))
                else:
                    g, p, R = eng.object_geoms_at(o, *self.pose_tuple(o, k))
                    prof = eng.profile(ms_all, g, p, R)
                self.ttab[o].append(prof)
        # pairwise: rack-carried object vs off-rack object (only if their regions can meet)
        self.tpair = {}
        for o, o2 in itertools.permutations(self.obj_keys, 2):
            for k in range(len(self.poses[o])):
                if not self.on_rack(o, k):
                    continue
                A = eng.carried_object_set(t, o, *self.pose_tuple(o, k), offsets)
                alo, ahi = (A.pos - A.rbound[None, :, None]).min((0, 1)), (A.pos + A.rbound[None, :, None]).max((0, 1))
                for k2 in range(len(self.poses[o2])):
                    if self.on_rack(o2, k2):
                        continue
                    g, p, R = eng.object_geoms_at(o2, *self.pose_tuple(o2, k2))
                    r = self.scene.model.geom_rbound[g]
                    blo, bhi = (p - r[:, None]).min(0), (p + r[:, None]).max(0)
                    if np.any(alo > bhi + CAP) or np.any(blo > ahi + CAP):
                        continue
                    self.tpair[(o, k, o2, k2)] = eng.profile(A, g, p, R)
        # the target's own moving parts vs the fixed environment (gripper only; door/rack
        # articulation is clean by construction, checked in the fixture report)
        ms_g = eng.moving_set(t, include_gripper=True, carried=[])
        moving_fx = set(sc.idx.articulated["rack1"]) | set(sc.idx.articulated["door"])
        keep = [i for i, g in enumerate(env_static[0]) if g not in moving_fx]
        self.target_env = eng.profile(ms_g, env_static[0][keep], env_static[1][keep], env_static[2][keep],
                                      s_parts=[env_parts[i] for i in keep])
        if self.target_env.min_dist < -ENV_TOL:
            self.problems.append(f"target gripper collides with fixed environment ({self.target_env.min_dist:+.4f} m)")

    # ------------------------------------------------------------------ queries
    def obj_index(self, key):
        return self.obj_keys.index(key)

    def exec_info(self, iv: Intervention, state) -> tuple[bool, list]:
        """(executable, blocking entities with profiles) in state."""
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
        """(F == 1 i.e. infeasible, direct blockers)."""
        bl = []
        for i, o in enumerate(self.obj_keys):
            pr = self.ttab[o][state[i]]
            if self.track is not None:
                self.track.add(("target", o, state[i]))
            if pr.status != "clear":
                bl.append((o, pr))
        for (o, k, o2, k2), pr in self.tpair.items():
            if state[self.obj_index(o)] == k and state[self.obj_index(o2)] == k2 and self.track is not None:
                self.track.add(("pair", o, k, o2, k2))
            if state[self.obj_index(o)] == k and state[self.obj_index(o2)] == k2 and pr.status != "clear":
                bl.append((o, pr))
                bl.append((o2, pr))
        return bool(bl), bl

    def F(self, state) -> int:
        return int(self.target_info(state)[0])

    def apply(self, state, iv: Intervention):
        s = list(state)
        s[self.obj_index(iv.obj)] = iv.pose_idx
        return tuple(s)

    def consulted_ambiguities(self) -> list:
        """Ambiguous (|d| < 4 mm) table entries that the oracle actually consulted."""
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
        """Recompute executability by a fresh sweep against all objects in `state`."""
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
        """Recompute F by rebuilding the target trajectory for `state` and sweeping again."""
        sc, eng = self.scene, self.eng
        t = self._target_traj(state)
        r0 = sc.articulation_targets()["rack1"]
        offsets = np.stack([np.zeros(t.n), r0 - t.joints["rack1"], np.zeros(t.n)], 1)
        env_static = eng.static_geoms(sc.idx.env_geoms + [g for k in ("rack0",) for g in sc.idx.articulated[k]])
        ms_dg = eng.moving_set(t, include_gripper=True, include_articulated=("door",), carried=[])
        ms_all = eng.moving_set(t, include_gripper=True, include_articulated=("rack1", "door"), carried=[])
        worst = CAP
        for i, o in enumerate(self.obj_keys):
            k = state[i]
            if self.on_rack(o, k):
                A = eng.carried_object_set(t, o, *self.pose_tuple(o, k), offsets)
                worst = min(worst, eng.profile(A, *env_static).min_dist,
                            eng.profile(A, ms_dg.geoms, ms_dg.pos, ms_dg.mat).min_dist)
                for j, o2 in enumerate(self.obj_keys):
                    if o2 != o and not self.on_rack(o2, state[j]):
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


__all__ = ["ObjSpec", "InterventionSpec", "SceneSpec", "RecourseProblem", "build_problem", "PEN_TOL"]
