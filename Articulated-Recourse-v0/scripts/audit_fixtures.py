"""Audit every RoboCasa dishwasher fixture and select one by explicit measured criteria.

Geometric audit (all fixtures, fixture frame): dimensions, door / rack joint ranges, door
opening angle, the largest loading pull of each rack that keeps >= 5 mm from the fully
opened door, how much of each pulled rack is reachable from above (in front of the tub
front and of the counter overhang), vertical clearances, self-collision in the loading state.

Robot audit (geometrically viable fixtures, inside the fitted kitchen): for each atomic target
skill (PUSH_RACK upper, PUSH_RACK lower, CLOSE_DOOR) search single base stations for which
EVERY waypoint has a collision-free IK solution (self-collision + arm vs fixed environment,
the manipulated body excluded).

    python scripts/audit_fixtures.py            -> out/fixture_audit.json, out/fixture_audit.md
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from artrecourse import OUT  # noqa: E402
from artrecourse.assets import COLLISION_GROUP, geom_local_points, load_asset_config  # noqa: E402
from artrecourse.exactgeom import exact_signed_distance, hull_normals  # noqa: E402
from artrecourse.fixture import fixture_bounds, load_fixture_spec  # noqa: E402

COUNTER_OVERHANG = 0.05      # countertop edge in front of the fixture front (scene.adapt_world)


# ----------------------------------------------------------------------------- geometry
class Fx:
    def __init__(self, fid):
        self.fid = fid
        spec, self.regions = load_fixture_spec(fid)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.body_geoms = {}
        for g in range(m.ngeom):
            if m.geom_group[g] == COLLISION_GROUP:
                self.body_geoms.setdefault(m.body(m.geom_bodyid[g]).name, []).append(g)
        self.local = {g: (geom_local_points(m, g), hull_normals(geom_local_points(m, g)))
                      for gs in self.body_geoms.values() for g in gs}
        self.j = {n: m.joint(n).id for n in ("door_joint", "rack0_joint", "rack1_joint")}

    def set(self, door=0.0, rack0=0.0, rack1=0.0):
        d, m = self.d, self.m
        d.qpos[:] = 0
        for n, v in (("door_joint", door), ("rack0_joint", rack0), ("rack1_joint", rack1)):
            d.qpos[m.jnt_qposadr[self.j[n]]] = v
        mujoco.mj_kinematics(m, d)

    def world_pts(self, g):
        return self.d.geom_xpos[g] + self.local[g][0] @ self.d.geom_xmat[g].reshape(3, 3).T

    def dist(self, ga, gb):
        ft = np.zeros(6)
        dd = mujoco.mj_geomDistance(self.m, self.d, ga, gb, 0.05, ft)
        if dd < 0.012:
            Ra, Rb = self.d.geom_xmat[ga].reshape(3, 3), self.d.geom_xmat[gb].reshape(3, 3)
            dd = exact_signed_distance(self.world_pts(ga), self.world_pts(gb), self.local[ga][1] @ Ra.T,
                                       self.local[gb][1] @ Rb.T)
        return dd

    def body_dist(self, a, b):
        return min(self.dist(x, y) for x in self.body_geoms[a] for y in self.body_geoms[b])

    def aabb(self, body):
        p = np.concatenate([self.world_pts(g) for g in self.body_geoms[body]])
        return p.min(0), p.max(0)

    def rack_floor(self, body):
        """Largest thin horizontal plate of the rack (its floor)."""
        best = None
        for g in self.body_geoms[body]:
            p = self.world_pts(g)
            lo, hi = p.min(0), p.max(0)
            ext = hi - lo
            if ext[2] < 0.012 and ext[0] * ext[1] > (0 if best is None else (best[1][0] - best[0][0]) * (best[1][1] - best[0][1])):
                best = (lo, hi)
        if best is None or (best[1][0] - best[0][0]) < 0.3:     # wire racks modelled without a floor plate
            lo, hi = self.aabb(body)
            best = (lo, np.array([hi[0], hi[1], lo[2] + 0.004]))
        return best


def geometric_audit(fid):
    f = Fx(fid)
    m = f.m
    rng = {n: m.jnt_range[f.j[n]].tolist() for n in f.j}
    qd = rng["door_joint"][1]
    f.set()
    lo, hi = fixture_bounds(fid)
    door_lo, door_hi = f.aabb("door")
    body_front = float(np.min([f.world_pts(g)[:, 1].min() for g in f.body_geoms["object"]]))
    counter_edge = lo[1] - COUNTER_OVERHANG
    tub_front = float(np.percentile([f.world_pts(g)[:, 1].min() for g in f.body_geoms["object"]], 25))
    f.set(door=qd)
    R0 = np.eye(3)
    Rd = f.d.xmat[m.body("door").id].reshape(3, 3)
    door_angle = float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ Rd) - 1) / 2, -1, 1))))
    out = {"fixture_id": fid, "dims_m": (hi - lo).round(3).tolist(), "joint_ranges": rng,
           "door_open_angle_deg": round(door_angle, 1), "racks": {}}
    for rack, other in (("rack1", "rack0"), ("rack0", "rack1")):
        rmax = rng[f"{rack}_joint"][1]
        pull = 0.0
        for p in np.arange(rmax, -1e-9, -0.01):
            f.set(door=qd, **{rack: p})
            if f.body_dist(rack, "door") >= 0.005:
                pull = float(p)
                break
        f.set(door=qd, **{rack: pull})
        fl = f.rack_floor(rack)
        inner_front = fl[0][1]
        reach_limit = min(tub_front, counter_edge) if rack == "rack1" else tub_front
        exposed = max(0.0, reach_limit - inner_front)
        f.set(door=qd, **{rack: pull})
        sc_door = f.body_dist(rack, "door")
        sc_other = f.body_dist(rack, other)
        f.set()
        rfl = f.rack_floor(rack)
        if rack == "rack1":
            above = [g for g in f.body_geoms["object"] if f.world_pts(g)[:, 2].min() > rfl[1][2] + 0.02]
            ceil = min(f.world_pts(g)[:, 2].min() for g in above) if above else np.nan
        else:
            ceil = f.aabb("rack1")[0][2]
        out["racks"][rack] = {
            "range_m": rmax, "max_loading_pull_m": round(pull, 3), "pull_fraction": round(pull / rmax, 2) if rmax else 0,
            "rack_depth_m": round(float(fl[1][1] - fl[0][1]), 3), "rack_width_m": round(float(fl[1][0] - fl[0][0]), 3),
            "top_down_reachable_depth_m": round(float(exposed), 3),
            "vertical_clearance_m": round(float(ceil - rfl[1][2]), 3),
            "clearance_to_open_door_m": round(float(sc_door), 4), "clearance_to_other_rack_m": round(float(sc_other), 4),
        }
    f.set(door=qd)
    out["door_open_vs_body_m"] = round(float(f.body_dist("door", "object")), 4)
    out["door_top_edge_forward_m"] = round(float(body_front - f.aabb("door")[0][1]), 3)
    return out


# ----------------------------------------------------------------------------- robot
def robot_audit(fid, quick=False):
    from artrecourse.primitives import close_door, push_rack
    from artrecourse.scene import Scene

    sc = Scene([], with_robot=True, fixture_id=fid)
    m, d = sc.model, sc.data
    res = {}
    skills = [("PUSH_RACK_upper", "rack1"), ("PUSH_RACK_lower", "rack0"), ("CLOSE_DOOR", None)]
    geo = geometric_audit(fid)
    qd = float(sc.idx.joint_range["door"][1])
    # door top edge (world, fully open) -> pedestal must stay in front of it
    sc.set_scene_articulation({"door": "max"})
    sc.set_articulation()
    mujoco.mj_kinematics(m, d)
    door_pts = np.concatenate([d.geom_xpos[g] + geom_local_points(m, g) @ d.geom_xmat[g].reshape(3, 3).T
                               for g in sc.idx.articulated["door"]])
    door_ymin, door_xmax = float(door_pts[:, 1].min()), float(np.abs(door_pts[:, 0]).max())
    stations = []
    for x, dy, z in itertools.product((-0.35, -0.15, 0.0, 0.15, 0.35, 0.55), (0.0, 0.10, 0.22), (0.25, 0.42, 0.6, 0.75)):
        y = door_ymin - 0.175 - dy if abs(x) < door_xmax + 0.16 else door_ymin + 0.05 - dy
        stations.append((x, y, z))
    if quick:
        stations = stations[::2]
    mid = m.body_mocapid[m.body("robot_base").id]
    for name, rack in skills:
        if rack is not None:
            pull = geo["racks"][rack]["max_loading_pull_m"]
            if pull < 0.08:
                res[name] = {"applicable": False, "reason": f"rack can only be pulled {pull} m with the door open"}
                continue
            sc.set_scene_articulation({"door": "max", rack: pull})
            traj = push_rack(sc, rack, {})
        else:
            sc.set_scene_articulation({"door": "max"})
            best = None
            for frac in (0.6, 0.45, 0.75, 0.9):
                traj = close_door(sc, frac)
                ok = _best_station(sc, traj, stations, mid, "door")
                if best is None or ok[1] > best[1][1]:
                    best = (frac, ok, traj)
                if ok[1] == 1.0:
                    break
            frac, (st, cov, nst), traj = best
            res[name] = {"applicable": True, "contact_fraction": frac, "best_station": st, "coverage": cov,
                         "n_fully_admissible_stations": nst, "n_waypoints": len(traj.waypoints)}
            continue
        st, cov, nst = _best_station(sc, traj, stations, mid, rack)
        res[name] = {"applicable": True, "from_pull_m": pull, "best_station": st, "coverage": cov,
                     "n_fully_admissible_stations": nst, "n_waypoints": len(traj.waypoints)}
    return res


def _best_station(sc, traj, stations, mid, manipulated):
    from artrecourse.geom import quat_from_yaw

    m, d = sc.model, sc.data
    env = list(sc.idx.env_geoms) + [g for k in ("door", "rack0", "rack1") if k != manipulated
                                   for g in sc.idx.articulated[k]]
    names = sorted(traj.waypoints, key=lambda n: int(n.split("_")[1]))
    best, n_full = (None, 0.0), 0
    for st in stations:
        d.mocap_pos[mid] = st
        d.mocap_quat[mid] = quat_from_yaw(np.pi / 2)
        q, ok_n = None, 0
        for n in names:
            i = int(n.split("_")[1])
            sc.set_articulation(**{k: v[i] for k, v in traj.joints.items()})
            p, R = traj.waypoints[n]
            r = sc.kin.solve(p, R, q0=q, seeds=6, iters=200)
            if not r.ok or sc.kin.self_collision(r.q) or sc.kin.env_collision(r.q, env, skip_hand=False)[0]:
                # the hand touching the manipulated body is intended; other contacts are not
                if not r.ok or sc.kin.self_collision(r.q) or sc.kin.env_collision(r.q, env, skip_hand=True)[0]:
                    break
            ok_n += 1
            q = r.q
        cov = ok_n / len(names)
        if cov == 1.0:
            n_full += 1
        if cov > best[1]:
            best = ([round(float(v), 3) for v in st], cov)
    sc.set_articulation()
    return best[0], round(best[1], 3), n_full


def main():
    fids = load_asset_config()["fixture"]["audit_pool"]
    t0 = time.time()
    geo = {}
    for fid in fids:
        try:
            geo[fid] = geometric_audit(fid)
        except Exception as e:  # noqa: BLE001
            geo[fid] = {"fixture_id": fid, "error": repr(e)}
        g = geo[fid]
        if "error" not in g:
            print(f"{fid}: door {g['door_open_angle_deg']}deg  upper pull {g['racks']['rack1']['max_loading_pull_m']}"
                  f" reach {g['racks']['rack1']['top_down_reachable_depth_m']}  lower pull {g['racks']['rack0']['max_loading_pull_m']}"
                  f" reach {g['racks']['rack0']['top_down_reachable_depth_m']}", flush=True)
        else:
            print(fid, g["error"])

    def geo_score(g):
        if "error" in g:
            return -1
        u, lw = g["racks"]["rack1"], g["racks"]["rack0"]
        return (min(u["top_down_reachable_depth_m"], 0.35) + min(lw["top_down_reachable_depth_m"], 0.35)
                + 0.002 * g["door_open_angle_deg"])

    ranked = sorted(geo, key=lambda f: -geo_score(geo[f]))
    shortlist = [f for f in ranked if geo_score(geo[f]) > 0.15][:6]
    if "Dishwasher054" not in shortlist:
        shortlist.append("Dishwasher054")     # the v0 fixture is always re-audited
    rob = {}
    for fid in shortlist:
        t = time.time()
        rob[fid] = robot_audit(fid)
        print(f"{fid}: robot {json.dumps(rob[fid])} ({time.time() - t:.0f}s)", flush=True)

    def full(fid, k):
        r = rob[fid].get(k, {})
        return r.get("applicable") and r.get("coverage") == 1.0

    def final_score(fid):
        g = geo[fid]
        s = geo_score(g)
        s += 1.0 * full(fid, "PUSH_RACK_upper") + 1.0 * full(fid, "PUSH_RACK_lower") + 1.0 * full(fid, "CLOSE_DOOR")
        return s

    sel = max(shortlist, key=final_score)
    report = {
        "criteria": ["door opens >= 30 deg", "at least one rack has >= 0.20 m of top-down reachable loading depth",
                     "atomic PUSH_RACK and CLOSE_DOOR fully IK-admissible from one base station each",
                     "score = reachable upper depth + reachable lower depth (each capped at 0.35 m) + 0.002*door angle"
                     " + 1 per fully admissible atomic skill"],
        "geometric": geo, "geometric_ranking": ranked, "robot_shortlist": shortlist, "robot": rob,
        "final_scores": {f: round(final_score(f), 3) for f in shortlist}, "selected": sel,
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "fixture_audit.json").write_text(json.dumps(report, indent=1, default=float))
    md = ["# Dishwasher fixture audit", "", f"Selected: **{sel}**", "", "## Criteria", ""]
    md += [f"- {c}" for c in report["criteria"]]
    md += ["", "## Geometry (all fixtures)", "",
           "| fixture | dims (m) | door deg | upper pull / reach / clearance | lower pull / reach / clearance |",
           "|---|---|---|---|---|"]
    for fid in ranked:
        g = geo[fid]
        if "error" in g:
            md.append(f"| {fid} | error | | | |")
            continue
        u, lw = g["racks"]["rack1"], g["racks"]["rack0"]
        md.append(f"| {fid} | {g['dims_m']} | {g['door_open_angle_deg']} | {u['max_loading_pull_m']} / "
                  f"{u['top_down_reachable_depth_m']} / {u['vertical_clearance_m']} | {lw['max_loading_pull_m']} / "
                  f"{lw['top_down_reachable_depth_m']} / {lw['vertical_clearance_m']} |")
    md += ["", "## Robot (shortlist): coverage of atomic skills from the best single base station", "",
           "| fixture | PUSH_RACK upper | PUSH_RACK lower | CLOSE_DOOR | score |", "|---|---|---|---|---|"]
    for fid in shortlist:
        r = rob[fid]

        def c(k):
            x = r.get(k, {})
            return "n/a" if not x.get("applicable") else f"{x['coverage']:.2f} ({x['n_fully_admissible_stations']} st.)"

        md.append(f"| {fid} | {c('PUSH_RACK_upper')} | {c('PUSH_RACK_lower')} | {c('CLOSE_DOOR')} | {final_score(fid):.2f} |")
    (OUT / "fixture_audit.md").write_text("\n".join(md) + "\n")
    print("SELECTED", sel)


if __name__ == "__main__":
    main()
