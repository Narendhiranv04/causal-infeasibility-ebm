"""Rendering: observations (RGB / depth / instance masks), swept-volume ghost renders,
dependency-graph panels and compact h264 videos (system ffmpeg)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np

from .assets import VISUAL_GROUP
from .geom import quat_from_yaw, quat_to_mat
from .robot import Q_HOME
from .scene import GRIPPER_GROUP, ROBOT_GROUP
from .sweep import PEN_TOL

RED = np.array([0.92, 0.12, 0.10, 1.0], np.float32)
ORANGE = np.array([1.0, 0.55, 0.05, 1.0], np.float32)
GHOST = np.array([0.15, 0.65, 0.95, 0.16], np.float32)
GREEN = np.array([0.20, 0.80, 0.30, 1.0], np.float32)
MAGENTA = np.array([0.95, 0.10, 0.85, 1.0], np.float32)


# ----------------------------------------------------------------------------- io helpers
def save_jpg(path: Path, img: np.ndarray, quality: int = 85):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_png(path: Path, img: np.ndarray):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3:
        img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def caption(img, lines, color=(255, 255, 255), y0=22, scale=0.55, box=True):
    import cv2

    img = np.ascontiguousarray(img).copy()
    if box and lines:
        h = 10 + 22 * len(lines)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, y0 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


class VideoWriter:
    def __init__(self, path: Path, w: int, h: int, fps: int = 20, crf: int = 27):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def add(self, img):
        self.p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


# ----------------------------------------------------------------------------- renderer
class SceneRenderer:
    """Renders a RecourseProblem's scene in arbitrary oracle states."""

    def __init__(self, pb, width=640, height=480):
        self.pb = pb
        self.sc = pb.scene
        self.m, self.d = self.sc.model, self.sc.data
        self.w, self.h = width, height
        self.r = mujoco.Renderer(self.m, height, width)
        self.q = Q_HOME.copy()
        m = self.m
        self.fx_vis = {k: [g for g in range(m.ngeom) if m.geom_group[g] == VISUAL_GROUP
                           and m.body(m.geom_bodyid[g]).name == f"dw_{k}"] for k in ("door", "rack1")}

    def close(self):
        self.r.close()

    # ------------------------------------------------------------------ state
    def set_state(self, state, robot_station="dishwasher_right", rest=True):
        pb, sc = self.pb, self.sc
        sc.set_articulation()
        for i, o in enumerate(pb.obj_keys):
            p = pb.poses[o][state[i]]
            sc.set_object_canonical(o, p.pos, p.yaw)
        sc.hide_gripper()
        if sc.kin is not None:
            sc.set_station(robot_station)
            if rest:
                self.q = Q_HOME.copy()
                sc.kin.fk(self.q, 0.08)
        sc.forward()

    def pose_robot(self, pos, R, opening, station):
        """Try to show the real arm at this TCP pose; returns True if IK succeeded."""
        sc = self.sc
        if sc.kin is None:
            return False
        sc.set_station(station)
        r = sc.kin.solve(pos, R, q0=self.q, seeds=1, iters=60)
        if not r.ok:
            r = sc.kin.solve(pos, R, q0=None, seeds=4, iters=150)
        if r.ok:
            self.q = r.q
            sc.kin.fk(r.q, opening)
            return True
        return False

    # ------------------------------------------------------------------ drawing
    def _option(self, robot=True, gripper=False):
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[VISUAL_GROUP] = 1
        opt.geomgroup[ROBOT_GROUP] = int(robot)
        opt.geomgroup[GRIPPER_GROUP] = int(gripper)
        return opt

    def render(self, cam="oblique", highlight=None, ghosts=None, robot=True, gripper=False):
        mujoco.mj_camlight(self.m, self.d)
        self.r.update_scene(self.d, cam, self._option(robot, gripper))
        scn = self.r.scene
        if highlight:
            gmap = {}
            for key, rgba in highlight.items():
                for g in self.sc.idx.obj_vis_geoms.get(key, []):
                    gmap[g] = rgba
            for i in range(scn.ngeom):
                gg = scn.geoms[i]
                if gg.objtype == mujoco.mjtObj.mjOBJ_GEOM and gg.objid in gmap:
                    gg.rgba[:] = gmap[gg.objid]
                    gg.matid = -1
        if ghosts:
            for (typ, dataid, size, pos, mat, rgba) in ghosts:
                if scn.ngeom >= scn.maxgeom:
                    break
                gg = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(gg, typ, size, pos, mat.reshape(9), rgba)
                gg.dataid = dataid
                gg.category = mujoco.mjtCatBit.mjCAT_DYNAMIC
                scn.ngeom += 1
        return self.r.render()

    def capture_geoms(self, geoms, rgba):
        """Snapshot visual geoms at the current kinematic state as ghost primitives."""
        m, d = self.m, self.d
        out = []
        for g in geoms:
            typ = int(m.geom_type[g])
            dataid = 2 * int(m.geom_dataid[g]) if typ == mujoco.mjtGeom.mjGEOM_MESH else -1
            out.append((typ, dataid, m.geom_size[g].copy(), d.geom_xpos[g].copy(), d.geom_xmat[g].copy(), rgba))
        return out

    def depth(self, cam="front"):
        self.r.enable_depth_rendering()
        mujoco.mj_camlight(self.m, self.d)
        self.r.update_scene(self.d, cam, self._option(True, False))
        dep = self.r.render().copy()
        self.r.disable_depth_rendering()
        return dep

    def instance_mask(self, cam="front"):
        """Pixel -> object index (1..N in pb.obj_keys order), 0 = background/fixture/robot."""
        self.r.enable_segmentation_rendering()
        mujoco.mj_camlight(self.m, self.d)
        self.r.update_scene(self.d, cam, self._option(True, False))
        seg = self.r.render().copy()
        self.r.disable_segmentation_rendering()
        lut = np.zeros(self.m.ngeom + 1, np.uint8)
        for i, o in enumerate(self.pb.obj_keys):
            for g in self.sc.idx.obj_vis_geoms[o] + self.sc.idx.obj_geoms[o]:
                lut[g] = i + 1
        ids, typ = seg[..., 0], seg[..., 1]
        mask = np.where(typ == mujoco.mjtObj.mjOBJ_GEOM, lut[np.clip(ids, 0, self.m.ngeom)], 0)
        return mask.astype(np.uint8)

    # ------------------------------------------------------------------ action playback
    def apply_sample(self, traj, i, state=None, iv=None):
        """Set the scene to sample i of a trajectory (relocation or target)."""
        sc, pb = self.sc, self.pb
        if traj.joints:   # target: rack + door articulate, rack-resting objects ride along
            sc.set_articulation(**{k: v[i] for k, v in traj.joints.items()})
            for o, (pos, yaw) in traj.carried.items():
                sc.set_object_canonical(o, pos[i], yaw[i])
        else:
            o = iv.obj
            pos, yaw = traj.carried[o]
            sc.set_object_canonical(o, pos[i], yaw[i])
        _ = state
        sc.set_gripper(traj.grip_pos[i], quat=traj.grip_quat[i], opening=traj.opening[i])

    def station_for(self, iv, phase):
        st = (iv.robot or {}).get("stations", {}) if iv is not None else {}
        key = {"approach": "pick", "grasp": "pick", "lift": "lift", "transfer": "transfer", "lower": "above_place",
               "release": "place", "retreat": "place"}.get(phase)
        return st.get(key) or "dishwasher_center"

    def action_frames(self, traj, iv=None, cam="oblique", highlight=None, n_frames=60, stop_index=None,
                      title=None, show_robot=True, station=None):
        idx = np.unique(np.linspace(0, (traj.n - 1) if stop_index is None else stop_index, n_frames).astype(int))
        frames = []
        for i in idx:
            self.apply_sample(traj, i, iv=iv)
            ok = False
            if show_robot:
                st = self.station_for(iv, traj.phase[i]) if iv is not None else (station or "dishwasher_center")
                from .geom import quat_to_mat as q2m

                ok = self.pose_robot(traj.grip_pos[i], q2m(traj.grip_quat[i]), traj.opening[i], st)
            if not ok:
                if self.sc.kin is not None:
                    self.sc.kin.fk(Q_HOME, 0.08)
                self.sc.set_station("dishwasher_right")
            mujoco.mj_kinematics(self.m, self.d)
            img = self.render(cam, highlight=highlight, robot=True if ok else False, gripper=not ok)
            self.last_raw = img
            lines = [title] if title else []
            lines.append(f"tau={traj.tau[i]:.2f}  phase={traj.phase[i]}" + ("" if ok else "  (floating hand: no IK)"))
            frames.append(caption(img, lines))
        return frames

    # ------------------------------------------------------------------ swept-volume debug render
    def sweep_image(self, traj, state, iv=None, blockers=(), tau_star=None, cam="oblique", k=6, title=None,
                    dest_slot=None, extra_lines=()):
        """Ghosted swept volume: moving geometry at every k-th tau (translucent), the tau* pose
        solid orange, blockers red."""
        sc = self.sc
        if iv is not None:
            moving = sc.idx.gripper_vis_geoms + sc.idx.obj_vis_geoms[iv.obj]
        else:
            moving = sc.idx.gripper_vis_geoms + self.fx_vis["door"] + self.fx_vis["rack1"] + \
                [g for o in traj.carried for g in sc.idx.obj_vis_geoms[o]]
        ghosts = []
        i_star = int(np.argmin(np.abs(traj.tau - tau_star))) if tau_star is not None else None
        for i in range(0, traj.n, k):
            self.apply_sample(traj, i, iv=iv)
            mujoco.mj_kinematics(self.m, self.d)
            ghosts += self.capture_geoms(moving, GHOST)
        if i_star is not None:
            self.apply_sample(traj, i_star, iv=iv)
            mujoco.mj_kinematics(self.m, self.d)
            blk = {g for b in blockers for g in sc.idx.obj_vis_geoms.get(b, [])}
            ghosts += self.capture_geoms([g for g in moving if g not in blk], ORANGE)
            ghosts += self.capture_geoms([g for g in moving if g in blk], RED)   # carried blockers at tau*
        self.set_state(state)
        mujoco.mj_kinematics(self.m, self.d)
        if dest_slot is not None:
            ghosts += slot_ghosts(self.pb.topo, outline=dest_slot, only=[dest_slot])
        img = self.render(cam, highlight={b: RED for b in blockers}, ghosts=ghosts, robot=False)
        lines = [title] if title else []
        lines += list(extra_lines)
        if dest_slot is not None:   # second panel: the same state, no ghosts, destination outlined
            mujoco.mj_kinematics(self.m, self.d)
            plain = self.render("workspace", highlight={b: RED for b in blockers},
                                ghosts=slot_ghosts(self.pb.topo, outline=dest_slot, only=[dest_slot]), robot=False)
            plain = caption(plain, [f"magenta = destination slot {dest_slot}", "red = blocking object (counterfactual state)"])
            return hstack(caption(img, lines), plain)
        if tau_star is not None:
            lines.append(f"ghosts: swept volume (every {k} samples)   orange: tau*={tau_star:.2f}   red: blocker(s)")
        return caption(img, lines)


# ----------------------------------------------------------------------------- placement topology
SLOT_RGBA = {"upper_rack": np.array([0.15, 0.45, 0.95, 0.30], np.float32),
             "lower_rack": np.array([0.45, 0.45, 0.45, 0.22], np.float32),
             "temporary_buffer": np.array([0.20, 0.75, 0.35, 0.30], np.float32)}


def slot_ghosts(topo, outline=None, outline_rgba=MAGENTA, skip_wide=True, only=None, skip_unreachable=True):
    """Translucent slot footprints (+ an opaque outline for `outline`) as ghost primitives."""
    out = []
    eye = np.eye(3)
    for sid, sl in topo.items():
        if only is not None and sid not in only:
            continue
        if skip_wide and sid.startswith("BW") and sid != outline:
            continue
        if skip_unreachable and not sl.reachable:
            continue
        cx, cy = sl.center
        hx, hy = sl.half_extent
        z = sl.z + 0.0015
        if sid == outline:
            t = 0.009
            for (px, py, sx, sy) in ((cx, cy - hy, hx, t), (cx, cy + hy, hx, t), (cx - hx, cy, t, hy), (cx + hx, cy, t, hy)):
                out.append((int(mujoco.mjtGeom.mjGEOM_BOX), -1, np.array([sx, sy, 0.004]), np.array([px, py, z + 0.003]),
                            eye.copy(), outline_rgba))
        else:
            out.append((int(mujoco.mjtGeom.mjGEOM_BOX), -1, np.array([hx * 0.94, hy * 0.94, 0.0015]), np.array([cx, cy, z]),
                        eye.copy(), SLOT_RGBA[sl.semantic]))
    return out


def project(model, data, cam: str, pts, w, h):
    c = model.camera(cam).id
    pos, R = data.cam_xpos[c], data.cam_xmat[c].reshape(3, 3)
    f = (h / 2) / np.tan(np.radians(model.cam_fovy[c]) / 2)
    out = []
    for p in np.atleast_2d(pts):
        v = R.T @ (np.asarray(p) - pos)
        out.append((int(w / 2 + f * v[0] / -v[2]), int(h / 2 - f * v[1] / -v[2])))
    return out


def label_slots(img, model, data, cam, topo, w, h, skip_wide=True):
    import cv2

    img = np.ascontiguousarray(img).copy()
    for sid, sl in topo.items():
        if (skip_wide and sid.startswith("BW")) or not sl.reachable:
            continue
        (u, v), = project(model, data, cam, [[sl.center[0], sl.center[1], sl.z + 0.01]], w, h)
        col = (255, 255, 255) if sl.semantic != "lower_rack" else (200, 200, 200)
        cv2.putText(img, sid, (u - 10, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, sid, (u - 10, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------- dependency graph panel
def dependency_panel(res: dict, seq_index=0, size=(560, 480), title="dependency graph (optimal sequence)"):
    """Matplotlib rendering of the interventional dependency graph of one optimal sequence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, title, ha="center", va="top", fontsize=10, weight="bold")
    if not res["sequence_proofs"]:
        ax.text(0.5, 0.5, "no recourse within depth 4", ha="center")
    else:
        proof = res["sequence_proofs"][seq_index]
        seq = proof["sequence"] + ["TARGET"]
        n = len(seq)
        ys = np.linspace(0.86, 0.08, n)
        pos = {s: (0.5, y) for s, y in zip(seq, ys)}
        obj = res["intervention_object"]
        for s in seq:
            x, y = pos[s]
            lab = "CLOSE_DISHWASHER" if s == "TARGET" else s.replace("move_", "move ").replace("_to_", "\n-> ")
            fc = "#ffd6d1" if s == "TARGET" else "#dbeafe"
            ax.text(x, y, lab, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec="#333", lw=0.8))
        colors = {"sweep": "#0b6bcb", "occupancy": "#c2410c", "occupancy+sweep": "#7c3aed", "target_blocker": "#b91c1c",
                  "indirect": "#555"}
        for j, e in enumerate(proof["edges"]):
            a, b = pos[e["from"]], pos[e["to"]]
            typ = e["cause"].get("type", "")
            dx = 0.27 + 0.04 * (j % 3)
            side = 1 if (j % 2 == 0) else -1
            ax.annotate("", xy=(a[0] + side * 0.02, b[1] + 0.03), xytext=(a[0] + side * 0.02, a[1] - 0.03),
                        arrowprops=dict(arrowstyle="-|>", color=colors.get(typ, "#333"), lw=1.4,
                                        connectionstyle=f"arc3,rad={-side * 0.35 * (1 + abs(a[1] - b[1]))}"))
            mid = (a[1] + b[1]) / 2
            txt = f"{e['blocker']}: {typ}"
            if "phase_star" in e["cause"]:
                txt += f"\n{e['cause']['phase_star']} d={e['cause']['min_dist']:+.3f}"
            ax.text(min(max(0.5 + side * dx, 0.16), 0.84), mid, txt, ha="center", va="center", fontsize=8,
                    color=colors.get(typ, "#333"), weight="bold")
        _ = obj
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def hstack(*imgs):
    import cv2

    h = max(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        if i.shape[0] != h:
            i = cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h))
        out.append(i)
    return np.concatenate(out, 1)


_ = (quat_from_yaw, quat_to_mat, PEN_TOL)
