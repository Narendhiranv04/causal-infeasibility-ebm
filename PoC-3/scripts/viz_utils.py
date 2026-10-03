"""Read-only scene / infeasibility visualization helpers (explanatory tooling; not part of the model or pipeline).

All geometry comes from the frozen pipeline: scene builders (poc2.scenes / poc2.tasks via poc3.dataset.build),
interventions via poc2.oracle.apply_option, swept poses via poc.envelope.poses (stacked per Polyline segment exactly as
poc2.oracle.envelope_sweep stacks them, so pose k is oracle distance sample k), and blockers / F / G / c / d from
poc2.oracle.evaluate. No collision checking is re-implemented here.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from poc.envelope import ENVELOPE_STEP, poses, world_corners
from poc.mj_scene import Composite, GeomWorld
from poc.types import EntityRole, InterventionKind
from poc2 import oracle as orc

C = {"structural": "#9e9e9e", "movable": "#5b8fd6", "blocker": "#d62728", "target": "#2ca02c", "gripper": "#1b5e20",
     "fixture": "#a9cfa0", "trajectory": "#111111", "envelope": "#2ca02c", "region": "#555555", "relocation": "#ff7f0e",
     "shift": "#7b3fbf", "ok": "#1a7f37"}
FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
ACTION_TEXT = {"make_space": "carry the gripped object straight through the shelf lane into the cupboard",
               "insertion": "carry the gripped object straight through the shelf lane into the cupboard",
               "storage_insertion": "carry the top-grasped object over the bin, then lower it into its storage spot",
               "storage_extraction": "lift the object off the shelf, then pull it straight out of the cupboard",
               "articulated_opening": "swing the hinged door open about its vertical hinge"}


# ------------------------------------------------------------ frozen-pipeline geometry

def apply_state(scene, options, x: int):
    for p, o in enumerate(options):
        if x >> p & 1:
            scene = orc.apply_option(scene, o.intervention)
    return scene


def bits(x: int, P: int) -> list[int]:
    return [p for p in range(P) if x >> p & 1]


def option_text(o) -> str:
    iv = o.intervention
    if iv.kind is InterventionKind.SHIFT_TARGET:
        return "SHIFT target/action by (" + ", ".join(f"{v * 1000:+.0f}" for v in iv.params) + ") mm"
    return f"move {iv.entity_id} -> region {o.region_id}"


def sample_poses(motion, moving, step: float = ENVELOPE_STEP) -> list:
    if isinstance(motion, orc.Polyline):
        return [pq for seg in motion.segments for pq in poses(seg, moving, step)[1]]
    return list(poses(motion, moving, step)[1])


def analyse(scene) -> dict:
    """Oracle assessment + the exact swept poses behind each oracle distance sample."""
    sw = orc.envelope_sweep(GeomWorld(scene.entities, scene.moving), scene.motion, scene.moving)
    pq = sample_poses(scene.motion, scene.moving)
    if len(pq) != len(sw.taus):
        raise AssertionError("rendered poses are not aligned with the oracle sweep samples")
    a = orc.evaluate(scene)
    tau_idx = {i: int(np.argmin(sw.distances[:, a.ids.index(i)])) for i in a.blockers}
    return {"a": a, "sweep": sw, "poses": pq, "tau_idx": tau_idx}


def parts_at(moving, pos, quat) -> list[tuple[str, np.ndarray]]:
    return [(n, world_corners(Composite(((n, b),)), pos, quat)) for n, b in moving.parts]


def entity_boxes(scene) -> list[tuple[str, str, np.ndarray]]:
    out = [(e.eid, "structural" if e.entity.role is EntityRole.STRUCTURAL else "movable", b.corners())
           for e in scene.entities for b in e.boxes]
    return out + [("fixture", "fixture", b.corners()) for b in scene.fixture]


def centre(scene, eid: str) -> np.ndarray:
    c = np.concatenate([b.corners() for e in scene.entities if e.eid == eid for b in e.boxes])
    return (c.min(axis=0) + c.max(axis=0)) / 2


def contact_point(scene, eid: str, pos, quat) -> np.ndarray:
    """Centre of the overlap of the blocker's AABB with the moving parts' AABBs at the given pose."""
    eb = np.concatenate([b.corners() for e in scene.entities if e.eid == eid for b in e.boxes])
    lo, hi = eb.min(axis=0), eb.max(axis=0)
    best = None
    for _, pc in parts_at(scene.moving, pos, quat):
        l2, h2 = np.maximum(lo, pc.min(axis=0)), np.minimum(hi, pc.max(axis=0))
        gap = float(np.max(l2 - h2))
        if best is None or gap < best[0]:
            best = (gap, (l2 + h2) / 2)
    return best[1]


def verify(info: dict, expect_F: int | None, label: str) -> None:
    a = info["a"]
    if expect_F is not None and a.F != expect_F:
        raise AssertionError(f"{label}: panel says F={expect_F} but the oracle gives F={a.F}")
    if any(a.c[a.ids.index(i)] <= 0 for i in a.blockers):
        raise AssertionError(f"{label}: highlighted entity without positive conflict")


# ------------------------------------------------------------ drawing primitives

def hull(pts2: np.ndarray) -> np.ndarray:
    try:
        return pts2[ConvexHull(pts2).vertices]
    except Exception:  # degenerate (flat) projection
        return pts2


def poly2d(ax, corners, plane, **kw) -> None:
    i, j = AXES[plane]
    ax.add_patch(Polygon(hull(corners[:, [i, j]]), closed=True, **kw))


def box3d(ax, corners, color, alpha=0.6, edge="k", lw=0.4) -> None:
    ax.add_collection3d(Poly3DCollection([corners[f] for f in FACES], facecolor=color, alpha=alpha, edgecolor=edge,
                                         linewidths=lw))


def draw_scene(ax, scene, plane: str, blockers=(), regions=(), dest_region=None, faded=(), label=True) -> None:
    """Static entities (2D projection or 3D when plane == '3d'), blockers red, regions dashed."""
    for eid, kind, cs in entity_boxes(scene):
        col = C["blocker"] if eid in blockers else C[kind]
        alpha = 0.18 if kind == "structural" and eid not in blockers else (0.35 if eid in faded else 0.85)
        if plane == "3d":
            if kind == "structural" and eid not in blockers and eid in ("top", "ceiling"):
                continue  # keep the view into the cupboard open
            box3d(ax, cs, col, alpha=min(alpha, 0.55), edge="#444" if eid in blockers else "#777")
        else:
            poly2d(ax, cs, plane, facecolor=col, alpha=alpha, edgecolor="#333" if eid in blockers else "#666",
                   lw=1.6 if eid in blockers else 0.6)
            if (label and kind != "structural") or eid in blockers:
                i, j = AXES[plane]
                c = (cs.min(axis=0) + cs.max(axis=0)) / 2
                ax.text(c[i], cs.max(axis=0)[j] + 0.006, "target body" if eid == "fixture" else eid, ha="center",
                        va="bottom", fontsize=7.5, weight="bold", color=C["blocker"] if eid in blockers else "#102040",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=12)
    if plane == "xy":
        for r in regions:
            sel = dest_region is not None and r.region_id == dest_region
            ax.add_patch(Rectangle((r.center[0] - r.half[0], r.center[1] - r.half[1]), 2 * r.half[0], 2 * r.half[1],
                                   fill=False, ls="--", lw=1.6 if sel else 0.9, ec=C["relocation"] if sel else C["region"]))
            ax.text(r.center[0], r.center[1] + r.half[1] + 0.008, r.region_id, ha="center", fontsize=6.5,
                    color=C["relocation"] if sel else C["region"])


def draw_motion(ax, scene, info, plane: str, color=C["target"], path_color=C["trajectory"], n_copies=3,
                ls="-", faint=False, show_hit=True) -> None:
    """Swept envelope (light), start / intermediate / goal copies, centroid path, and the pose of maximum conflict."""
    pq = info["poses"]
    cent = np.array([np.mean(np.concatenate([c for _, c in parts_at(scene.moving, p, q)]), axis=0) for p, q in pq])
    if plane == "3d":
        for k in np.linspace(0, len(pq) - 1, n_copies + 2).astype(int):
            for _, pc in parts_at(scene.moving, *pq[k]):
                box3d(ax, pc, color, alpha=0.07 if faint else (0.35 if k in (0, len(pq) - 1) else 0.12), edge=None, lw=0)
        ax.plot(cent[:, 0], cent[:, 1], cent[:, 2], color=path_color, lw=1.2 if faint else 2.4, ls=ls)
    else:
        i, j = AXES[plane]
        for k in np.linspace(0, len(pq) - 1, 40).astype(int):
            for _, pc in parts_at(scene.moving, *pq[k]):
                poly2d(ax, pc, plane, facecolor=C["envelope"], alpha=0.025 if faint else 0.05, lw=0)
        if not faint:
            for k in np.linspace(0, len(pq) - 1, n_copies + 2).astype(int)[1:-1]:
                for _, pc in parts_at(scene.moving, *pq[k]):
                    poly2d(ax, pc, plane, facecolor="none", edgecolor=C["gripper"], alpha=0.35, lw=0.6)
            for k, lw in ((0, 1.4), (len(pq) - 1, 1.4)):
                for n, pc in parts_at(scene.moving, *pq[k]):
                    poly2d(ax, pc, plane, facecolor=C["target"] if n in ("object", "door") else C["gripper"],
                           alpha=0.55, edgecolor="k", lw=0.5)
        ax.plot(cent[:, i], cent[:, j], color=path_color, lw=1.0 if faint else 2.6, ls=ls, alpha=0.6 if faint else 1)
        ax.annotate("", xy=(cent[-1, i], cent[-1, j]), xytext=(cent[-6, i], cent[-6, j]),
                    arrowprops=dict(arrowstyle="-|>", color=path_color, lw=2, alpha=0.6 if faint else 1))
    if show_hit and not faint:
        for eid, k in info["tau_idx"].items():
            p, q = pq[k]
            for n, pc in parts_at(scene.moving, p, q):
                if plane == "3d":
                    box3d(ax, pc, C["target"], alpha=0.55, edge=C["blocker"], lw=1.0)
                else:
                    poly2d(ax, pc, plane, facecolor=C["target"], alpha=0.5, edgecolor=C["blocker"], lw=1.6)
            cp = contact_point(scene, eid, p, q)
            if plane == "3d":
                ax.scatter(*cp, color=C["blocker"], marker="X", s=120, depthshade=False, zorder=10)
            else:
                i, j = AXES[plane]
                ax.plot(cp[i], cp[j], marker="X", ms=13, color=C["blocker"], mec="white", mew=1.2, zorder=10)


def relocation_arrows(ax, before, after, plane: str) -> None:
    i, j = AXES.get(plane, (0, 1))
    for e0, e1 in zip(before.entities, after.entities):
        a, b = centre(before, e0.eid), centre(after, e1.eid)
        if np.linalg.norm(a - b) > 1e-9:
            if plane == "3d":
                ax.plot(*np.stack([a, b]).T, color=C["relocation"], lw=2.5)
                ax.scatter(*b, color=C["relocation"], s=40)
            else:
                ax.annotate("", xy=(b[i], b[j]), xytext=(a[i], a[j]),
                            arrowprops=dict(arrowstyle="-|>", color=C["relocation"], lw=2.4, ls="--"))
                poly2d(ax, np.concatenate([bb.corners() for bb in e0.boxes]), plane, facecolor="none",
                       edgecolor=C["relocation"], ls=":", lw=1.2)


def frame_axes(ax, scenes_infos, plane: str, pad: float = 0.03) -> None:
    pts = []
    for scene, info in scenes_infos:
        pts += [cs for _, _, cs in entity_boxes(scene)]
        pts += [c for p, q in info["poses"][::max(1, len(info["poses"]) // 30)] for _, c in parts_at(scene.moving, p, q)]
    allp = np.concatenate(pts)
    lo, hi = allp.min(axis=0) - pad, allp.max(axis=0) + pad
    if plane == "3d":
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_zlim(min(lo[2], 0), hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=28, azim=-60)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    else:
        i, j = AXES[plane]
        ax.set_xlim(lo[i], hi[i]), ax.set_ylim(lo[j], hi[j])
        ax.set_aspect("equal")
        ax.set_xlabel(f"{'xyz'[i]} [m]", fontsize=7), ax.set_ylabel(f"{'xyz'[j]} [m]", fontsize=7)
        ax.tick_params(labelsize=6)


def legend(fig, loc="lower center", y=0.0) -> None:
    items = [Patch(color=C["structural"], alpha=0.5, label="fixed structure"), Patch(color=C["movable"], label="movable object"),
             Patch(color=C["blocker"], label="current blocker (c > 0)"), Patch(color=C["target"], label="carried object / moving part"),
             Patch(color=C["envelope"], alpha=0.25, label="swept envelope"),
             Line2D([], [], color=C["trajectory"], lw=2.5, label="action trajectory"),
             Line2D([], [], color=C["shift"], lw=2.5, label="SHIFT_TARGET (shifted path)"),
             Line2D([], [], color=C["relocation"], lw=2.5, ls="--", label="relocation (old -> new)"),
             Line2D([], [], color=C["region"], lw=1, ls="--", label="placement region"),
             Line2D([], [], color=C["blocker"], marker="X", lw=0, ms=10, label="collision location")]
    fig.legend(handles=items, loc=loc, ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, y))


def status(ax, F, plane: str = "xy") -> None:
    txt, col = {1: ("INFEASIBLE", C["blocker"]), 0: ("FEASIBLE", C["ok"])}.get(F, (str(F), C["relocation"]))
    kw = dict(transform=ax.transAxes, fontsize=11, weight="bold", color="white",
              bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))
    if plane == "3d":
        ax.text2D(0.02, 0.92, txt, **kw)
    else:
        ax.text(0.02, 0.93, txt, **kw)
