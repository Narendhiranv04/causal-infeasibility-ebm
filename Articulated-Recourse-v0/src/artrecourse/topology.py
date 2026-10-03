"""Semantic placement topology (v0.1).

Every placement is an exact pose on a NAMED slot with a NAMED orientation template; no free
offsets. Slots are derived from measured geometry:

  upper rack (pulled to its loading position, Dishwasher054):
    U1 left-bay front, U2 left-bay back, U3 right-bay front, U4 right-bay back
        (narrow bays between the side wall and the outer tine plate: cups, mugs, bottles, cans)
    U5 tine field (bowls; cookware with orientation templates)
    Rows are the front / back half of the TOP-DOWN-REACHABLE depth: from the rack's inner front
    wall to the countertop edge (anything behind the edge would hit the counter when lifted).
  lower rack:
    L1..L4 are recorded with their measured geometry but marked reachable=False: with the
    door at its real limit the lower rack can be pulled only 0.15 m and only 0.09 m of it is
    reachable from above (see out/fixture_audit.md).
  staging (temporary buffer): ONE visible drying tray on the countertop next to the dishwasher,
    a 2 x 4 grid: B1-B4 front row, B5-B8 back row (narrow items), BW1-BW4 whole columns
    (bowls, utensil holder; overlapping their two narrow cells).

Orientation templates fix yaw AND the anchor rule (derived from the object's footprint):
  upright / handle_back (mug handle +y) / cap (bottle) ...
  cookware: handle_out (handle over the rack front wall), handle_left / handle_right (handle
  across the rack, footprint centred between the side walls).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .placements import measure_supports

NARROW = ("mug", "cup", "bottle")                       # dishwasher items that fit a rack bay
TRAY_NARROW = ("mug", "cup", "bottle", "can", "box")
TRAY_WIDE = ("bowl", "utensil_holder")
TRAY_COLS = 4
TRAY = {"x": (0.34, 1.06), "y": (-0.318, -0.098), "base": 0.012, "rim_h": 0.022, "rim_t": 0.010}


@dataclass
class Slot:
    slot_id: str
    support_id: str
    semantic: str               # upper_rack | lower_rack | temporary_buffer
    center: tuple               # world x, y of the slot frame origin
    z: float                    # support surface height
    half_extent: tuple          # usable footprint (x, y) half sizes, slot frame = world-aligned
    capacity: int = 1
    categories: tuple = ()
    orientations: dict = field(default_factory=dict)   # category -> [template names]
    reachable: bool = True
    overlaps: tuple = ()        # slots sharing physical space (cookware templates)

    def to_dict(self):
        d = asdict(self)
        d["world_pose_xyz"] = [round(self.center[0], 4), round(self.center[1], 4), round(self.z, 4)]
        return d


def build_topology(scene) -> dict[str, Slot]:
    sup = measure_supports(scene)
    lb, rb, tn = sup["rack_left_bay"], sup["rack_right_bay"], sup["rack_tines"]
    counter_edge = scene.world["countertop"]["y"][0]
    y0 = lb.y[0] + 0.004                      # inner face of the rack front wall
    y1 = min(lb.y[1], counter_edge - 0.004)   # top-down reachable limit
    ym = (y0 + y1) / 2
    slots = {}

    def narrow(sid, s, front):
        ya, yb = (y0, ym) if front else (ym, y1)
        slots[sid] = Slot(sid, s.name, "upper_rack", ((s.x[0] + s.x[1]) / 2, (ya + yb) / 2), s.z,
                          ((s.x[1] - s.x[0]) / 2, (yb - ya) / 2), 1, NARROW,
                          {"mug": ["handle_back"], "cup": ["upright"], "bottle": ["upright"], "can": ["upright"]})

    narrow("U1", lb, True)
    narrow("U2", lb, False)
    narrow("U3", rb, True)
    narrow("U4", rb, False)
    slots["U5"] = Slot("U5", tn.name, "upper_rack", ((tn.tine_x[0] + tn.tine_x[-1]) / 2, ym), tn.z,
                       ((tn.tine_x[-1] - tn.tine_x[0]) / 2 + 0.01, (y1 - y0) / 2), 1,
                       ("bowl", "pan", "utensil_holder"),
                       {"bowl": ["upright"], "utensil_holder": ["upright"],
                        "pan": ["handle_out", "handle_left", "handle_right"]})
    # (a skillet across the rack physically covers parts of the bays; that is measured geometrically
    #  -- destination_overlap -- rather than declared as a slot overlap)
    # lower rack (measured; NOT reachable while the upper rack is pulled over it and the door
    # limits the lower rack's own pull to 0.15 m -- static context only, never a destination)
    if "lower_tines" in sup:
        lt, ll, lr = sup["lower_tines"], sup["lower_left_bay"], sup["lower_right_bay"]
        ly0, ly1 = lt.y
        lym = (ly0 + ly1) / 2
        tx = ((lt.tine_x[0] + lt.tine_x[-1]) / 2)
        for sid, (cx, cy, s_, cats) in {"L1": (tx, (ly0 + lym) / 2, lt, ("bowl",)),
                                        "L2": (tx, (lym + ly1) / 2, lt, ("bowl",)),
                                        "L3": ((ll.x[0] + ll.x[1]) / 2, lym, ll, NARROW),
                                        "L4": ((lr.x[0] + lr.x[1]) / 2, lym, lr, NARROW)}.items():
            slots[sid] = Slot(sid, s_.name, "lower_rack", (cx, cy), s_.z,
                              ((s_.x[1] - s_.x[0]) / 2, (ly1 - ly0) / 4), 1, cats,
                              {c: ["upright"] for c in cats}, reachable=False)
    # staging tray: a 2 x 3 drying-mat grid. Narrow cells B1-B3 (front row) and B4-B6 (back
    # row); wide cells BW1-BW3 = one whole column (front + back) for bowls / utensil holder.
    tz = scene.world["countertop"]["z"][1] + TRAY["base"]
    ix0, ix1 = TRAY["x"][0] + TRAY["rim_t"], TRAY["x"][1] - TRAY["rim_t"]
    iy0, iy1 = TRAY["y"][0] + TRAY["rim_t"], TRAY["y"][1] - TRAY["rim_t"]
    nc = TRAY_COLS
    w, h = (ix1 - ix0) / nc, (iy1 - iy0) / 2
    for k in range(nc):
        cx = ix0 + w * (k + 0.5)
        for row, (cy, sid) in enumerate(((iy0 + h / 2, f"B{k + 1}"), (iy0 + 1.5 * h, f"B{k + 1 + nc}"))):
            slots[sid] = Slot(sid, "staging_tray", "temporary_buffer", (cx, cy), tz, (w / 2, h / 2), 1, TRAY_NARROW,
                              {c: ["upright"] for c in TRAY_NARROW}, overlaps=(f"BW{k + 1}",))
        slots[f"BW{k + 1}"] = Slot(f"BW{k + 1}", "staging_tray", "temporary_buffer", (cx, (iy0 + iy1) / 2), tz,
                                   (w / 2, (iy1 - iy0) / 2), 1, TRAY_WIDE, {c: ["upright"] for c in TRAY_WIDE},
                                   overlaps=(f"B{k + 1}", f"B{k + 1 + nc}"))
    return slots


def lower_rack_record(scene) -> list[dict]:
    """Measured lower-rack cells (documentation only; reachable=False)."""
    from .assets import geom_local_points

    m, d = scene.model, scene.data
    pts = np.concatenate([d.geom_xpos[g] + geom_local_points(m, g) @ d.geom_xmat[g].reshape(3, 3).T
                          for g in scene.idx.articulated["rack0"]])
    lo, hi = pts.min(0), pts.max(0)
    xs = np.linspace(lo[0], hi[0], 3)
    ys = np.linspace(lo[1], hi[1], 3)
    out = []
    k = 1
    for j in range(2):
        for i in range(2):
            out.append({"slot_id": f"L{k}", "semantic": "lower_rack", "reachable": False,
                        "center": [round((xs[i] + xs[i + 1]) / 2, 4), round((ys[j] + ys[j + 1]) / 2, 4)],
                        "z": round(float(lo[2]), 4),
                        "reason": "lower rack pull limited to 0.15 m by the 38.6 deg door; 0.09 m reachable from above"})
            k += 1
    return out


def orientation_pose(slot: Slot, template: str, meta: dict, scene_lanes: tuple) -> tuple[np.ndarray, float]:
    """Exact canonical-frame pose (anchor position, yaw) for an object on a slot template.

    scene_lanes = (x_left_wall_inner, x_right_wall_inner, y_front_inner, y_reach_limit).
    """
    lo, hi = np.array(meta["canonical_xy_min"]), np.array(meta["canonical_xy_max"])
    cx, cy = slot.center
    if template in ("upright",):
        yaw, ax, ay = 0.0, cx, cy
    elif template == "handle_back":                     # mug: handle towards +y
        yaw = np.pi / 2
        ax, ay = cx, cy - (hi[0] + lo[0]) / 2           # centre the rotated footprint on the slot
    elif template == "handle_out":                      # cookware: handle out over the front wall
        # body seated so that the handle's rising section (first 6 cm past the body) is inside
        # the rack before it crosses the front wall
        yaw = -np.pi / 2
        ax, ay = cx, scene_lanes[2] + (-lo[0]) + 0.06
    elif template in ("handle_left", "handle_right"):   # cookware across the rack, footprint centred between walls
        xl, xr = scene_lanes[0], scene_lanes[1]
        span_c = (xl + xr) / 2
        if template == "handle_left":
            yaw = np.pi
            ax = span_c + (hi[0] + lo[0]) / 2
        else:
            yaw = 0.0
            ax = span_c - (hi[0] + lo[0]) / 2
        ay = scene_lanes[2] + (-lo[1]) + 0.06          # same seat as handle_out: turning in place
    else:
        raise KeyError(template)
    if template in ("upright",):
        # centre the footprint (not the anchor) for asymmetric objects
        ax -= (hi[0] + lo[0]) / 2
        ay -= (hi[1] + lo[1]) / 2
    return np.array([ax, ay, slot.z]), float(yaw)


def lanes(scene) -> tuple:
    sup = measure_supports(scene)
    counter_edge = scene.world["countertop"]["y"][0]
    return (sup["rack_left_bay"].x[0], sup["rack_right_bay"].x[1], sup["rack_left_bay"].y[0] + 0.004,
            min(sup["rack_left_bay"].y[1], counter_edge - 0.004))
