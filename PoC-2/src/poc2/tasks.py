"""PoC-2 Stage 4: three benchmark-inspired task families with distinct prescribed envelopes (plan2.md 20).

  storage_insertion    top grasp; transfer at height over an open storage bin, then lower vertically
                       to a deposit spot (two-segment Polyline). Bin walls / a fixed divider are
                       structural; items in the bin obstruct the descent column.
  storage_extraction   side grasp from the front of a cabinet cubby; lift the target over a front
                       rail, then withdraw it outward (two-segment Polyline starting deep inside).
                       Short items lying in the lane are cleared by the lift, tall ones are not.
  articulated_opening  a cabinet / appliance door rotating about a vertical hinge axis (PoC-1
                       HingeMotion, single-axis); items on the counter in the swept quarter-disc,
                       an optional side wall that stops the swing; the appliance can be slid.

All geometry is primitive axis-aligned boxes (validated PoC-1 regimes), every intervention has unit
cost, and the oracle (M, V, F, G, C*, S*) is the shared poc2.oracle used by every family.
"""

import math
from dataclasses import dataclass

import numpy as np

from poc import cases as cs
from poc.envelope import HingeMotion, LinearMotion
from poc.mj_scene import Box3D, Composite, Entity3D
from poc.types import ActionSpec, Entity, EntityRole, Intervention, InterventionKind
from poc2.oracle import Polyline
from poc2.scenes import ObjectSpec, fits_in_region, gripped, lane_half, shelf_box, to_region
from poc2.types import PlacementRegion, RepairOption

FAMILIES = ("storage_insertion", "storage_extraction", "articulated_opening")
REGION_HALF = {"storage_insertion": (0.04, 0.04, 0.07), "storage_extraction": (0.04, 0.04, 0.095),
               "articulated_opening": (0.04, 0.04, 0.09)}
P_RANGE = (5, 10)
SHIFT_COUNT_P = (0.4, 0.4, 0.2)              # probability of 0, 1 or 2 repositioning alternatives
ITEM_HALF_XY = (0.020, 0.035)                # [m] movable item footprint half-extent
DEPTH = (0.002, 0.020)                       # [m] intrusion of a blocker / structural cause
CLEAR = (0.006, 0.040)                       # [m] clearance of a distractor
HARD = (0.001, 0.005)                        # [m] clearance of a hard negative's closest item
CAUSE_P = 0.3                                # probability a repairable scene has a structural cause
# storage insertion
BIN_C, WALL_T, START_DX, REST_GAP = (0.20, 0.0), 0.01, 0.40, 0.005
SI_SHIFT = 0.04
# storage extraction
X_OUT, SE_SHIFT, SE_SLOTS = -0.30, 0.04, (0.05, 0.12, 0.19)
# articulated opening
DOOR_T, DOOR_Z0, AO_SHIFT = 0.012, 0.005, 0.06


@dataclass(frozen=True)
class TaskSpec:
    family: str
    scene_id: str
    target_half: tuple[float, float, float]
    params: tuple[tuple[str, float], ...]                 # family scalars
    structures: tuple[tuple[str, tuple, tuple], ...]      # optional fixed boxes (id, centre, half)
    regions: tuple[PlacementRegion, ...]
    objects: tuple[ObjectSpec, ...]
    shifts: tuple[tuple[float, float, float], ...]        # mutually exclusive repositionings

    def p(self, key: str) -> float:
        return dict(self.params)[key]


def _fixed(eid, center, half) -> Entity3D:
    return Entity3D(Entity(eid, EntityRole.STRUCTURAL), (Box3D(tuple(center), tuple(half)),))


def shift_option(vec) -> RepairOption:
    name = "shift" + "".join(f"{ax}{v:+.3f}" for ax, v in zip("xyz", vec) if v)
    return RepairOption(Intervention(name, InterventionKind.SHIFT_TARGET, "obj", params=tuple(vec)))


def top_grasp(th) -> Composite:
    """Object held from above: fingers on its +-y sides, palm and a vertical wrist above it."""
    hx, hy, hz = th
    fx = min(hx, 0.02)
    return Composite(parts=(("object", Box3D((0.0, 0.0, 0.0), tuple(th))),
                            ("finger_l", Box3D((0.0, hy + 0.01, 0.0), (fx, 0.01, 0.7 * hz))),
                            ("finger_r", Box3D((0.0, -hy - 0.01, 0.0), (fx, 0.01, 0.7 * hz))),
                            ("palm", Box3D((0.0, 0.0, hz + 0.01), (hx, hy + 0.02, 0.01))),
                            ("wrist", Box3D((0.0, 0.0, hz + 0.10), (0.02, 0.02, 0.08)))))


def _structure(spec: TaskSpec) -> tuple[list, Composite, object, tuple]:
    th, p = spec.target_half, spec.p
    if spec.family == "storage_insertion":
        bx, by, bh, (cx, cy) = p("bin_hx"), p("bin_hy"), p("bin_h"), BIN_C
        walls = [("bin_xn", (cx - bx - WALL_T / 2, cy, bh / 2), (WALL_T / 2, by + WALL_T, bh / 2)),
                 ("bin_xp", (cx + bx + WALL_T / 2, cy, bh / 2), (WALL_T / 2, by + WALL_T, bh / 2)),
                 ("bin_yn", (cx, cy - by - WALL_T / 2, bh / 2), (bx, WALL_T / 2, bh / 2)),
                 ("bin_yp", (cx, cy + by + WALL_T / 2, bh / 2), (bx, WALL_T / 2, bh / 2))]
        sx, sy, zh, zr = p("spot_x"), p("spot_y"), p("z_high"), th[2] + REST_GAP
        motion = Polyline((LinearMotion((cx - bx - START_DX, sy, zh), (sx, sy, zh)),
                           LinearMotion((sx, sy, zh), (sx, sy, zr))))
        return walls, top_grasp(th), motion, ()
    if spec.family == "storage_extraction":
        w, d, h, rail = p("cub_w"), p("cub_d"), p("cub_h"), p("rail_h")
        board = [("shelf", (d / 2, 0.0, -0.01), (d / 2, w, 0.01)), ("ceiling", (d / 2, 0.0, h + 0.01), (d / 2, w, 0.01)),
                 ("wall_l", (d / 2, w + 0.01, h / 2), (d / 2, 0.01, h / 2)),
                 ("wall_r", (d / 2, -w - 0.01, h / 2), (d / 2, 0.01, h / 2)),
                 ("back", (d + 0.01, 0.0, h / 2), (0.01, w + 0.02, h / 2)), ("rail", (0.0075, 0.0, rail / 2), (0.0075, w, rail / 2))]
        xt, yt, lift = p("xt"), p("yt"), p("lift")
        motion = Polyline((LinearMotion((xt, yt, th[2]), (xt, yt, th[2] + lift)),
                           LinearMotion((xt, yt, th[2] + lift), (X_OUT, yt, th[2] + lift))))
        return board, gripped(th, 0.5 * th[2]), motion, (Box3D((xt, yt, th[2]), tuple(th)),)
    bw, bd, bh = p("body_w"), p("body_d"), p("body_h")
    door = Composite(parts=(("door", Box3D((bw / 2, -DOOR_T / 2, (bh - 0.01) / 2), (bw / 2, DOOR_T / 2, (bh - 0.01) / 2))),))
    motion = HingeMotion((-bw / 2, -bd / 2, DOOR_Z0), (0.0, 0.0, -1.0), 0.0, p("theta"))
    return [], door, motion, (Box3D((0.0, 0.0, bh / 2), (bw / 2, bd / 2, bh / 2)),)


def build(spec: TaskSpec):
    """(Scene3D, regions, unit-cost options) rebuilt deterministically from a TaskSpec."""
    fixed, moving, motion, fixture = _structure(spec)
    ents = tuple(shelf_box(o.eid, o.center, o.half) for o in spec.objects)
    structural = tuple(_fixed(*s) for s in fixed) + tuple(_fixed(*s) for s in spec.structures)
    scene = cs.Scene3D(spec.family, ActionSpec(spec.family, "obj"), moving, motion, structural + ents, fixture)
    by_id = {r.region_id: r for r in spec.regions}
    options = [to_region(e, by_id[rid]) for e, o in zip(ents, spec.objects) for rid in o.destinations]
    return scene, spec.regions, tuple(options) + tuple(shift_option(v) for v in spec.shifts)


# ------------------------------------------------------------ randomized samplers

def _u(rng, lo_hi) -> float:
    return float(rng.uniform(*lo_hi))


def _regions(rng, family, spots) -> tuple[PlacementRegion, ...]:
    pick = rng.permutation(len(spots))[: int(rng.integers(2, min(4, len(spots)) + 1))]
    half = REGION_HALF[family]
    return tuple(PlacementRegion(f"r{j + 1}", (spots[s][0], spots[s][1], half[2]), half) for j, s in enumerate(pick))


def _catalogue(rng, objects, regions, occupied, n_shifts) -> tuple[ObjectSpec, ...]:
    """One random legal (fitting, not current) destination per object, then extra ones up to a P target."""
    def legal(eid, half, reg):
        return occupied.get(eid) != reg.region_id and fits_in_region((reg.center[0], reg.center[1], half[2]), half, reg)
    pairs = [(eid, r.region_id) for eid, _, half in objects for r in regions if legal(eid, half, r)]
    first = {}
    for i in rng.permutation(len(pairs)):
        first.setdefault(pairs[i][0], pairs[i])
    chosen = set(first.values())
    extra = [pairs[i] for i in rng.permutation(len(pairs)) if pairs[i] not in chosen]
    chosen |= set(extra[: max(0, int(rng.integers(P_RANGE[0], P_RANGE[1] + 1)) - len(chosen) - n_shifts)])
    return tuple(ObjectSpec(eid, tuple(c), tuple(h), tuple(r.region_id for r in regions if (eid, r.region_id) in chosen))
                 for eid, c, h in objects)


def _shifts(rng, axis: int, step: float) -> tuple:
    vec = lambda s: tuple(s * step if k == axis else 0.0 for k in range(3))  # noqa: E731
    return {0: (), 1: (vec(float(rng.choice([-1.0, 1.0]))),), 2: (vec(-1.0), vec(1.0))}[int(rng.choice(3, p=SHIFT_COUNT_P))]


def _roles(rng, intent, n_lane) -> list[str]:
    """Per lane item: 'block', 'hard' (closest clearance of a negative) or 'clear'."""
    k = int(rng.integers(1, min(2, n_lane) + 1)) if intent == "repairable" else 0
    roles = ["block"] * k + ["clear"] * (n_lane - k)
    if intent == "negative":
        roles[0] = "hard"
    return roles


def _offset(rng, role) -> float:
    return {"block": _u(rng, DEPTH), "hard": -_u(rng, HARD), "clear": -_u(rng, CLEAR)}[role]


def sample_storage_insertion(rng, scene_id: str, intent: str) -> TaskSpec:
    th = (_u(rng, (0.025, 0.035)), _u(rng, (0.020, 0.030)), _u(rng, (0.030, 0.050)))
    bx, by, bh, (cx, cy) = _u(rng, (0.16, 0.20)), _u(rng, (0.17, 0.21)), _u(rng, (0.10, 0.14)), BIN_C
    cxh, cyh = th[0], th[1] + 0.02                      # half footprint of the descent column
    room = 2 * ITEM_HALF_XY[1] + CLEAR[1] + 0.005       # an item of maximal size + clearance fits beside it
    sx = _u(rng, (cx - bx + cxh + room, cx + bx - cxh - room))
    sy = _u(rng, (cy - by + cyh + room, cy + by - cyh - room))
    cause = intent == "repairable" and rng.random() < CAUSE_P
    axis = int(rng.integers(0, 2))                     # one opposite pair of column sides (no corner overlap)
    sides = (2 * axis + rng.permutation(2))[: int(rng.integers(1, 3))]
    roles = _roles(rng, "negative" if cause and intent == "repairable" else intent, len(sides))
    objects = []
    for j, (side, role) in enumerate(zip(sides, roles)):
        h = (_u(rng, ITEM_HALF_XY), _u(rng, ITEM_HALF_XY), _u(rng, (0.02, (bh - 0.01) / 2)))
        o, along = _offset(rng, role), _u(rng, (-0.5, 0.5))
        if side < 2:
            s = 1 if side == 0 else -1
            c = (sx + s * (cxh + h[0] - o), sy + along * cyh, h[2])
        else:
            s = 1 if side == 2 else -1
            c = (sx + along * cxh, sy + s * (cyh + h[1] - o), h[2])
        objects.append([f"b{j + 1}", c, h])
    counter = [(cx - bx - 0.12, sg * (by + 0.08)) for sg in (1, -1)] + [(cx - bx - 0.28, sg * (by + 0.08)) for sg in (1, -1)]
    for j, k in enumerate(rng.permutation(len(counter))[: max(3 - len(objects), int(rng.integers(1, 4)))]):
        h = (_u(rng, ITEM_HALF_XY), _u(rng, ITEM_HALF_XY), _u(rng, (0.02, 0.06)))
        objects.append([f"c{j + 1}", (*counter[k], h[2]), h])  # counter distractors beside the transfer path
    structures = ()
    if cause:  # fixed divider across the bin that intrudes into the column from +y
        y = sy + cyh + 0.005 - _u(rng, DEPTH)
        structures = (("divider", (cx, y, bh / 2), (bx, 0.005, bh / 2)),)
    spots = [(cx + bx + 0.12, cy + 0.08), (cx + bx + 0.12, cy - 0.08), (cx, cy + by + 0.12), (cx, cy - by - 0.12)]
    regions = _regions(rng, "storage_insertion", spots)
    shifts = _shifts(rng, 1, SI_SHIFT)
    params = (("bin_hx", bx), ("bin_hy", by), ("bin_h", bh), ("spot_x", sx), ("spot_y", sy),
              ("z_high", bh + th[2] + 0.02 + _u(rng, (0.0, 0.02))))
    return TaskSpec("storage_insertion", scene_id, th, params, structures, regions,
                    _catalogue(rng, objects, regions, {}, len(shifts)), shifts)


def sample_storage_extraction(rng, scene_id: str, intent: str) -> TaskSpec:
    th = (_u(rng, (0.030, 0.040)), _u(rng, (0.025, 0.032)), _u(rng, (0.045, 0.065)))
    w, rail, xt, yt = _u(rng, (0.20, 0.26)), _u(rng, (0.025, 0.035)), _u(rng, (0.27, 0.31)), _u(rng, (-0.05, 0.05))
    lift = rail + _u(rng, (0.004, 0.012))
    d = xt + th[0] + _u(rng, (0.03, 0.06))
    top = max(2 * th[2] + lift, 1.5 * th[2] + lift + 0.02)  # highest point of the lifted composite
    h = max(top + _u(rng, (0.02, 0.05)), 0.20)
    L = lane_half(th)
    cause = intent == "repairable" and rng.random() < CAUSE_P
    slots = rng.permutation(2 * len(SE_SLOTS))[: int(rng.integers(2, 5))]
    roles = _roles(rng, "negative" if cause and intent == "repairable" else intent, len(slots))
    objects = []
    for j, (slot, role) in enumerate(zip(slots, roles)):
        x, s = SE_SLOTS[slot // 2], (1 if slot % 2 == 0 else -1)
        if role == "clear" and rng.random() < 0.5:  # short item lying in the lane, below the lifted object
            hz = _u(rng, (0.004, (lift - 0.004) / 2))
            objects.append([f"b{j + 1}", (x, yt + _u(rng, (-0.3, 0.3)) * L, hz),
                            (_u(rng, ITEM_HALF_XY), _u(rng, ITEM_HALF_XY), hz)])
            continue
        hz = _u(rng, (top / 2 + 0.005, REGION_HALF["storage_extraction"][2]))  # taller than the lifted composite
        hy = _u(rng, ITEM_HALF_XY)
        objects.append([f"b{j + 1}", (x, yt + s * (L + hy - _offset(rng, role)), hz), (_u(rng, ITEM_HALF_XY), hy, hz)])
    structures = ()
    if cause:  # fixed post at the opening intruding into the lane from +y
        structures = (("post", (0.025, yt + L + 0.01 - _u(rng, DEPTH), h / 2), (0.01, 0.01, h / 2)),)
    spots = [(x, sgn * (w - 0.05)) for x in (0.08, 0.20, 0.30) if x + 0.04 <= d for sgn in (1, -1)]
    regions = _regions(rng, "storage_extraction", spots)
    shifts = _shifts(rng, 1, SE_SHIFT)
    params = (("cub_w", w), ("cub_d", d), ("cub_h", h), ("rail_h", rail), ("xt", xt), ("yt", yt), ("lift", lift))
    return TaskSpec("storage_extraction", scene_id, th, params, structures, regions,
                    _catalogue(rng, objects, regions, {}, len(shifts)), shifts)


def sample_articulated_opening(rng, scene_id: str, intent: str) -> TaskSpec:
    bw, bd, bh, theta = _u(rng, (0.20, 0.28)), _u(rng, (0.18, 0.24)), _u(rng, (0.16, 0.24)), math.radians(_u(rng, (100, 110)))
    px, py, reach = -bw / 2, -bd / 2, math.hypot(bw, DOOR_T)
    overshoot = DOOR_T * math.sin(theta) - bw * math.cos(theta)  # exact: outer door corner beyond the hinge plane
    cause = intent == "repairable" and rng.random() < CAUSE_P
    angles = rng.permutation(np.array([15, 45, 75]))[: int(rng.integers(1 if cause else 2, 4))]
    roles = _roles(rng, "negative" if cause and intent == "repairable" else intent, len(angles))
    objects = []
    for j, (phi, role) in enumerate(zip(angles, roles)):
        a, b, hz = _u(rng, ITEM_HALF_XY), _u(rng, ITEM_HALF_XY), _u(rng, (0.03, 0.08))
        R, phi = reach - _offset(rng, role), math.radians(float(phi))
        objects.append([f"b{j + 1}", (px + R * math.cos(phi) + a, py - R * math.sin(phi) - b, hz), (a, b, hz)])
    counter = [(px + reach + 0.15, py - 0.20), (px - 0.20, py - 0.30), (px + reach + 0.15, py - 0.40)]
    for j, k in enumerate(rng.permutation(len(counter))[: max(3 - len(objects), int(rng.integers(0, 3)))]):
        h = (_u(rng, ITEM_HALF_XY), _u(rng, ITEM_HALF_XY), _u(rng, (0.03, 0.08)))
        objects.append([f"c{j + 1}", (*counter[k], h[2]), h])
    gap = _u(rng, (0.01, overshoot - 0.005)) if cause else overshoot + _u(rng, CLEAR)
    structures = (("wall", (px - gap - 0.015, py + (bd - 0.45) / 2, 0.2), (0.015, (bd + 0.45) / 2, 0.2)),)
    spots = [(bw / 2 + 0.12, 0.0), (bw / 2 + 0.12, -0.10), (bw / 2 + 0.12, 0.10), (px + 0.05, py - reach - 0.10)]
    regions = _regions(rng, "articulated_opening", spots)
    shifts = _shifts(rng, 0, AO_SHIFT)
    params = (("body_w", bw), ("body_d", bd), ("body_h", bh), ("theta", theta))
    return TaskSpec("articulated_opening", scene_id, (0.0, 0.0, 0.0), params, structures, regions,
                    _catalogue(rng, objects, regions, {}, len(shifts)), shifts)


SAMPLERS = {"storage_insertion": sample_storage_insertion, "storage_extraction": sample_storage_extraction,
            "articulated_opening": sample_articulated_opening}


# ------------------------------------------------ deterministic mechanism regression cases

def _region(rid, family, x, y) -> PlacementRegion:
    half = REGION_HALF[family]
    return PlacementRegion(rid, (x, y, half[2]), half)


def regression_cases() -> dict[str, tuple[TaskSpec, frozenset, frozenset]]:
    """One hand-built case per family: (spec, expected C*, expected S*) derived from the geometry by hand."""
    si = TaskSpec("storage_insertion", "si_reg", (0.03, 0.025, 0.04),
                  (("bin_hx", 0.18), ("bin_hy", 0.19), ("bin_h", 0.12), ("spot_x", 0.20), ("spot_y", 0.0), ("z_high", 0.19)),
                  (("divider", (0.20, 0.04, 0.06), (0.18, 0.005, 0.06)),),  # intrudes 10 mm into the column (y <= 0.045)
                  (_region("r1", "storage_insertion", 0.50, 0.08), _region("r2", "storage_insertion", 0.20, -0.31)),
                  (ObjectSpec("b1", (0.13, 0.0, 0.04), (0.03, 0.03, 0.04), ("r1", "r2")),   # 10 mm clear of the column
                   ObjectSpec("c1", (-0.10, 0.27, 0.04), (0.03, 0.03, 0.04), ("r1",))),
                  ((0.0, -SI_SHIFT, 0.0), (0.0, SI_SHIFT, 0.0)))
    se = TaskSpec("storage_extraction", "se_reg", (0.035, 0.028, 0.055),
                  (("cub_w", 0.23), ("cub_d", 0.37), ("cub_h", 0.20), ("rail_h", 0.03), ("xt", 0.29), ("yt", 0.0),
                   ("lift", 0.038)), (),
                  (_region("r1", "storage_extraction", 0.08, -0.18), _region("r2", "storage_extraction", 0.20, 0.18),
                   _region("r3", "storage_extraction", 0.30, -0.18)),
                  (ObjectSpec("b1", (0.12, 0.068, 0.09), (0.03, 0.03, 0.09), ("r1", "r2")),   # tall, 10 mm into the lane
                   ObjectSpec("p1", (0.05, 0.0, 0.01), (0.03, 0.03, 0.01), ("r2",)),          # flat, below the lifted object
                   ObjectSpec("d1", (0.19, -0.098, 0.09), (0.03, 0.03, 0.09), ("r3",))),      # tall, 20 mm clear
                  ((0.0, -SE_SHIFT, 0.0),))
    reach, px, py = math.hypot(0.24, DOOR_T), -0.12, -0.10
    corner = (px + (reach + 0.02) * math.cos(math.pi / 4), py - (reach + 0.02) * math.sin(math.pi / 4))
    ao = TaskSpec("articulated_opening", "ao_reg", (0.0, 0.0, 0.0),
                  (("body_w", 0.24), ("body_d", 0.20), ("body_h", 0.20), ("theta", math.radians(105.0))),
                  (("wall", (px - 0.04 - 0.015, py + (0.20 - 0.45) / 2, 0.2), (0.015, (0.20 + 0.45) / 2, 0.2)),),
                  (_region("r1", "articulated_opening", 0.24, 0.0), _region("r2", "articulated_opening", 0.24, -0.10)),
                  (ObjectSpec("b1", (corner[0] + 0.03, corner[1] - 0.03, 0.06), (0.03, 0.03, 0.06), ("r1", "r2")),
                   ObjectSpec("c1", (px + reach + 0.15, py - 0.20, 0.06), (0.03, 0.03, 0.06), ("r1",))),
                  ((AO_SHIFT, 0.0, 0.0), (-AO_SHIFT, 0.0, 0.0)))
    return {
        "storage_insertion": (si, frozenset({frozenset({"divider"})}), frozenset({frozenset({"shifty-0.040"})})),
        "storage_extraction": (se, frozenset({frozenset({"b1"})}), frozenset({frozenset({"b1->r1"}), frozenset({"b1->r2"})})),
        "articulated_opening": (ao, frozenset({frozenset({"wall"})}),
                                frozenset({frozenset({"shiftx+0.060", "b1->r1"}), frozenset({"shiftx+0.060", "b1->r2"})})),
    }
