# Articulated-Recourse-v0

**Problem.** Given a robot manipulation that is currently infeasible, find the minimum
executable sequence of object rearrangements whose swept-volume effects make it feasible.

This directory is isolated from PoC-1/2/3 and holds a MuJoCo dataset generator built on
real assets. Every label comes from MuJoCo geometry that is then re-verified exactly.

## Quick start

```bash
python assets/download_assets.py          # ~130 MB subset of RoboCasa + Menagerie + 1 GSO model
python scripts/preview_assets.py          # validate assets, previews, write assets/asset_manifest.json
python scripts/audit_fixtures.py          # dishwasher audit -> out/fixture_audit.{json,md}
python scripts/build_canonical.py         # 8 canonical scenes -> canonical/V0..V7
python scripts/build_canonical.py --compose   # canonical/comparison.png + summary.json
python scripts/inspect_scene.py V5        # compact oracle summary of one scene
python -m pytest tests -q                 # full test suite (~10 min, builds 3 scenes)
# after approval only:
python scripts/generate_v0.py             # 2,800-scene quick v0 -> out/v0 (+ report.json/md)
```

`MUJOCO_GL=glfw` is used for rendering (EGL is unavailable on this machine). `.asset_cache/`
and `out/` are gitignored.

## World

| element | source |
|---|---|
| Dishwasher | RoboCasa **Dishwasher054** (lightwheel fixtures, CC BY 4.0). Unmodified joints: `door_joint` hinge 0–0.6736 rad, `rack1_joint` / `rack0_joint` slides 0–0.40 m |
| Objects | RoboCasa objaverse objects (CC BY 4.0) at RoboCasa registry scales; bowl scaled 1.4 instead of 2.0 for realistic 16–18 cm bowls. GSO utensil holder (fallback source: RoboCasa has no free-standing one) |
| Robot | mujoco_menagerie `franka_emika_panda` (Apache-2.0) on a mobile base with discrete stations |
| Kitchen | Textured box counters and cabinets, built the way RoboCasa builds counters (RoboCasa marble, wood, tile and plaster textures) |

Visual meshes are used only for rendering: they don't collide and have no mass. All
collision uses the source convex decompositions. Primitive collision boxes and cylinders are
converted to exact convex meshes, so every pair goes through the same convex distance path.

### Fixture selection (v0.1 audit)

`scripts/audit_fixtures.py` measured all 25 RoboCasa dishwashers and wrote the results to
`out/fixture_audit.{json,md}`. Per fixture it records:

- dimensions and joint ranges;
- the largest pull of each rack that keeps 5 mm from the fully opened door;
- how much of each pulled rack is reachable from above, in front of the tub and the
  countertop edge;
- clearances and self-collision;
- for the shortlist, a search for a single base station from which the atomic skills are
  fully IK-admissible.

**Dishwasher054 is selected:**

| Property | Value |
|---|---|
| Door opening | 38.6° |
| Upper rack loading pull | 0.39 m |
| Upper rack reachable depth | 0.236 m (largest of all fixtures) |
| `PUSH_RACK(upper)` | fully admissible |
| `CLOSE_DOOR` | fully admissible (17 stations) |

**No fixture has a usable lower rack.** No RoboCasa door opens past 47°, so at most 0.14 m
of the lower rack is ever reachable, and `PUSH_RACK(lower)` is not admissible anywhere. The
lower rack is therefore measured (L1–L4) but used only as static context.

### Target action: atomic, one per scene

Each scene prescribes one next action, and the oracle query is F(s, a_target).

- **`PUSH_RACK(rack)`:** the real slide joint goes from the loading pull to 0. Internally
  the hand grasps the top of the front wall, then switches to a palm push once the hand would
  reach the tub frame. Objects resting on the rack ride along.
- **`CLOSE_DOOR`:** the real hinge goes from its open value to 0, with the closed hand
  pushing the door's outer face.

Acceptance gate: the target must be fully IK-admissible from one base station, checking
every waypoint, self-collision, and arm against the fixed environment (the manipulated body
excepted). All canonical scenes use `PUSH_RACK(upper)`. `CLOSE_DOOR` is implemented and
admissible, but it has no object-recourse story with this fixture: when both racks are in,
nothing the robot can reach lies in the door's sweep.

### Placement topology

Every pose is a **named slot + named orientation template**, defined in `topology.py`;
canonical scenes contain no free offsets. Slot metadata (`inputs.placement_topology`)
records: id, support, semantic type, exact pose, footprint, capacity 1, allowed categories,
orientation templates, reachability and overlaps.

| Slots | Location | Notes |
|---|---|---|
| U1, U2 / U3, U4 | Front and back rows of the left / right bays | Cups, mugs, bottles. Rows split the top-down reachable depth. |
| U5 | Tine field | Bowls and the utensil holder. Skillet templates: `handle_out`, `handle_left`, `handle_right`. |
| B1–B4 (front), B5–B8 (back) | Narrow cells of one visible 2 × 4 drying tray beside the dishwasher | |
| BW1–BW4 | Whole tray columns | Bowls and the utensil holder; overlap their two narrow cells. |
| L1–L4 | Lower rack | Never a destination. |

### Relocation primitive

`RELOCATE` keeps the same deterministic phases as before. The carry height is now
**derived**: the object's bottom clears, by 3 cm, the highest support-boundary lip that the
straight transfer crosses. Lips are the rack walls and tine plates, the tray rims and the
countertop edge. It no longer uses the fixed 13 cm.

Consequently, transfers from the rack to the tray's back row pass low over the front row. A
measured corridor table (bottle leaving U3) shows the effect:

- an object on B1 blocks B5 and B6;
- on B2, it blocks B3, B7 and B8;
- on B3, it blocks B4 and B8.

Each conflict and dependency edge records its moving part: `gripper`, `carried_object`,
`articulated_fixture` or `rack_borne_object`.

## Oracle

**Sweep.** Each action is sampled densely in τ: every 1 cm in coarse mode, every 2.5 mm in
fine mode. At each τ the moving set (gripper hand and fingers at the commanded opening,
carried objects, articulated bodies) is posed by MuJoCo kinematics. Signed distance to every
static entity's collision geoms is then computed with `mj_geomDistance`.

**Exact verification.** MuJoCo's convex distance proved unreliable near contact. It
returned spurious exact zeros, and for one pair 2.2 cm apart it reported −2.0 cm (native
CCD) and +0.42 cm (legacy libccd). Every pair that MuJoCo puts within 1.2 cm is therefore
re-decided exactly (`exactgeom.py`):

- intersection by LP feasibility;
- separation by an exact minimum-norm solve on the Minkowski difference;
- depth by a separating-axis bound.

**Labels.** A pair is a conflict at ≤ −4 mm, clear at ≥ +4 mm, and ambiguous in between.
A scene is rejected if the oracle *consults* any ambiguous entry.

**Factorization.** State is the pose index of every object. Each sweep is evaluated against
each entity separately, so executability and F(s, target) factorize exactly into
per-(action, entity, pose) tables. The target also has pairwise terms for rack-carried
objects against static ones. `full_check_*` recomputes the same quantities by brute force in
explicit states, and the tests compare the two.

**Search.** Breadth-first search to depth 4, where each object moves at most once,
enumerates **all** tied minimum sequences. It also records the irreducible alternative
plans.

**Labels produced:**

- directed `D_enable` and `D_disable` at s0;
- symmetric compatibility labels: same object, destination overlap, final-placement
  collision, support (slot) incompatibility;
- per-sequence **interventional proofs**. Step j is a prerequisite of step k (or of the
  target) if I_k fails without I_j and succeeds after it. Each proof records the blocker
  identity, a sweep/occupancy/occupancy+sweep cause, the phase, τ\* and the distance.

**Variants** are decided structurally from the oracle output (`variants.py`,
`configs/variants.yaml`), never from how the scene was generated. The v0.1 gates add:

- **V3:** the destination slot is statically free before the prerequisite repair.
- **V4:** the two chains use different mechanisms (occupancy plus swept volume).
- **V5:** the direct blocker is not the first object moved.
- **V7:** at most 6 dependency edges; 1–5 real resource conflicts (destinations shared or
  overlapping between different objects); at least 2 mechanisms; an alternative branch.
- **V4, V5, V7:** at least one carried-object swept-volume edge.

**State-conditioned export (`labels.state_conditioned`).** For every BFS-expanded state up
to the optimal depth, the export records the executable set, F, and each legal one-step
intervention's enable and disable effects. These are training pairs of the form
(s, I_p, I_q).

## Record schema (`oracle.json` / `scene.json`)

`inputs` holds what a model **may** consume:

- objects and assets, initial poses as (x, y, z, qx, qy, qz, qw), articulation state;
- the target action;
- placement candidates;
- candidate interventions with their deterministic trajectories, stored in `inputs.npz`.

`labels` holds **supervision and evaluation only**:

- F, direct blockers and their details;
- per-action executability, minimum clearance per entity, τ\*, conflict magnitudes;
- enable/disable edges, compatibility edges;
- optimal sequences and cost, sequence proofs, irreducible plans, no-recourse;
- the distance profiles in `labels.npz`.

The record lists both key sets explicitly, and a test checks that no label key leaks into
`inputs`.

## Canonical scenes (`canonical/V*/`)

Each scene directory contains `scene.png`, `placement_map.png`, `dependency_graph.png`, `failed_action.mp4`, `repair.mp4`, `oracle.json`,
`sweep_target.png`, one `sweep_step*.png` per repair step, the observations
`obs_front/obs_oblique/depth_front/mask_front.png`, and the `.npz` arrays. The physical
stories are in `src/artrecourse/canonical.py` and in each `oracle.json`.

## Layout

```
assets/        download_assets.py, asset_manifest.json
configs/       assets.yaml, placements.yaml, variants.yaml, generation.yaml
src/artrecourse/
  assets.py      loading, visual/collision separation, canonical frames, validation
  fixture.py     Dishwasher054 loading, primitive->convex conversion, articulation checks
  scene.py       MuJoCo world (kitchen, fixture, mocap objects, floating hand, mobile Panda)
  placements.py  measured support surfaces, exact placement poses, placement validity
  primitives.py  RELOCATE / CLOSE_DISHWASHER trajectories, grasp templates
  robot.py       Panda IK and admissibility checks
  sweep.py       swept-volume distance profiles
  exactgeom.py   exact verification of near contact
  interventions.py  scene spec, candidate catalogue, factorized conflict tables
  oracle.py      BFS, all tied optima, enable/disable/compatibility, sequence proofs
  variants.py    structural variant definitions
  quality.py     coarse/fine agreement, ambiguity gates
  records.py     record schema with inputs/labels separation
  render.py      observations, ghosted sweeps, dependency panels, videos
  topology.py    named semantic slots + orientation templates (v0.1)
  canonical.py   the 8 hand-designed scenes
  generator.py   randomized generation (not yet run)
scripts/       preview_assets, audit_fixtures, build_canonical, inspect_scene, generate_v0
tests/         test_assets, test_geometry, test_oracle
```
