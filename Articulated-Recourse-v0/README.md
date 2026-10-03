# Articulated-Recourse-v0

**Problem.** Given a robot manipulation that is currently infeasible, find the minimum
executable sequence of object rearrangements whose swept-volume effects make it feasible.

This directory is isolated from PoC-1/2/3 and holds a MuJoCo dataset generator built on
real assets. Every label comes from MuJoCo geometry that is then re-verified exactly.

## Quick start

```bash
python assets/download_assets.py          # ~130 MB subset of RoboCasa + Menagerie + 1 GSO model
python scripts/preview_assets.py          # validate assets, previews, write assets/asset_manifest.json
python scripts/build_canonical.py         # 8 canonical scenes -> canonical/V0..V7
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

### Why the upper rack

The dishwasher door's real maximum opening is 38.6°. At that angle a fully pulled lower rack
would pass 17.5 cm into the door, which is a physically impossible configuration. A fully
pulled upper rack clears the door by 3.9 mm. Scenes therefore use the door at its maximum,
the **upper rack in its pulled loading position** and the lower rack in.

### Target action

`CLOSE_DISHWASHER` has two phases, both using the real joints:

1. τ ∈ [0, 0.5]: `PUSH_RACK` moves `rack1_joint` from 0.40 to 0. Everything resting on the
   rack rides along. The hand grasps the front-wall edge, then switches to a palm push near
   the end, where the hand would otherwise hit the tub ceiling.
2. τ ∈ [0.5, 1]: `CLOSE_DOOR` moves `door_joint` from 0.6736 to 0. The closed hand pushes the
   door's outer face.

Physical failure modes this produces:

- **Door hit:** a skillet handle sticking out of the rack front is hit by the closing door.
- **Ceiling jam:** an item taller than the 0.187 m tub clearance (bottles, utensil holder)
  jams against the tub ceiling.
- **Side or back hit:** items overhanging the rack sides or back hit the tub walls.

### Relocation primitive

`RELOCATE` runs these phases in order:

1. Top-down approach (10 cm).
2. Grasp.
3. Vertical lift until the object's bottom is 13 cm above the higher of the two supports
   (this clears the 11.7 cm rack walls).
4. Straight transfer with interpolated yaw.
5. Lower.
6. Release.
7. Retreat (10 cm).

Each category has a deterministic, ordered list of grasp templates: rim azimuths, positions
along a handle, or finger yaw. The first candidate that is clear of the **fixed**
environment and robot-admissible is used. The trajectory is therefore a pure function of
source pose, destination pose, category geometry and the fixture.

### Robot admissibility

The robot is a filter only and never produces a recourse label. Every waypoint needs a
collision-free IK solution, checking self-collision and arm against the fixed environment.

- The (pre-grasp, grasp) waypoints share one base station, and so do (place, retreat).
- The base may move while the object is held high.
- The target articulation is the same in every scene, so its robot coverage is reported but
  is not a per-scene filter. See Problems below.

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
`configs/variants.yaml`), never from how the scene was generated.

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

Each scene directory contains `scene.png`, `dependency_graph.png`, `failed_action.mp4`, `repair.mp4`, `oracle.json`,
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
  canonical.py   the 8 hand-designed scenes
  generator.py   randomized generation (not yet run)
scripts/       preview_assets, build_canonical, inspect_scene, generate_v0
tests/         test_assets, test_geometry, test_oracle
```
