"""PoC-3 Stage 1 feature-contract tests: geometry conventions, sensitivity, and no oracle leakage."""

import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from poc import cases as cs
from poc import envelope as pe
from poc import mj_scene
from poc import oracle as po
from poc.envelope import HingeMotion, LinearMotion
from poc.mj_scene import Box3D, Composite, Entity3D
from poc.types import Entity, EntityRole, InterventionKind
from poc2 import dataset as ds
from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2.types import RepairOption
from poc3 import TRAINING_SEEDS
from poc3 import features as ft

torch.manual_seed(TRAINING_SEEDS[0])  # explicit seed (no model is initialised here, kept for uniformity)
STAGE1 = {c.name: (c.scene, c.regions, c.options) for c in scenes.stage1_cases()}
TASKS = {k: tasks.build(spec) for k, (spec, _, _) in tasks.regression_cases().items()}
ALL = {**STAGE1, **TASKS}
L0 = ft.L0


def cols(names, *wanted):
    return [names.index(w) for w in wanted]


# ------------------------------------------------------------ contract shape

@pytest.mark.parametrize("name", sorted(ALL))
def test_contract_shapes(name):
    scene, regions, options = ALL[name]
    f = ft.extract(scene, regions, options)
    assert f.entities.shape == (len(scene.entities) + bool(scene.fixture), 9)
    assert f.action.shape == (8, 9) and f.moving.shape == (6,) and f.candidates.shape == (len(options), 14)
    assert np.all(np.isfinite(f.entities)) and np.all(np.isfinite(f.candidates))
    assert np.all(f.entities[:, 6:].sum(axis=1) == 1)


def test_union_aabb_for_multi_box_and_rotated_entities():
    two = (Box3D((0.0, 0.0, 0.1), (0.05, 0.02, 0.1)), Box3D((0.2, 0.0, 0.05), (0.01, 0.06, 0.05)))
    c, h = ft.boxes_aabb(two)
    assert np.allclose(c, (0.08, 0.0, 0.1)) and np.allclose(h, (0.13, 0.06, 0.1))
    quarter = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))  # 90 deg about z swaps hx, hy
    c, h = ft.boxes_aabb((Box3D((0.1, 0.2, 0.3), (0.05, 0.02, 0.01), quarter),))
    assert np.allclose(c, (0.1, 0.2, 0.3)) and np.allclose(h, (0.02, 0.05, 0.01))
    scene, regions, options = STAGE1["M1_independent"]
    ent = Entity3D(Entity("shelfpair", EntityRole.STRUCTURAL), two)
    f = ft.extract(replace(scene, entities=scene.entities + (ent,)), regions, options)
    assert np.allclose(f.entities[len(scene.entities)], [0.16, 0.0, 0.2, 0.26, 0.12, 0.2, 0, 1, 0])


def test_rotation_6d_is_the_first_two_columns_column_major():
    assert np.allclose(ft.rot6d((1.0, 0.0, 0.0, 0.0)), [1, 0, 0, 0, 1, 0])
    th = 0.7  # rotation about -z by th
    R = np.array([[math.cos(th), math.sin(th), 0], [-math.sin(th), math.cos(th), 0], [0, 0, 1]])
    _, quat = ft.frame_at(HingeMotion((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0.0, th), 1.0)
    assert np.allclose(ft.rot6d(quat), np.concatenate([R[:, 0], R[:, 1]]))


def test_polyline_samples_use_global_progress_per_segment():
    a, b, c = (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.2, 0.0, 0.4)
    poly = orc.Polyline((LinearMotion(a, b), LinearMotion(b, c)))
    assert np.allclose(ft.frame_at(poly, 0.0)[0], a) and np.allclose(ft.frame_at(poly, 0.5)[0], b)
    assert np.allclose(ft.frame_at(poly, 1.0)[0], c)
    assert np.allclose(ft.frame_at(poly, 3 / 7)[0], (0.2 * 6 / 7, 0, 0))   # segment 0, local t = 6/7
    assert np.allclose(ft.frame_at(poly, 4 / 7)[0], (0.2, 0, 0.4 / 7))     # segment 1, local t = 1/7
    A = ft.action_tensor(poly)
    assert np.allclose(A[:, :3] * L0, [ft.frame_at(poly, k / 7)[0] for k in range(8)])


def test_hinge_action_keeps_the_pivot_and_rotates():
    scene = TASKS["articulated_opening"][0]
    A = ft.action_tensor(scene.motion)
    assert np.allclose(A[:, :3], np.asarray(scene.motion.pivot) / L0)
    assert np.allclose(A[0, 3:], [1, 0, 0, 0, 1, 0]) and not np.allclose(A[-1, 3:], A[0, 3:])


def test_moving_composite_has_local_center_offset_and_half_extents():
    comp = Composite((("a", Box3D((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))), ("b", Box3D((0.3, 0.0, 0.0), (0.1, 0.1, 0.1)))))
    assert np.allclose(ft.moving_tensor(comp) * L0, [0.15, 0, 0, 0.25, 0.1, 0.1])
    door = TASKS["articulated_opening"][0].moving
    c, h = ft.aabb(door.corners())
    assert np.allclose(ft.moving_tensor(door) * L0, np.concatenate([c, h])) and abs(c[0]) > 0.1  # offset from pivot


# ------------------------------------------------------------ candidate conventions

@pytest.mark.parametrize("name", sorted(ALL))
def test_candidate_conventions(name):
    scene, regions, options = ALL[name]
    f = ft.extract(scene, regions, options)
    by_id = {r.region_id: r for r in regions}
    names = ft.CANDIDATE_COLUMNS
    for p, o in enumerate(options):
        u, iv = f.candidates[p], o.intervention
        if iv.kind is InterventionKind.RELOCATE:
            i = [e.eid for e in scene.entities].index(iv.entity_id)
            after = ft.boxes_aabb(orc.apply_option(scene, iv).entities[i].boxes)[0]
            now = ft.boxes_aabb(scene.entities[i].boxes)[0]
            assert np.allclose(u[:2], [1, 0]) and f.affected[p] == i
            assert np.allclose(u[cols(names, "px", "py", "pz")] * L0, after)
            assert np.allclose(u[cols(names, "dx", "dy", "dz")] * L0, after - now)
            assert np.allclose(u[8:11] * L0, by_id[o.region_id].center) and np.allclose(u[11:] * L0, by_id[o.region_id].half)
        else:
            assert np.allclose(u[:2], [0, 1]) and np.allclose(u[8:], 0)
            assert np.allclose(u[2:5] * L0, iv.params)
            assert np.allclose(u[5:8] * L0, ft.reference_point(orc.apply_option(scene, iv).motion))
            assert f.affected[p] == ft.NO_ENTITY


def test_every_shift_target_uses_no_entity_in_every_family():
    seen = set()
    for name, (scene, regions, options) in ALL.items():
        f = ft.extract(scene, regions, options)
        for p, o in enumerate(options):
            if o.intervention.kind is InterventionKind.SHIFT_TARGET:
                assert f.affected[p] == ft.NO_ENTITY, (name, p)
                seen.add((name, bool(scene.fixture)))
            else:
                assert f.affected[p] >= 0
    assert {"storage_extraction", "articulated_opening", "storage_insertion"} <= {n for n, _ in seen}
    assert {fx for _, fx in seen} == {True, False}  # covered with and without a fixture


def test_shift_reference_points_per_motion_type():
    hinge = TASKS["articulated_opening"][0].motion
    assert np.allclose(ft.reference_point(hinge), hinge.pivot)
    poly = TASKS["storage_insertion"][0].motion
    assert np.allclose(ft.reference_point(poly), poly.segments[0].start)
    lin = STAGE1["M4_envelope_coupling"][0].motion
    assert np.allclose(ft.reference_point(lin), lin.start)


# ------------------------------------------------------------ sensitivity

def test_action_sensitivity():
    scene, regions, options = STAGE1["M1_independent"]
    other = replace(scene, motion=replace(scene.motion, goal=(0.25, 0.01, scene.motion.goal[2])))
    f, g = ft.extract(scene, regions, options), ft.extract(other, regions, options)
    assert not np.allclose(f.action, g.action) and np.array_equal(f.entities, g.entities)


def test_size_sensitivity_same_centroid():
    scene, regions, options = STAGE1["M1_independent"]
    i = next(k for k, e in enumerate(scene.entities) if e.entity.movable)
    ent = scene.entities[i]
    bigger = replace(ent, boxes=(replace(ent.boxes[0], half=tuple(1.5 * h for h in ent.boxes[0].half)),))
    other = replace(scene, entities=scene.entities[:i] + (bigger,) + scene.entities[i + 1:])
    f, g = ft.extract(scene, regions, options), ft.extract(other, regions, options)
    diff = np.argwhere(~np.isclose(f.entities, g.entities))
    assert len(diff) and set(diff[:, 0]) == {i} and set(diff[:, 1]) <= {3, 4, 5}


# ------------------------------------------------------------ no oracle leakage

def test_input_contains_only_contract_tensors():
    assert {fl.name for fl in fields(ft.SceneFeatures)} == {"entities", "action", "moving", "candidates", "affected"}
    forbidden = ("F", "G", "c_", "conflict", "blocker", "B0", "cause", "S_star", "alpha", "beta", "valid", "cost",
                 "status", "family", "id", "seed")
    for name in ft.ENTITY_COLUMNS + ft.ACTION_COLUMNS + ft.MOVING_COLUMNS + ft.CANDIDATE_COLUMNS:
        assert not any(name.startswith(b) or b in name.split("_") for b in forbidden), name


@pytest.mark.parametrize("name", sorted(ALL))
def test_extraction_never_calls_the_oracle(name, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("feature extraction touched the oracle")
    for mod, attrs in ((po, ("assess", "conflict_3d", "feasibility_excluding", "is_valid", "admissible_minimal_repairs")),
                       (pe, ("sweep", "poses")), (cs, ("validity",)), (ds, ("label",)),
                       (orc, ("evaluate", "envelope_sweep", "repair_table", "causes", "direct_row", "repair_result",
                              "pair_effect", "validity_edge"))):
        for a in attrs:
            monkeypatch.setattr(mod, a, boom)
    monkeypatch.setattr(mj_scene.GeomWorld, "__init__", boom)
    ft.extract(*ALL[name])


def _renamed(scene, regions, options):
    ren = {e.eid: f"zz{k}" for k, e in enumerate(scene.entities)}
    ents = tuple(replace(e, entity=replace(e.entity, entity_id=ren[e.eid])) for e in scene.entities)
    regs = tuple(replace(r, region_id=f"R{k}") for k, r in enumerate(regions))
    rmap = {r.region_id: n.region_id for r, n in zip(regions, regs)}
    opts = []
    for k, o in enumerate(options):
        iv = o.intervention
        iv = replace(iv, intervention_id=f"opt{k}", entity_id=ren.get(iv.entity_id, iv.entity_id), length=3)
        opts.append(RepairOption(iv, rmap.get(o.region_id)))
    return replace(scene, family="renamed", entities=ents), regs, tuple(opts)


@pytest.mark.parametrize("name", sorted(ALL))
def test_ids_names_family_and_cost_do_not_enter(name):
    f, g = ft.extract(*ALL[name]), ft.extract(*_renamed(*ALL[name]))
    for fl in fields(ft.SceneFeatures):
        assert np.array_equal(getattr(f, fl.name), getattr(g, fl.name))


def test_labels_change_but_features_change_only_in_moved_geometry():
    scene, regions, options = STAGE1["M1_independent"]
    before = orc.causes(orc.evaluate(scene))
    eid = sorted(before.blockers)[0]
    i = [e.eid for e in scene.entities].index(eid)
    ent = scene.entities[i]
    away = replace(ent, boxes=tuple(replace(b, center=(b.center[0], b.center[1] + math.copysign(0.05, b.center[1]),
                                                       b.center[2])) for b in ent.boxes))
    other = replace(scene, entities=scene.entities[:i] + (away,) + scene.entities[i + 1:])
    assert orc.causes(orc.evaluate(other)).blockers != before.blockers   # oracle labels differ
    f, g = ft.extract(scene, regions, options), ft.extract(other, regions, options)
    d_e = np.argwhere(~np.isclose(f.entities, g.entities))
    assert set(d_e[:, 0]) == {i} and set(d_e[:, 1]) <= {0, 1, 2}
    d_c = np.argwhere(~np.isclose(f.candidates, g.candidates))
    moved = {p for p, o in enumerate(options) if o.intervention.entity_id == eid}
    assert set(d_c[:, 0]) <= moved and set(d_c[:, 1]) <= set(cols(ft.CANDIDATE_COLUMNS, "dx", "dy", "dz"))
    assert np.array_equal(f.action, g.action) and np.array_equal(f.moving, g.moving)
    assert np.array_equal(f.affected, g.affected)


def test_contract_record_is_complete():
    c = ft.contract()
    assert c["K_A"] == 8 and c["L0_m"] == 0.5 and len(c["candidate_columns"]) == 14
    assert "global normalized progress" in c["conventions"] and "union AABB" in c["conventions"]
