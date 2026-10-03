"""v0.1 acceptance gates on the canonical scenes (read from canonical/V*/oracle.json)."""
import json
from pathlib import Path

import numpy as np
import pytest

from artrecourse.variants import check_variant, n_real_compat, repair_edges

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "canonical"
VS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]


def load(v):
    f = CANON / v / "oracle.json"
    if not f.exists():
        pytest.skip("run scripts/build_canonical.py first")
    return json.loads(f.read_text())


def as_res(rec):
    lab = rec["labels"]
    res = dict(lab)
    res.update(consulted_ambiguities=rec["quality"]["consulted_ambiguities"],
               executable_initially=[k for k, x in lab["per_action"].items() if k != "TARGET" and x["executable_initially"]],
               intervention_object={i["id"]: i["object"] for i in rec["inputs"]["candidate_interventions"]})
    return res


@pytest.mark.parametrize("v", VS)
def test_initial_objects_and_destinations_on_named_slots(v):
    rec = load(v)
    topo = rec["inputs"]["placement_topology"]
    for o, sl in rec["inputs"]["initial_slots"].items():
        assert sl["slot"] in topo
        assert sl["orientation"] in topo[sl["slot"]]["orientations"][next(x["category"] for x in rec["inputs"]["objects"]
                                                                            if x["key"] == o)]
    for iv in rec["inputs"]["candidate_interventions"]:
        assert iv["destination_slot"] in topo and topo[iv["destination_slot"]]["reachable"]
        assert iv["destination_orientation"]


@pytest.mark.parametrize("v", VS)
def test_no_ad_hoc_offsets_and_category_compatibility(v):
    rec = load(v)
    assert rec["quality"]["ad_hoc_offsets"] == 0
    assert rec["quality"]["gate_violations"] == []
    for o in rec["spec"]["objects"]:
        assert tuple(o["dxy"]) == (0.0, 0.0)
    for i in rec["spec"]["interventions"]:
        assert tuple(i["dxy"]) == (0.0, 0.0)


@pytest.mark.parametrize("v", VS)
def test_capacity_one_slots_never_double_occupied(v):
    rec = load(v)
    topo = rec["inputs"]["placement_topology"]
    for st in rec["labels"]["state_conditioned"]:
        used = {}
        for o, so in st["state_slots"].items():
            sl = so.split(":")[0]
            for x in (sl, *topo[sl]["overlaps"]):
                assert x not in used or used[x] == o, (st["state"], sl, used.get(x), o)
            used[sl] = o


@pytest.mark.parametrize("v", VS)
def test_target_fully_robot_admissible_and_atomic(v):
    rec = load(v)
    assert rec["quality"]["target_fully_admissible"] is True
    assert rec["inputs"]["target_action"]["atomic"] is True
    assert rec["inputs"]["target_action"]["name"] in ("PUSH_RACK", "CLOSE_DOOR")


@pytest.mark.parametrize("v", VS)
def test_coarse_fine_and_no_consulted_ambiguity(v):
    rec = load(v)
    assert rec["quality"]["coarse_fine"]["agree"], rec["quality"]["coarse_fine"]["disagreements"]
    assert rec["quality"]["consulted_ambiguities"] == []
    assert rec["quality"]["coarse_fine"]["fine_consulted_ambiguities"] == []


@pytest.mark.parametrize("v", VS)
def test_every_prerequisite_edge_has_interventional_proof(v):
    rec = load(v)
    for p in rec["labels"]["sequence_proofs"]:
        for e in p["edges"]:
            assert any(val is False for val in e["proof"].values()) and any(val is True for val in e["proof"].values())
            assert e["blocker"] and "type" in e["cause"]


def test_v3_destination_statically_free():
    res = as_res(load("V3"))
    es = [e for e in repair_edges(res) if not e["cause"]["destination_occupied"]]
    assert es and all(e["destination_slot_free_before_prerequisite"] for e in es)


def test_v4_two_distinct_mechanisms():
    res = as_res(load("V4"))
    es = repair_edges(res)
    assert any(e["cause"]["destination_occupied"] for e in es)
    assert any(e["cause"]["swept_volume"] and not e["cause"]["destination_occupied"] for e in es)


def test_v5_blocker_is_not_first_corrective_object():
    rec = load("V5")
    res = as_res(rec)
    for seq in res["optimal_sequences"]:
        assert res["intervention_object"][seq[0]] not in res["direct_target_blockers"]


def test_v7_compatibility_density_and_edges():
    res = as_res(load("V7"))
    assert 1 <= n_real_compat(res) <= 5
    assert 3 <= len(repair_edges(res)) <= 6


@pytest.mark.parametrize("v", ["V4", "V5", "V7"])
def test_carried_object_sweep_dependency_present(v):
    res = as_res(load(v))
    assert any(e["cause"]["mechanism"] == "carried_object" for e in repair_edges(res))


@pytest.mark.parametrize("v", VS)
def test_structural_definition(v):
    rec = load(v)
    ok, why = check_variant(v, as_res(rec), {"objects": rec["inputs"]["objects"],
                                            "candidate_interventions": rec["inputs"]["candidate_interventions"]})
    assert ok, why
    assert rec["quality"]["accepted"]


def test_fixture_audit_selected_by_measured_criteria():
    from artrecourse.assets import load_asset_config

    a = json.loads((ROOT / "out" / "fixture_audit.json").read_text())
    assert a["selected"] == load_asset_config()["fixture"]["id"]
    r = a["robot"][a["selected"]]
    assert r["PUSH_RACK_upper"]["coverage"] == 1.0 and r["CLOSE_DOOR"]["coverage"] == 1.0
    assert max(a["final_scores"].values()) == a["final_scores"][a["selected"]]
    _ = np


def test_state_conditioned_export_consistent_with_oracle(problems):
    """Local effects re-derived by actual intervention match the exported labels (V5)."""
    from artrecourse.oracle import state_conditioned_export

    pb, res = problems["V5"]
    exp = state_conditioned_export(pb)
    for st in exp[:6]:
        s = tuple(st["state"])
        for eff in st["effects"]:
            p = pb.iv_by_id[eff["p"]]
            s1 = pb.apply(s, p)
            for q in eff["enables"]:
                assert not pb.full_check_intervention(pb.iv_by_id[q], s) and pb.full_check_intervention(pb.iv_by_id[q], s1)
