"""Interventional labels, oracle optimality and canonical variant definitions."""
import itertools
import json
from pathlib import Path

import pytest

from artrecourse.oracle import state_after
from artrecourse.records import LABEL_KEYS
from artrecourse.variants import check_variant

CANON = Path(__file__).resolve().parents[1] / "canonical"


def test_enable_labels_reproduce_by_actual_intervention(problems):
    for pb, res in problems.values():
        for e in res["enable_edges"]:
            p, q = pb.iv_by_id[e["from"]], pb.iv_by_id[e["to"]]
            assert pb.full_check_intervention(p, pb.s0)
            assert not pb.full_check_intervention(q, pb.s0)
            assert pb.full_check_intervention(q, pb.apply(pb.s0, p))


def test_compatibility_labels_symmetric(problems):
    for pb, res in problems.values():
        for kind, pairs in res["compatibility_edges"].items():
            for pr in pairs:
                a, b = pr[0], pr[1]
                assert a < b    # stored once, unordered
                if kind == "destination_overlap":
                    ia, ib = pb.iv_by_id[a], pb.iv_by_id[b]
                    d1 = pb.eng.static_distance(ia.obj, pb.pose_tuple(ia.obj, ia.pose_idx), ib.obj, pb.pose_tuple(ib.obj, ib.pose_idx))
                    d2 = pb.eng.static_distance(ib.obj, pb.pose_tuple(ib.obj, ib.pose_idx), ia.obj, pb.pose_tuple(ia.obj, ia.pose_idx))
                    assert abs(d1 - d2) < 1e-3


def test_enablement_need_not_be_symmetric(problems):
    edges = {(e["from"], e["to"]) for _, res in problems.values() for e in res["enable_edges"]}
    assert edges and any((b, a) not in edges for a, b in edges)


def test_bfs_sequences_executable_prefixes_and_final_feasible(problems):
    for pb, res in problems.values():
        assert res["optimal_sequences"]
        for seq in res["optimal_sequences"]:
            s = pb.s0
            for ivid in seq:
                iv = pb.iv_by_id[ivid]
                assert pb.full_check_intervention(iv, s), (ivid, s)   # every prefix executable (brute force)
                s = pb.apply(s, iv)
            assert pb.full_check_target(s) == 0
        assert pb.full_check_target(pb.s0) == 1


def test_no_shorter_valid_sequence_exists(problems):
    for pb, res in problems.values():
        L = res["optimal_cost"]
        for n in range(1, L):
            for seq in itertools.permutations([iv.id for iv in pb.ivs], n):
                if len({pb.iv_by_id[i].obj for i in seq}) < n:
                    continue
                s, ok = pb.s0, True
                for ivid in seq:
                    if not pb.executable(pb.iv_by_id[ivid], s):
                        ok = False
                        break
                    s = pb.apply(s, pb.iv_by_id[ivid])
                assert not (ok and not pb.F(s)), seq


def test_sequence_proofs_are_interventional(problems):
    pb, res = problems["V5"]
    for proof in res["sequence_proofs"]:
        seq = proof["sequence"]
        for e in proof["edges"]:
            if e["to"] == "TARGET":
                continue
            k = seq.index(e["to"])
            without = state_after(pb, [x for x in seq[:k] if x != e["from"]])
            assert not pb.full_check_intervention(pb.iv_by_id[e["to"]], without)
            assert pb.full_check_intervention(pb.iv_by_id[e["to"]], state_after(pb, seq[:k]))


@pytest.mark.parametrize("v", ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"])
def test_canonical_satisfies_structural_definition(v):
    f = CANON / v / "oracle.json"
    if not f.exists():
        pytest.skip("run scripts/build_canonical.py first")
    rec = json.loads(f.read_text())
    lab = rec["labels"]
    res = dict(lab)
    res.update(consulted_ambiguities=rec["quality"]["consulted_ambiguities"],
               executable_initially=[k for k, x in lab["per_action"].items() if k != "TARGET" and x["executable_initially"]],
               target_clearance=lab["target_clearance"],
               intervention_object={i["id"]: i["object"] for i in rec["inputs"]["candidate_interventions"]})
    scene = {"objects": rec["inputs"]["objects"], "candidate_interventions": rec["inputs"]["candidate_interventions"]}
    ok, why = check_variant(v, res, scene)
    assert ok, why
    assert rec["quality"]["accepted"], rec["quality"]
    assert not rec["verification"]["factorisation_spot_check"]["mismatches"]


def test_labels_never_leak_into_inputs():
    for f in CANON.glob("V*/oracle.json"):
        rec = json.loads(f.read_text())
        flat = json.dumps(rec["inputs"])
        for k in LABEL_KEYS + ("min_dist", "tau_star", "blockers", "executable_initially"):
            assert f'"{k}"' not in flat, (f, k)
