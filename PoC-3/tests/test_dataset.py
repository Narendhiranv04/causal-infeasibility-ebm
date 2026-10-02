"""PoC-3 Stage 2 dataset tests: pre-declared protocol, bitsets, determinism, exact labels, digest."""

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from poc2 import dataset as pd
from poc3 import dataset as ds
from poc3 import features as ft

OUT = Path(__file__).resolve().parents[1] / "out" / "s2"
N_SMALL = 4  # first slots of each validation stream: 3 repairable + 1 negative


@pytest.fixture(scope="module")
def small():
    rows, stats = [], []
    for fam in ds.FAMILIES:
        r, s = ds.generate_stream("val", fam, N_SMALL)
        rows += r
        stats.append(s)
    records, arrays = ds.assemble(rows)
    return records, arrays, stats


# ------------------------------------------------------------ pre-declared protocol

def test_predeclared_protocol():
    assert ds.FAMILIES == ("make_space", "storage_insertion", "storage_extraction", "articulated_opening")
    assert [ds.FAMILY_CODE[f] for f in ds.FAMILIES] == [0, 1, 2, 3]
    assert ds.SPLITS == ("train", "val", "test") and ds.SPLIT_SEED == {"train": 31, "val": 41, "test": 51}
    assert ds.COMPOSITION == {"train": (420, 140), "val": (90, 30), "test": (90, 30)}
    assert sum(sum(c) for c in ds.COMPOSITION.values()) * len(ds.FAMILIES) == 3200
    for s in ds.SPLITS:
        plan = ds.slots(s)
        assert all((i == "negative") == (j % 4 == 3) for j, i in enumerate(plan))
        assert ds.max_attempts(s) == 10 * len(plan)
    assert ds.P_MAX == 10 and ds.N_STATES == 1024 and ds.FINE_STEP == 1e-3


def test_split_streams_are_disjoint_and_not_poc2():
    seen = set()
    for s in ds.SPLITS:
        for f in ds.FAMILIES:
            for a in range(ds.max_attempts(s)):
                key = tuple(ds.stream(s, f, a))
                assert key[:2] == (ds.SPLIT_SEED[s], ds.FAMILY_CODE[f]) and key not in seen
                seen.add(key)
    assert pd.SEED not in ds.SPLIT_SEED.values()


def test_acceptance_never_inspects_interaction_structure():
    src = inspect.getsource(ds)
    for name in ("compatible_mobius", "reconstruct", "order_decision", "interaction_edges", "density",
                 "graph_metrics", "geometric_class", "validity_binding", "hopfield", "EnergyModel"):
        assert re.search(rf"\b{name}\(", src) is None, name
    assert inspect.getsource(ds.judge).count("pd.rejection_reason") == 1


# ------------------------------------------------------------ bitsets

def test_every_state_index_round_trips_to_its_own_bit():
    for x in range(ds.N_STATES):
        packed = ds.pack_states({x}, 10)
        bits = np.unpackbits(packed, bitorder="little")
        assert packed.shape == (128,) and packed.dtype == np.uint8
        assert packed[x // 8] == 1 << (x % 8) and bits.sum() == 1 and bits[x] == 1
        assert ds.unpack_states(packed, 10) == {x}


def test_random_masks_round_trip_and_high_bits_are_rejected():
    rng = np.random.default_rng(107)
    for P in range(1, 11):
        for _ in range(20):
            S = frozenset(int(x) for x in np.flatnonzero(rng.random(2 ** P) < 0.3))
            packed = ds.pack_states(S, P)
            assert ds.unpack_states(packed, P) == S
            assert not np.unpackbits(packed, bitorder="little")[2 ** P:].any()
        with pytest.raises(ValueError):
            ds.pack_states({2 ** P}, P)
        if P < 10:
            with pytest.raises(ValueError):
                ds.unpack_states(ds.pack_states({2 ** P}, 10), P)


# ------------------------------------------------------------ generation and exact labels

def test_small_streams_are_deterministic(small):
    records, arrays, _ = small
    again = ds.assemble(ds.generate_stream("val", "storage_extraction", N_SMALL)[0])
    sel = [j for j, r in enumerate(records) if r["family"] == "storage_extraction"]
    assert again[0] == [records[j] for j in sel]
    for k in ds.ARRAYS:
        assert np.array_equal(again[1][k], arrays[k][sel])


def test_rows_rebuild_and_relabel_exactly(small):
    records, arrays, _ = small
    for j, r in enumerate(records):
        assert ds.check_row(r, arrays, j) == []
        assert r["scene_id"] == r["spec"]["scene_id"] == ds.scene_id(r["split"], r["family"], r["slot"])


def test_records_hold_no_labels_and_features_still_extract(small):
    records, arrays, _ = small
    allowed = {"scene_id", "split", "family", "intent", "slot", "stream", "spec", "candidates"}
    assert all(set(r) == allowed for r in records)
    assert set(arrays) == set(ds.ARRAYS) and all(a.dtype.kind in "iub" for a in arrays.values())
    for r in records:
        f = ft.extract(*ds.build(pd.spec_from_json(r["spec"])))
        assert len(f.candidates) == len(r["candidates"])


def test_global_checks_on_small_set(small):
    records, arrays, stats = small
    checks = ds.global_checks(records, arrays)
    assert all(v for k, v in checks.items() if isinstance(v, bool) and k != "counts_exact")
    assert not checks["counts_exact"] and checks["duplicate_spec_ids"] == []
    assert all(s["accepted"] == N_SMALL and set(s["rejections"]) == set(ds.REASONS) for s in stats)


def test_tampering_is_detected(small):
    records, arrays, _ = small
    j = 0
    bad = {k: v.copy() for k, v in arrays.items()}
    P = int(bad["P"][j])
    opt = set(ds.unpack_states(bad["optimal"][j], P))
    bad["optimal"][j] = ds.pack_states(opt ^ {next(x for x in range(2 ** P) if x not in opt)}, P)
    bad["cost"][j][0] += 1
    errors = ds.check_row(records[j], bad, j)
    assert "S_star_mismatch" in errors and "cost_mismatch" in errors


# ------------------------------------------------------------ storage and content digest

def test_digest_is_content_based_and_sensitive(small, tmp_path):
    records, arrays, _ = small
    d = ds.content_digest(records, arrays)
    ds.save(tmp_path, records, arrays, {"note": "test"})
    assert ds.content_digest(*ds.load(tmp_path, expected=d)) == d
    np.savez(tmp_path / "labels.npz", **arrays)  # different container (uncompressed), same content
    assert ds.content_digest(*ds.load(tmp_path, expected=None)) == d
    with pytest.raises(ValueError):
        ds.load(tmp_path, expected="0" * 64)
    flipped = {k: v.copy() for k, v in arrays.items()}
    flipped["admissible"][0, 0] ^= 1
    assert ds.content_digest(records, flipped) != d
    assert ds.content_digest(records[::-1], arrays) != d


@pytest.mark.skipif(ds.STAGE2_DIGEST is None or not (OUT / "labels.npz").exists(), reason="dataset not generated")
def test_pinned_dataset_passes_global_checks():
    records, arrays = ds.load(OUT)  # verifies the pinned content digest
    checks = ds.global_checks(records, arrays)
    assert len(records) == 3200 and all(v for v in checks.values() if isinstance(v, bool))
