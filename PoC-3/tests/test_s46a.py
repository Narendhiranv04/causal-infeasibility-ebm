"""PoC-3 Stage 4.6A tests: oracle categories, concentration, upper bound, guards, integrity, no training."""

import importlib.util
import inspect
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from poc2 import oracle as orc
from poc2 import scenes, tasks
from poc2 import structure as st
from poc3 import dataset as ds
from poc3 import metrics as mt
from poc3 import model as md
from poc3 import relational as rl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "s46a_error_audit.py"
_spec = importlib.util.spec_from_file_location("s46a_error_audit", SCRIPT)
au = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(au)
SEEDS = (7, 17, 27)


def test_categories_are_mutually_exclusive_and_exhaustive():
    universe = range(6)
    for S_star in ({1}, {1, 2}, {0, 3, 5}):
        for r in range(0, 4):
            for S1 in itertools.combinations(universe, r):
                c = au.category(S1, S_star)
                flags = [set(S1) == S_star, set(S1) != S_star and bool(set(S1) & S_star), not set(S1) & S_star]
                assert sum(flags) == 1 and c == au.CATEGORIES[flags.index(True)]


def test_R_set_and_R_top1_differ_on_a_tied_example():
    pairs = [({3}, {3, 5}), ({3, 5}, {3, 5}), ({6}, {3, 5})]   # U-OVERLAP, U-EXACT, P-TOP1
    cats = [au.category(S1, S) for S1, S in pairs]
    R_set = sum(c != "U-EXACT" for c in cats) / len(cats)
    R_top1 = sum(c == "P-TOP1" for c in cats) / len(cats)
    assert cats == ["U-OVERLAP", "U-EXACT", "P-TOP1"] and R_set == 2 / 3 and R_top1 == 1 / 3


def test_oracle_unary_top1_uses_the_lowest_state_index():
    assert au.oracle_unary_top1(frozenset({9, 4, 6})) == 4 and au.oracle_unary_top1(frozenset()) is None


def test_order_two_reconstruction_reproduces_exact_S_star_on_known_cases():
    cases = [(c.scene, c.options) for c in scenes.stage1_cases()]
    cases += [tasks.build(s)[0::2] for s, _, _ in tasks.regression_cases().values()]
    for sc, op in cases:
        t = orc.repair_table(sc, op)
        a = st.compatible_mobius(t.G, t.M)
        assert st.order_decision(t, st.reconstruct(a, 2)) == st.exact_decision(t)


def _rows(hits):
    return [{"hit": h} for h in hits]


def test_error_concentration_on_a_synthetic_example():
    cats = ["U-EXACT", "U-EXACT", "U-OVERLAP", "P-TOP1", "P-TOP1"]
    by_seed = {7: _rows([1, 0, 1, 0, 0]), 17: _rows([1, 1, 1, 0, 1]), 27: _rows([1, 1, 0, 0, 0])}
    c = au.error_concentration(by_seed, cats)
    assert c["n_errors_pooled"] == 7
    assert math.isclose(c["P_error_given_C"]["U-EXACT"], 1 / 6) and math.isclose(c["P_error_given_C"]["P-TOP1"], 5 / 6)
    assert math.isclose(c["P_C_given_error"]["P-TOP1"], 5 / 7) and math.isclose(sum(c["P_C_given_error"].values()), 1.0)
    assert math.isclose(c["enrichment_P_TOP1"], (5 / 6) / (2 / 9))
    perfect = {s: _rows([1, 1, 1, 0, 0]) for s in SEEDS}
    assert au.error_concentration(perfect, cats)["enrichment_P_TOP1"] == math.inf


def test_perfect_pair_upper_bound_formula_matches_the_counterfactual():
    cats = np.array(["U-EXACT"] * 6 + ["P-TOP1"] * 4)
    hits = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 0], dtype=float)
    fixed = np.where(cats == "P-TOP1", 1.0, hits)
    p, h = (cats == "P-TOP1").mean(), hits[cats == "P-TOP1"].mean()
    assert math.isclose(au.perfect_pair(hits.mean(), p, h), fixed.mean())


def test_repair_size_stratification_preserves_counts():
    rng = np.random.default_rng(0)
    cats, sizes = rng.choice(au.CATEGORIES, 200), rng.integers(1, 6, 200)
    table = {c: {k: int(((cats == c) & (sizes == k)).sum()) for k in range(1, 6)} for c in au.CATEGORIES}
    assert sum(sum(r.values()) for r in table.values()) == 200
    assert all(sum(r.values()) == (cats == c).sum() for c, r in table.items())


def test_bootstrap_is_deterministic_with_seed_107():
    by_a = {s: _rows(np.random.default_rng(s).integers(0, 2, 40)) for s in SEEDS}
    by_b = {s: _rows(np.random.default_rng(s + 1).integers(0, 2, 40)) for s in SEEDS}
    x, y = au.paired(by_b, by_a, list(range(40))), au.paired(by_b, by_a, list(range(40)))
    assert x == y and mt.BOOTSTRAP_SEED == 107 and not x["low_sample_exploratory"]
    assert au.paired(by_b, by_a, list(range(10)))["low_sample_exploratory"]


def test_row_guard_rejects_test_and_non_validation_rows():
    recs = [{"split": "val", "intent": "repairable"}, {"split": "test", "intent": "repairable"},
            {"split": "val", "intent": "negative"}, {"split": "train", "intent": "repairable"}]
    assert au.guard(recs, [0]) == [0]
    for j in (1, 2, 3):
        with pytest.raises(RuntimeError):
            au.guard(recs, [0, j])
    with pytest.raises(RuntimeError, match="test split"):
        au.guard(recs, [1])


def test_script_contains_no_training_and_writes_no_checkpoint():
    src = SCRIPT.read_text().replace(au.__doc__, "")
    for banned in ("tr.train(", ".step(", ".backward(", "Adam", "torch.save", "torch.optim", "optimizer", "make_model", ".pt\")"):
        assert banned not in src, banned
    writes = [ln for ln in src.splitlines() if "write_text" in ln or "savez" in ln or "open(" in ln]
    assert len(writes) == 1 and "metrics.json" in writes[0]


def _ck(kind, seed, model):
    return {"model_kind": kind, "seed": seed, "dataset_digest": ds.STAGE2_DIGEST, "state_dict": model.state_dict(),
            "architecture": rl.__doc__.strip(), "calibrated_temperature": 1.0}


@pytest.mark.parametrize("kind", ["unary", "pairwise", "rel_cf"])
def test_checkpoint_integrity_rejects_wrong_digest_kind_seed_and_keys(kind):
    torch.manual_seed(7)
    model = rl.RelationalEnergy("cf") if kind == "rel_cf" else md.EnergyModel(pairwise=kind == "pairwise")
    good = _ck(kind, 7, model)
    assert au.check_checkpoint(good, kind, 7) is not None
    for bad in (good | {"dataset_digest": "0" * 64}, good | {"seed": 17}, good | {"model_kind": "other"}):
        with pytest.raises(ValueError):
            au.check_checkpoint(bad, kind, 7)
    missing = dict(good["state_dict"])
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError):
        au.check_checkpoint(good | {"state_dict": missing}, kind, 7)
    if kind == "rel_cf":
        with pytest.raises(ValueError):
            au.check_checkpoint(good | {"architecture": "changed"}, kind, 7)


def test_decision_rule_precedence():
    base = {"R_top1": 0.2, "deficit": 0.3, "pair_ci_lo": 0.01, "enrichment": 2.0, "stratified_deficit": True,
            "perfect_gain": 0.1, "err_U_EXACT": 0.05, "err_P_TOP1": 0.6, "share_unary_errors": 0.3}
    assert au.decide(base)["outcome"] == "A"
    assert au.decide(base | {"R_top1": 0.05})["outcome"] == "B"
    assert au.decide(base | {"perfect_gain": 0.01, "err_U_EXACT": 0.2})["outcome"] == "C"
    assert au.decide(base | {"stratified_deficit": False})["outcome"] == "B"
    assert au.decide(base | {"perfect_gain": 0.0})["outcome"] == "unclassified"
    assert json.loads(json.dumps(au.decide(base)))["criteria_A"]["A1"] is True
