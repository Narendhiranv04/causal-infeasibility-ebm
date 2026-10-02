"""PoC-3 metric tests: soft targets, exact outcome partition, scene-balanced summaries, seed aggregation."""

import math

import numpy as np
import pytest

from poc2 import oracle as orc
from poc2 import scenes
from poc2 import structure as st
from poc3 import metrics as mt

CASES = {c.name: c for c in scenes.stage1_cases()}


@pytest.fixture(scope="module")
def m2():
    case = CASES["M2_placement_competition"]  # static validity binds: V-invalid consistent states exist
    t = orc.repair_table(case.scene, case.options)
    return t, st.exact_decision(t)


def test_soft_target_weights_tied_optima_equally():
    assert np.allclose(mt.soft_target({0b011, 0b101}, 3), [1.0, 0.5, 0.5])
    assert np.allclose(mt.soft_target({0}, 4), 0.0)
    assert np.allclose(mt.soft_target({0b1000}, 4), [0, 0, 0, 1])
    with pytest.raises(ValueError):
        mt.soft_target(set(), 3)


def test_brier_is_mean_squared_error_over_candidates():
    assert mt.brier([0.2, 0.7], [0.2, 0.7]) == 0.0
    assert math.isclose(mt.brier([1.0, 0.0, 0.5], [0.0, 0.0, 1.0]), (1 + 0 + 0.25) / 3)


def test_every_state_has_exactly_one_outcome_and_minimal_equals_hit(m2):
    t, opt = m2
    for x in range(len(t.M)):
        r = mt.score_state(x, t.M, t.V, t.F, t.K, opt)
        assert sum(bool(r[k]) for k in mt.OUTCOMES) == 1
        assert r["minimal"] == r["hit"] and r["n_pred"] == bin(x).count("1") and r["empty"] == (x == 0)
        assert (r["excess_cost"] is None) == (not r["valid_feasible"])


def test_outcome_classes_on_a_known_scene(m2):
    t, opt = m2
    x = next(iter(opt))
    best = mt.score_state(x, t.M, t.V, t.F, t.K, opt)
    assert best["hit"] and best["valid_feasible"] and best["minimal"] and best["excess_cost"] == 0
    assert mt.score_state(0, t.M, t.V, t.F, t.K, opt)["infeasible"]       # doing nothing leaves it blocked
    m_bad = next(x for x in range(len(t.M)) if t.M[x] == 0)
    assert mt.score_state(m_bad, t.M, t.V, t.F, t.K, opt)["M_violation"]
    v_bad = next(x for x in range(len(t.M)) if t.M[x] == 1 and t.V[x] == 0)
    assert mt.score_state(v_bad, t.M, t.V, t.F, t.K, opt)["V_violation"]
    extra = [x for x in range(len(t.M)) if t.M[x] == 1 and t.V[x] == 1 and t.F[x] == 0 and x not in opt]
    if extra:
        r = mt.score_state(extra[0], t.M, t.V, t.F, t.K, opt)
        assert r["valid_feasible"] and not r["hit"] and r["excess_cost"] > 0


def _row(**kw):
    base = {"hit": False, "valid_feasible": False, "minimal": False, "M_violation": False, "V_violation": False,
            "infeasible": False, "excess_cost": None, "n_pred": 0, "empty": True, "brier": 0.0,
            "family": "make_space", "intent": "repairable"}
    return base | kw


def test_summarize_is_scene_balanced_with_histogram():
    rows = [_row(hit=True, valid_feasible=True, minimal=True, excess_cost=0, n_pred=2, empty=False, brier=0.1),
            _row(valid_feasible=True, excess_cost=2, n_pred=3, empty=False, brier=0.3),
            _row(M_violation=True, n_pred=2, empty=False, brier=0.5),
            _row(infeasible=True, brier=0.1)]
    s = mt.summarize(rows)
    assert s["n"] == 4 and s["hit"] == 0.25 and s["valid_feasible"] == 0.5 and s["M_violation"] == 0.25
    assert s["n_valid_feasible"] == 2 and s["mean_excess_cost_valid_feasible"] == 1.0
    assert s["mean_n_pred"] == 7 / 4 and math.isclose(s["brier"], 0.25) and s["empty"] == 0.25
    assert s["size_histogram"]["0"] == 1 and s["size_histogram"]["2"] == 2 and s["size_histogram"]["3"] == 1
    assert sum(s["size_histogram"].values()) == 4 and len(s["size_histogram"]) == 11


def test_grouped_separates_negatives_and_families():
    rows = [_row(hit=True), _row(intent="negative", empty=True, hit=True),
            _row(intent="negative", empty=False, n_pred=1, family="storage_insertion")]
    g = mt.grouped(rows, ("make_space", "storage_insertion"))
    assert g["overall"]["repairable"]["n"] == 1 and g["overall"]["negative"]["n"] == 2
    assert g["overall"]["negative"]["false_positive_repair_rate"] == 0.5
    assert g["per_family"]["storage_insertion"]["negative"]["false_positive_repair_rate"] == 1.0
    assert g["per_family"]["storage_insertion"]["repairable"] == {"n": 0}


def test_across_seeds_mean_and_sample_std():
    agg = mt.across_seeds([{"a": 1.0, "b": {"c": 2}}, {"a": 2.0, "b": {"c": 2}}, {"a": 3.0, "b": {"c": 2}}])
    assert agg["a"] == {"mean": 2.0, "std": 1.0} and agg["b"]["c"] == {"mean": 2.0, "std": 0.0}
    assert math.isnan(mt.across_seeds([1.0, math.nan, 2.0])["mean"])


# ------------------------------------------------------------ Stage 4 metrics

def test_set_scores():
    assert mt.set_scores({1, 2}, {1, 2}) == {"set_recovery": True, "set_precision": 1.0, "set_recall": 1.0}
    assert mt.set_scores({1, 5}, {1, 2, 3, 4}) == {"set_recovery": False, "set_precision": 0.5, "set_recall": 0.25}


def test_distribution_scores_are_exact_and_normalized():
    E, states = np.array([0.0, 1.0, 1.0, 3.0]), np.array([0, 1, 2, 3])
    d = mt.distribution_scores(E, states, {1, 2}, 2, 1.0)
    p = np.exp(-E) / np.exp(-E).sum()
    assert math.isclose(d["prob_sum"], 1.0) and math.isclose(mt.boltzmann(E, 0.3).sum(), 1.0)
    assert math.isclose(d["nll"], -np.log(p[1])) and math.isclose(d["kl"], d["nll"] - np.log(2))
    assert math.isclose(d["mass"], p[1] + p[2])
    y_hat = np.array([p[1] + p[3], p[2] + p[3]])
    assert math.isclose(d["brier"], np.mean((y_hat - 0.5) ** 2)) and math.isclose(d["mae"], np.mean(np.abs(y_hat - 0.5)))
    extreme = mt.distribution_scores(np.array([0.0, 2000.0]), np.array([0, 1]), {1}, 1, 1.0)
    assert math.isclose(extreme["nll"], 2000.0)  # stable, no log(0)


def test_paired_bootstrap_is_deterministic_with_seed_107():
    d = np.random.default_rng(0).normal(0.1, 0.3, 200)
    a, b = mt.paired_bootstrap(d), mt.paired_bootstrap(d)
    assert a == b and (mt.BOOTSTRAP_RESAMPLES, mt.BOOTSTRAP_SEED) == (10_000, 107)
    assert a["ci95"][0] <= a["mean_diff"] <= a["ci95"][1]
    assert mt.paired_bootstrap(np.zeros(50)) == {"n": 50, "mean_diff": 0.0, "ci95": [0.0, 0.0], "excludes_zero": False}


def test_collapse_rule():
    rows = [_row(n_pred=1, empty=False)] * 96 + [_row(n_pred=2, empty=False)] * 4
    assert mt.collapse(rows)["collapsed"] and mt.collapse(rows)["max_size_share"] == 0.96
    mixed = [_row(n_pred=1, empty=False)] * 60 + [_row(n_pred=2, empty=False)] * 40 + [_row(intent="negative")] * 50
    assert not mt.collapse(mixed)["collapsed"] and mt.collapse(mixed)["n_distinct_sizes"] == 2
    assert mt.collapse([_row()] * 60 + [_row(n_pred=1, empty=False)] * 40)["collapsed"]  # 60 % empty


def test_summarize_extra_keys_and_optional_brier():
    rows = [_row(set_recall=0.5), _row(set_recall=1.0)]
    s = mt.summarize([{k: v for k, v in r.items() if k != "brier"} for r in rows], extra=("set_recall",))
    assert s["set_recall"] == 0.75 and "brier" not in s
