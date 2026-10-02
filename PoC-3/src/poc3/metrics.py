"""PoC-3 evaluation metrics (plan3.md sections 34, 35, 50, 60). Labels are used here only, never as inputs.

OUTCOME OF ONE PREDICTED STATE x (bitmask over the scene's P candidates), exactly one of:
  M_violation   M(x) = 0 (two options of one choice group)
  V_violation   M(x) = 1, V(x) = 0 (repaired static scene invalid)
  infeasible    M(x) = V(x) = 1, F(x) = 1 (the prescribed action is still blocked after the repair)
  valid_feasible M(x) = V(x) = 1, F(x) = 0
  hit = 1[x in S*]; minimal = valid_feasible and K(x) = K*. Because S* is the set of ALL valid feasible
  states of minimal cost, minimal == hit by definition (both are reported). excess_cost = K(x) - K* is
  averaged over valid-feasible predictions only.
Soft target y_p = (1 / |S*|) sum_{x in S*} x_p (tied optima weighted equally). Brier = mean_p (pi_p - y_p)^2
per scene, then averaged equally across scenes. Across seeds: mean and sample std (ddof = 1).
"""

import math

import numpy as np

OUTCOMES = ("valid_feasible", "M_violation", "V_violation", "infeasible")
RATES = ("hit", "valid_feasible", "minimal", "M_violation", "V_violation", "infeasible", "empty")
P_MAX = 10


def soft_target(optimal, P: int) -> np.ndarray:
    states = sorted(optimal)
    if not states:
        raise ValueError("a scene always has at least one oracle optimum")
    bits = (np.array(states)[:, None] >> np.arange(P)[None, :]) & 1
    return bits.mean(axis=0)


def brier(pi: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(pi, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def score_state(x: int, M, V, F, K, optimal) -> dict:
    """Exact oracle outcome of one predicted state (M, V, F, K indexed by state, as in the PoC-2 table)."""
    m = M[x] == 1
    v = bool(m and V[x] == 1)
    ok = bool(v and F[x] == 0)
    k_star = int(K[next(iter(optimal))])
    return {"hit": x in optimal, "valid_feasible": ok, "minimal": ok and int(K[x]) == k_star,
            "M_violation": not m, "V_violation": bool(m) and not v, "infeasible": v and F[x] == 1,
            "excess_cost": int(K[x]) - k_star if ok else None, "n_pred": int(x).bit_count(), "empty": x == 0}


def summarize(rows: list[dict]) -> dict:
    """Scene-balanced summary of per-scene rows (each row: score_state fields + 'brier')."""
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    out.update({k: float(np.mean([r[k] for r in rows])) for k in RATES})
    excess = [r["excess_cost"] for r in rows if r["excess_cost"] is not None]
    out["n_valid_feasible"] = len(excess)
    out["mean_excess_cost_valid_feasible"] = float(np.mean(excess)) if excess else math.nan
    out["mean_n_pred"] = float(np.mean([r["n_pred"] for r in rows]))
    out["brier"] = float(np.mean([r["brier"] for r in rows]))
    sizes = np.bincount([r["n_pred"] for r in rows], minlength=P_MAX + 1)
    out["size_histogram"] = {str(k): int(v) for k, v in enumerate(sizes)}
    return out


def grouped(rows: list[dict], families) -> dict:
    """All / repairable (primary) / negative, overall and per family; negatives add the false-positive rate."""
    def block(sel):
        rep = [r for r in sel if r["intent"] == "repairable"]
        neg = [r for r in sel if r["intent"] == "negative"]
        b = {"all": summarize(sel), "repairable": summarize(rep), "negative": summarize(neg)}
        if neg:
            b["negative"]["false_positive_repair_rate"] = float(np.mean([not r["empty"] for r in neg]))
        return b
    return {"overall": block(rows), "per_family": {f: block([r for r in rows if r["family"] == f]) for f in families}}


def across_seeds(per_seed: list[dict]) -> dict:
    """Mean and sample std (ddof = 1) of every numeric leaf across seeds (histograms: mean counts)."""
    first = per_seed[0]
    if isinstance(first, dict):
        return {k: across_seeds([s[k] for s in per_seed]) for k in first}
    vals = np.array(per_seed, dtype=float)
    if np.isnan(vals).any():
        return {"mean": math.nan, "std": math.nan}
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
