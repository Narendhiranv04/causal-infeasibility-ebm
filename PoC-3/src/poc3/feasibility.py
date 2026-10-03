"""PoC-3 Stage 4.6C diagnostic: dense feasibility supervision on the frozen REL-CF-PAIR-SHIFT representation.

  l(x) = b(g_0) + q^T x + sum_{p<r} Q_pr x_p x_r,     F = 1 means INFEASIBLE, F_hat(x) = 1[l(x) >= 0]
l is a FEASIBILITY POLYNOMIAL, not the final repair-selection energy: K(x) never enters it. q, Q and g_0 come from
an unchanged relational.RelationalEnergy("cf", "shift") (same relation tokens, counterfactual contexts, unary input
z_p, unary head, pair embedding and SHIFT-only mask); the only addition is the scalar scene bias
b = Linear(256, 128) -> GELU -> Linear(128, 1) on g_0. Model inputs are the five Stage-1 tensors only; oracle F is a
training target held in DenseSample. Repair from predicted feasibility: among admissible states with F_hat = 0, the
minimum exact cost K, ties to the lowest state index; if none, NO_PREDICTED_FEASIBLE_STATE (no fallback).
Dense loss: BCEWithLogits(l(x), F(x)) averaged over the scene's stable admissible states (base / fine agree), then
equally over scenes. Scene-balanced balanced accuracy = mean of the class recalls defined in that scene.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as fnn
from torch import nn

from poc3 import features as ft
from poc3 import model as md
from poc3 import relational as rl
from poc3 import train as tr

THRESHOLD_LOGIT = 0.0
NO_PREDICTED_FEASIBLE_STATE = None


class FeasibilityModel(nn.Module):
    """Frozen REL-CF-PAIR-SHIFT architecture (as `base`) plus one scene feasibility bias on g_0."""

    def __init__(self):
        super().__init__()
        self.base = rl.RelationalEnergy("cf", "shift")
        self.bias = md.mlp(rl.CONTEXT_DIM, 1, md.HIDDEN)
        self.pairwise = True

    def terms(self, entities, action, moving, candidates, affected):
        """(b, q, Q); q and Q are exactly base.forward's outputs (the base modules are reused, not copied)."""
        z, g0 = self.base.unary_inputs(entities, action, moving, candidates, affected)
        q = self.base.score(z).squeeze(-1)
        P = len(candidates)
        Q = q.new_zeros(P, P)
        if P > 1:
            p, r = torch.triu_indices(P, P, offset=1)
            pair = self.base.f_2(self.base.pair_descriptor(self.base.phi_pair(z), g0, p, r)).squeeze(-1)
            shift = candidates[:, rl.SHIFT] == 1
            pair = pair * (shift[p] | shift[r]).to(pair.dtype)
            Q = Q.index_put((p, r), pair).index_put((r, p), pair)
        return self.bias(g0).squeeze(-1), q, Q

    def forward(self, entities, action, moving, candidates, affected):
        _, q, Q = self.terms(entities, action, moving, candidates, affected)
        return q, Q


def load(state_dict: dict) -> FeasibilityModel:
    model = FeasibilityModel()
    model.load_state_dict(state_dict)
    return model.eval()


@dataclass(frozen=True)
class DenseSample:
    inputs: dict              # Stage-1 model tensors only
    states: np.ndarray        # (A,) admissible state indices, ascending (exact M = V = 1)
    bits: torch.Tensor        # (A, P) float64
    F: torch.Tensor           # (A,) float64 oracle feasibility target (1 = infeasible), never an input
    stable: torch.Tensor      # (A,) bool, base / fine F agree
    cost: np.ndarray          # (P,) exact stored costs, used only to select a repair after prediction
    optimal: frozenset[int]   # canonical Stage-2 S* (evaluation only)
    meta: dict


def make_dense_sample(scene, regions, options, states, F, stable, optimal, cost, meta: dict | None = None):
    states = np.asarray(states, dtype=np.int64)
    if list(states) != sorted(set(states.tolist())):
        raise ValueError("admissible states must be unique and ascending")
    bits = torch.as_tensor((states[:, None] >> np.arange(len(options))[None, :]) & 1, dtype=torch.float64)
    return DenseSample(md.as_tensors(ft.extract(scene, regions, options)), states, bits,
                       torch.as_tensor(np.asarray(F, dtype=float)), torch.as_tensor(np.asarray(stable, dtype=bool)),
                       np.asarray(cost, dtype=float), frozenset(optimal), meta or {})


def feasibility_logits(model: FeasibilityModel, s: DenseSample) -> torch.Tensor:
    """l(x) on the admissible states via the frozen Stage-4 polynomial with a zero-cost view (K excluded)."""
    b, q, Q = model.terms(**s.inputs)
    no_cost = SimpleNamespace(bits=s.bits, cost=torch.zeros(s.bits.shape[1], dtype=torch.float64))
    return b.double() + tr.admissible_energies(q, Q, no_cost)


def dense_loss(model: FeasibilityModel, batch: list[DenseSample]) -> torch.Tensor:
    per_scene = [fnn.binary_cross_entropy_with_logits(feasibility_logits(model, s)[s.stable], s.F[s.stable])
                 for s in batch]
    return torch.stack(per_scene).mean()


def predicted_infeasible(logits) -> np.ndarray:
    return np.asarray(logits) >= THRESHOLD_LOGIT


def derive_repair(logits: np.ndarray, s: DenseSample):
    """Min exact-cost predicted-feasible admissible state (lowest index on ties) or NO_PREDICTED_FEASIBLE_STATE."""
    ok = ~predicted_infeasible(logits)
    if not ok.any():
        return NO_PREDICTED_FEASIBLE_STATE
    K = s.bits.numpy() @ s.cost
    cand = np.flatnonzero(ok)
    return int(s.states[cand[np.argmin(K[cand])]])


def state_metrics(logits: np.ndarray, s: DenseSample, mask=None) -> dict:
    """Per-scene state feasibility metrics on stable admissible states (optionally a further subset)."""
    keep = s.stable.numpy() & (np.ones(len(s.states), bool) if mask is None else np.asarray(mask))
    y, pred, lg = s.F.numpy()[keep], predicted_infeasible(logits)[keep], np.asarray(logits)[keep]
    nan = float("nan")
    if not len(y):
        return {"n_states": 0, "bce": nan, "recall_F0": nan, "recall_F1": nan, "balanced_accuracy": nan,
                "false_feasible": nan, "false_infeasible": nan}
    r0 = float(np.mean(~pred[y == 0])) if (y == 0).any() else nan
    r1 = float(np.mean(pred[y == 1])) if (y == 1).any() else nan
    bce = float(fnn.binary_cross_entropy_with_logits(torch.as_tensor(lg), torch.as_tensor(y)))
    return {"n_states": int(len(y)), "bce": bce, "recall_F0": r0, "recall_F1": r1,
            "balanced_accuracy": float(np.nanmean([r0, r1])), "false_feasible": 1 - r1, "false_infeasible": 1 - r0}
