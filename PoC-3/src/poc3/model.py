"""PoC-3 Stage 1: learned intervention energy for one scene (plan3.md sections 8, 15-19).

  h_i = phi_e(e_i);  h_a = phi_a([A, moving]);  h_s = phi_s([mean_i h_i, max_i h_i, h_a])      (DeepSets)
  h_p = phi_I([h_s, h_affected(p), u_p])         h_affected = 0 when the candidate has no entity
  q_p = f_u(h_p)
  Q_pr = f_2([h_p + h_r, |h_p - h_r|, h_p * h_r, h_s]) for p < r, mirrored: Q = Q^T, diag(Q) = 0
  E_geom(x) = q^T x + sum_{p<r} Q_pr x_p x_r;  E(x) = E_geom(x) + K(x) with the EXACT cost K (never an input)

Every block is a 2-layer MLP (hidden 128, GELU). No attention, GNN or transformer. The pair input is
symmetric in (p, r), so Q does not depend on candidate order; mean / max pooling makes the model invariant
to entity order. The unary model (pairwise=False) has Q = 0 exactly. E(0) = 0 for every scene.
No training here.
"""

import numpy as np
import torch
from torch import nn

from poc import energy as en
from poc3.features import ACTION_COLUMNS, CANDIDATE_COLUMNS, ENTITY_COLUMNS, K_A, MOVING_COLUMNS, SceneFeatures

HIDDEN = 128
MAX_PARAMS = 500_000  # plan3.md section 19


def mlp(n_in: int, n_out: int, hidden: int = HIDDEN) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


def as_tensors(f: SceneFeatures, dtype: torch.dtype = torch.float32) -> dict:
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype)  # noqa: E731
    return {"entities": t(f.entities), "action": t(f.action), "moving": t(f.moving), "candidates": t(f.candidates),
            "affected": torch.as_tensor(f.affected, dtype=torch.long)}


class EnergyModel(nn.Module):
    """Scene-conditioned unary (+ optional symmetric pairwise) intervention energy."""

    def __init__(self, pairwise: bool = True, hidden: int = HIDDEN):
        super().__init__()
        self.pairwise, self.hidden = pairwise, hidden
        self.phi_e = mlp(len(ENTITY_COLUMNS), hidden, hidden)
        self.phi_a = mlp(K_A * len(ACTION_COLUMNS) + len(MOVING_COLUMNS), hidden, hidden)
        self.phi_s = mlp(3 * hidden, hidden, hidden)
        self.phi_i = mlp(2 * hidden + len(CANDIDATE_COLUMNS), hidden, hidden)
        self.f_u = mlp(hidden, 1, hidden)
        self.f_2 = mlp(4 * hidden, 1, hidden) if pairwise else None

    def scene_embedding(self, entities: torch.Tensor, action: torch.Tensor, moving: torch.Tensor):
        h_e = self.phi_e(entities)
        h_a = self.phi_a(torch.cat([action.reshape(-1), moving]))
        h_s = self.phi_s(torch.cat([h_e.mean(dim=0), h_e.max(dim=0).values, h_a]))
        return h_e, h_s

    def candidate_embeddings(self, h_e, h_s, candidates: torch.Tensor, affected: torch.Tensor) -> torch.Tensor:
        padded = torch.cat([h_e, h_e.new_zeros(1, self.hidden)])            # row N = "no entity"
        h_aff = padded[torch.where(affected < 0, len(h_e), affected)]
        return self.phi_i(torch.cat([h_s.expand(len(candidates), -1), h_aff, candidates], dim=1))

    def forward(self, entities, action, moving, candidates, affected) -> tuple[torch.Tensor, torch.Tensor]:
        """(q, Q): unary coefficients (P,) and symmetric zero-diagonal pair coefficients (P, P)."""
        if len(entities) == 0:
            raise ValueError("a scene has at least one static entity")
        h_e, h_s = self.scene_embedding(entities, action, moving)
        h = self.candidate_embeddings(h_e, h_s, candidates, affected)
        q = self.f_u(h).squeeze(-1)
        P = len(candidates)
        Q = h.new_zeros(P, P)
        if self.pairwise and P > 1:
            p, r = torch.triu_indices(P, P, offset=1)
            z = torch.cat([h[p] + h[r], (h[p] - h[r]).abs(), h[p] * h[r], h_s.expand(len(p), -1)], dim=1)
            beta = self.f_2(z).squeeze(-1)
            Q = Q.index_put((p, r), beta).index_put((r, p), beta)
        return q, Q

    def coefficients(self, f: SceneFeatures) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = next(self.parameters()).dtype
        return self(**as_tensors(f, dtype))


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bit_matrix(P: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """(2^P, P) states, row x has bit p = option p (PoC-1 / PoC-2 bitmask convention)."""
    return torch.as_tensor(en.binary_matrix(P), dtype=dtype)


def state_energies(q: torch.Tensor, Q: torch.Tensor, cost: torch.Tensor | None = None) -> torch.Tensor:
    """Exact energy of all 2^P intervention subsets: q^T x + sum_{p<r} Q_pr x_p x_r (+ K(x) if cost given)."""
    X = bit_matrix(len(q), q.dtype)
    E = X @ q + 0.5 * ((X @ Q) * X).sum(dim=1)
    return E if cost is None else E + X @ cost.to(q.dtype)
