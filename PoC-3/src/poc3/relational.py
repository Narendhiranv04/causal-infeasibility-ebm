"""PoC-3 Stage 4.5 diagnostic: relational unary intervention energy (approved exception to the file table).

Hypothesis tested: the Stage-4 encoder loses geometry because entities are embedded independently and pooled
before they meet the action. Here every static MOVABLE / STRUCTURAL entity i is related to every one of the
8 existing action samples k BEFORE pooling. Built only from the frozen Stage-1 tensors (entities, action,
moving, candidates, affected); no oracle, sweep, distance or collision quantity is computed.

  R_k      rotation from the stored 6D columns (Gram-Schmidt; exact for the stored rotations)
  z_ik     [R_k^T (c_i - p_k), |R_k^T| h_i, c_moving_local, h_moving_local, role one-hot]   (15 values, L0 units)
  e_i      [mean_k, max_k] phi_rel(z_ik)            phi_rel: 2-layer MLP 15 -> 128 -> 64, GELU
  g        [mean_i, max_i] e_i                       (256 values)
  The target fixture is excluded from the relation pool (under the frozen oracle it enters exact V only).
  R1 static  g_p = g_0 for every candidate.
  R2 cf      g_p = g(s | do(I_p)) at the level of input geometry only: RELOCATE(i) replaces entity i's centre
             by the candidate's proposed centre (size / role kept); SHIFT_TARGET(d) shifts all 8 action
             positions by d (rotations and moving composite kept). Single-candidate, no subset enumeration.
  q_p = f([g_0, g_p, g_p - g_0, u_p, e_affected(p)])   f: 2-layer MLP 910 -> 128 -> 1; e_affected = 0 for
  SHIFT_TARGET (NO_ENTITY). Unary energy only: Q = 0, so the Stage-4 structured loss / inference apply as is.
R1 and R2 have identical modules and parameter counts; only the context computation differs.
"""

import torch
from torch import nn

from poc3 import model as md
from poc3.features import CANDIDATE_COLUMNS, ENTITY_COLUMNS, MOVING_COLUMNS

REL_HIDDEN, REL_EMBED = 128, 64
TOKEN = 3 + 3 + len(MOVING_COLUMNS) + 3
ENTITY_DIM = 2 * REL_EMBED          # [mean_k, max_k]
CONTEXT_DIM = 2 * ENTITY_DIM        # [mean_i, max_i]
VARIANTS = ("static", "cf")
PAIR_MODES = (None, "all", "shift")
PAIR_EMBED = 64
PAIR_ARCHITECTURE = """Stage-4.6B pair head (optional; the unary path above is unchanged):
  h_p = phi_pair(z_p)                      phi_pair: 2-layer MLP 910 -> 128 -> 64, GELU (z_p = unary input)
  z_pr = [h_p + h_r, |h_p - h_r|, h_p * h_r, g_0]   symmetric in (p, r)
  Q_pr = Q_rp = f_2(z_pr) for p < r, Q_pp = 0       f_2: 2-layer MLP 448 -> 128 -> 1, GELU
  pair = "all": every pair; pair = "shift": Q_pr kept only if candidate p or r is SHIFT_TARGET (candidate
  kind only, applied after prediction; identical parameters). E = q^T x + sum_{p<r} Q_pr x_p x_r + K."""
FIXTURE = ENTITY_COLUMNS.index("role_target_fixture")
SHIFT = CANDIDATE_COLUMNS.index("kind_shift_target")
DELTA = slice(CANDIDATE_COLUMNS.index("dx"), CANDIDATE_COLUMNS.index("dz") + 1)
PROPOSED = slice(CANDIDATE_COLUMNS.index("px"), CANDIDATE_COLUMNS.index("pz") + 1)


def rotation_from_6d(r6: torch.Tensor) -> torch.Tensor:
    """(..., 6) first two rotation columns [R00 R10 R20 R01 R11 R21] -> (..., 3, 3) rotation matrices."""
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True)
    return torch.stack([b1, b2, torch.linalg.cross(b1, b2, dim=-1)], dim=-1)


def relation_tokens(centres, halves, roles, pos, R, moving) -> torch.Tensor:
    """(..., N, K, 15) raw entity-action relations; pos may carry leading candidate dimensions."""
    d = centres[..., :, None, :] - pos[..., None, :, :]                  # (..., N, K, 3)
    rel = torch.einsum("kab,...nka->...nkb", R, d)                          # R_k^T (c_i - p_k)
    half = torch.einsum("kab,...na->...nkb", R.abs(), halves)               # |R_k^T| h_i
    shape = rel.shape[:-1]
    return torch.cat([rel, half.expand(*shape, -1), moving.expand(*shape, -1),
                      roles[..., :, None, :].expand(*shape, -1)], dim=-1)


class RelationalEnergy(nn.Module):
    """Relational energy; forward returns (q, Q) like the Stage-1 EnergyModel (Q = 0 unless a pair head)."""

    def __init__(self, variant: str, pair: str | None = None):
        super().__init__()
        if variant not in VARIANTS or pair not in PAIR_MODES:
            raise ValueError(f"variant must be one of {VARIANTS} and pair one of {PAIR_MODES}")
        self.variant, self.pair, self.pairwise = variant, pair, pair is not None
        z_dim = 3 * CONTEXT_DIM + len(CANDIDATE_COLUMNS) + ENTITY_DIM
        self.phi_rel = md.mlp(TOKEN, REL_EMBED, REL_HIDDEN)
        self.score = md.mlp(z_dim, 1, md.HIDDEN)
        if self.pairwise:  # absent for the unary model, so Stage-4.5 checkpoints still load strictly
            self.phi_pair = md.mlp(z_dim, PAIR_EMBED, md.HIDDEN)
            self.f_2 = md.mlp(3 * PAIR_EMBED + CONTEXT_DIM, 1, md.HIDDEN)

    def embed(self, centres, halves, roles, pos, R, moving) -> torch.Tensor:
        h = self.phi_rel(relation_tokens(centres, halves, roles, pos, R, moving))
        return torch.cat([h.mean(dim=-2), h.max(dim=-2).values], dim=-1)    # pool over action samples

    @staticmethod
    def pool(e: torch.Tensor) -> torch.Tensor:
        return torch.cat([e.mean(dim=-2), e.max(dim=-2).values], dim=-1)    # pool over entities

    def contexts(self, entities, action, moving, candidates, affected):
        """(g0, g_p for every candidate, static entity embeddings, affected pool index or -1)."""
        keep = entities[:, FIXTURE] == 0
        if not keep.any():
            raise ValueError("the relation pool needs at least one movable or structural entity")
        ents = entities[keep]
        c, h, role = ents[:, 0:3], ents[:, 3:6], ents[:, 6:9]
        pos, R = action[:, :3], rotation_from_6d(action[:, 3:])
        e0 = self.embed(c, h, role, pos, R, moving)
        g0 = self.pool(e0)
        slot = torch.cumsum(keep.long(), 0) - 1                             # entity index -> pool row
        safe = affected.clamp(min=0)
        aff = torch.where((affected >= 0) & keep[safe], slot[safe], torch.full_like(affected, -1))
        P = len(candidates)
        g = g0.expand(P, -1)
        if self.variant == "cf" and P:
            shift = candidates[:, SHIFT] == 1
            g = g.clone()
            if shift.any():                                                 # shifted action samples
                e_s = self.embed(c, h, role, pos + candidates[shift][:, None, DELTA], R, moving)
                g[shift] = self.pool(e_s)
            reloc = (~shift) & (aff >= 0)
            if reloc.any():                                                 # one entity at its proposed centre
                rows = aff[reloc]
                e_new = self.embed(candidates[reloc][:, None, PROPOSED], h[rows][:, None], role[rows][:, None],
                                   pos, R, moving)[:, 0]
                mask = torch.nn.functional.one_hot(rows, len(ents)).bool()[..., None]
                g[reloc] = self.pool(torch.where(mask, e_new[:, None, :], e0[None]))
        return g0, g, e0, aff

    def unary_inputs(self, entities, action, moving, candidates, affected):
        """(z_p for every candidate, g_0): the unchanged REL-CF / REL-STATIC unary input."""
        g0, g, e0, aff = self.contexts(entities, action, moving, candidates, affected)
        padded = torch.cat([e0, e0.new_zeros(1, ENTITY_DIM)])
        e_aff = padded[torch.where(aff < 0, len(e0), aff)]
        z = torch.cat([g0.expand(len(candidates), -1), g, g - g0, candidates, e_aff], dim=1)
        return z, g0

    @staticmethod
    def pair_descriptor(h: torch.Tensor, g0: torch.Tensor, p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return torch.cat([h[p] + h[r], (h[p] - h[r]).abs(), h[p] * h[r], g0.expand(len(p), -1)], dim=1)

    def forward(self, entities, action, moving, candidates, affected) -> tuple[torch.Tensor, torch.Tensor]:
        z, g0 = self.unary_inputs(entities, action, moving, candidates, affected)
        q = self.score(z).squeeze(-1)
        P = len(candidates)
        Q = q.new_zeros(P, P)
        if self.pairwise and P > 1:
            p, r = torch.triu_indices(P, P, offset=1)
            pair_score = self.f_2(self.pair_descriptor(self.phi_pair(z), g0, p, r)).squeeze(-1)
            if self.pair == "shift":
                shift = candidates[:, SHIFT] == 1
                pair_score = pair_score * (shift[p] | shift[r]).to(pair_score.dtype)
            Q = Q.index_put((p, r), pair_score).index_put((r, p), pair_score)
        return q, Q


def load(state_dict: dict, variant: str, pair: str | None = None) -> RelationalEnergy:
    model = RelationalEnergy(variant, pair)
    model.load_state_dict(state_dict)
    return model.eval()
