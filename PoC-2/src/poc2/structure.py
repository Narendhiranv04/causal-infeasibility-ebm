"""PoC-2 Stage 3: compatible-domain interaction structure and pairwise necessity (plan2.md section 19).

PRE-REGISTERED RULES (fixed in code and tests before any aggregate Stage-3 statistic was computed)

  Domain        Coefficients and reconstructions use only choice-consistent subsets (M = 1). The
                consistent domain is downward closed, so the exact inclusion-exclusion coefficient
                a(T) = sum_{U subset T} (-1)^{|T|-|U|} G(U) of a consistent T uses only consistent U.
                Inconsistent subsets stay NaN: no geometry is invented for them.
  Significance  a(T) is significant iff its sign agrees at base (2 mm) and fine (1 mm) envelope
                resolution and min(|a_b|, |a_f|) > max(SIG_FACTOR |a_b - a_f|, COEF_FLOOR), with
                SIG_FACTOR = 10 and COEF_FLOOR = 1e-6 m (PoC-1 energy.significant, unchanged).
  Reconstruction tolerance  eps_repr = max(10 max_{S: M=1} |G_base(S) - G_fine(S)|, COEF_FLOOR);
                the representation order is the smallest k with max_{S: M=1} |G(S) - G_k(S)| <= eps_repr.
  Order-k decision  S*_k = all subsets minimising J_k(S) = B 1[G_k(S) > G_FEAS_TOL] + K(S) over the
                admissible set {M(S) = 1, V(S) = 1}, using the EXACT M, V and unit K at every order;
                G_FEAS_TOL = COEF_FLOOR. Only the approximation of geometric G varies with k.
                "S*_k correct" means set equality with the exact S* (all tied repairs).
  Sources       choice edge: p, q in one choice group (M({p,q}) = 0); static-compatibility edge:
                V(p) = V(q) = 1, V(pq) = 0 with M = 1; geometric edge: significant beta^G = a({p,q}).
                A significant beta^G > 0 with alpha_p < 0 and alpha_q < 0 is SUBSTITUTABLE (overlapping
                benefit); any other significant beta^G is PHYSICAL (conditional / envelope-changing).
                M and V edges are never reported as geometric beta^G.
"""

from itertools import combinations

import numpy as np

from poc import energy as en
from poc.oracle import admissible_minimal_repairs
from poc2.oracle import RepairTable, validity_edge

COEF_FLOOR = 1e-6          # [m] coefficient / feasibility resolution (as PoC-1 Stage 5)
SIG_FACTOR = en.SIG_FACTOR  # 10: base-vs-fine change multiplier
G_FEAS_TOL = COEF_FLOOR    # [m] a surrogate G_k above this means "still infeasible"
REPR_FACTOR = 10.0         # eps_repr multiplier on the measured base-vs-fine G variation
ORDERS = (1, 2, 3)
SUBSTITUTABLE, PHYSICAL = "substitutable", "physical"


def compatible_mobius(G: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Exact coefficients a(T) on the choice-consistent domain (NaN elsewhere)."""
    return en.mobius(np.where(M == 1, G, np.nan))


def reconstruct(coeffs: np.ndarray, k: int) -> np.ndarray:
    """G_k(S) = sum of a(T) over T subset S with |T| <= k; finite exactly on the consistent domain."""
    return en.zeta(en.truncate(coeffs, k))


def representation_errors(G: np.ndarray, M: np.ndarray, G_hat: np.ndarray) -> dict:
    ok = M == 1
    return en.error_metrics(G[ok], G_hat[ok])


def eps_repr(G_base: np.ndarray, G_fine: np.ndarray, M: np.ndarray) -> float:
    ok = M == 1
    return max(REPR_FACTOR * float(np.max(np.abs(G_base[ok] - G_fine[ok]))), COEF_FLOOR)


def representation_order(G: np.ndarray, M: np.ndarray, coeffs: np.ndarray, eps: float) -> int:
    ok = M == 1
    for k in range(int(en.popcount(len(G))[ok].max()) + 1):
        if np.max(np.abs(G[ok] - reconstruct(coeffs, k)[ok])) <= eps:
            return k
    raise AssertionError("the full expansion reproduces G exactly on the consistent domain")


def significant(coeffs_base: np.ndarray, coeffs_fine: np.ndarray) -> np.ndarray:
    """Pre-registered resolution-aware significance (False wherever a coefficient is undefined)."""
    finite = np.isfinite(coeffs_base) & np.isfinite(coeffs_fine)
    b, f = np.where(finite, coeffs_base, 0.0), np.where(finite, coeffs_fine, 0.0)
    return finite & en.significant(b, f, COEF_FLOOR, SIG_FACTOR)


def order_decision(table: RepairTable, G_hat: np.ndarray) -> frozenset[int]:
    """S*_k with the exact M, V and K; only the geometric G is approximated."""
    admissible = ((table.M == 1) & (table.V == 1)).astype(int)
    F_hat = np.where(table.M == 1, np.nan_to_num(G_hat, nan=np.inf) > G_FEAS_TOL, 1).astype(int)
    return admissible_minimal_repairs(F_hat, admissible, table.K, int(table.K.max()))


def exact_decision(table: RepairTable) -> frozenset[int]:
    admissible = ((table.M == 1) & (table.V == 1)).astype(int)
    return admissible_minimal_repairs(np.where(table.M == 1, table.F, 1), admissible, table.K, int(table.K.max()))


def interaction_edges(table: RepairTable, coeffs: np.ndarray, sig: np.ndarray) -> dict:
    """Pairwise structure split by source. Geometric edges need significance; M / V edges never become beta^G."""
    P = len(table.option_ids)
    choice, validity, geometric = [], [], []
    for p, q in combinations(range(P), 2):
        x = (1 << p) | (1 << q)
        if table.M[x] == 0:
            choice.append((p, q))
            continue
        if validity_edge(table, p, q):
            validity.append((p, q))
        if sig[x]:
            beta, ap, aq = coeffs[x], coeffs[1 << p], coeffs[1 << q]
            kind = SUBSTITUTABLE if beta > 0 and ap < 0 and aq < 0 else PHYSICAL
            geometric.append((p, q, float(beta), kind))
    return {"choice": choice, "validity": validity, "geometric": geometric}


def density(n_edges: int, P: int) -> float:
    return n_edges / (P * (P - 1) / 2) if P > 1 else 0.0
