"""Stage 2: pseudo-Boolean interaction-order analysis and exact QUBO.

Tables are indexed by bitmask x (bit p is x_p), exactly as the Stage 1
exhaustive enumeration. For a measured set function G the Möbius coefficients

    a(T) = sum_{U subset T} (-1)^{|T|-|U|} G(U),      G(S) = sum_{T subset S} a(T)

give G_0 = a({}), alpha_p = a({p}), beta_pq = G(pq) - G(p) - G(q) + G({}),
gamma_pqr = a({p,q,r}). The order-k approximation G_k keeps |T| <= k: it is
computed from the table (never fitted) and is exact on every |S| <= k.

QUBO (plan.md section 10, Part B):

    H(x) = kappa * G_2(x) + lambda * K(x) = x^T Q x + q^T x + c

with Q symmetric and zero-diagonal (x_p^2 = x_p is folded into q).
"""

import math

import numpy as np

ENERGY_TIE_TOL = 1e-6  # [repair actions] energies closer than this are tied optima
SUBSTITUTABLE = "substitutable"  # either intervention alone removes the same conflict
PHYSICAL = "physical"            # one intervention changes the geometric effect of the other


def n_vars(table) -> int:
    P = int(round(math.log2(len(table))))
    if 2 ** P != len(table):
        raise ValueError(f"table length {len(table)} is not 2^P")
    return P


def _transform(table, sign: float) -> np.ndarray:
    P = n_vars(table)
    a = np.array(table, dtype=float).reshape((2,) * P)
    for axis in range(P):
        view = np.moveaxis(a, axis, 0)
        view[1] += sign * view[0]
    return a.reshape(-1)


def mobius(table) -> np.ndarray:
    """Möbius coefficients a(T) of a set-function table G(x)."""
    return _transform(table, -1.0)


def zeta(coeffs) -> np.ndarray:
    """Inverse of mobius: G(x) = sum_{T subset x} a(T)."""
    return _transform(coeffs, +1.0)


def popcount(n_entries: int) -> np.ndarray:
    return np.array([bin(x).count("1") for x in range(n_entries)])


def truncate(coeffs, k: int) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    return np.where(popcount(len(coeffs)) <= k, coeffs, 0.0)


def approximate(table, k: int) -> np.ndarray:
    """Order-k Möbius truncation G_k evaluated on every subset."""
    return zeta(truncate(mobius(table), k))


def degree(coeffs, tol: float) -> int:
    """Largest |T| with |a(T)| > tol (0 for a constant function)."""
    sizes = popcount(len(coeffs))[np.abs(np.asarray(coeffs)) > tol]
    return int(sizes.max(initial=0))


def interaction_order(table, tol: float) -> int:
    """Smallest k whose truncation reproduces the table within tol everywhere."""
    table = np.asarray(table, dtype=float)
    return next(k for k in range(n_vars(table) + 1) if np.max(np.abs(table - approximate(table, k))) <= tol)


def coefficient(coeffs, *variables: int) -> float:
    return float(coeffs[sum(1 << p for p in set(variables))])


def error_metrics(exact, approx) -> dict:
    err = np.asarray(approx, dtype=float) - np.asarray(exact, dtype=float)
    return {"max_abs": float(np.max(np.abs(err))), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2)))}


def argmin_sets(values, tie_tol: float = ENERGY_TIE_TOL) -> frozenset[int]:
    """All bitmasks x whose value is within tie_tol of the minimum."""
    values = np.asarray(values, dtype=float)
    return frozenset(int(x) for x in np.flatnonzero(values <= values.min() + tie_tol))


def binary_matrix(P: int) -> np.ndarray:
    """X[x, p] = x_p for every bitmask x."""
    return (np.arange(2 ** P)[:, None] >> np.arange(P)[None, :]) & 1


def qubo_weights(B: int, contact_tol: float) -> tuple[float, float]:
    """kappa, lambda for H = kappa G_2 + lambda K.

    lambda = 1: energy is measured in primitive repair actions.
    kappa = B / contact_tol: any infeasible S has G >= max_i c_i > contact_tol,
    so kappa G(S) > B = K_max + 1 > K(S') for every S'. Residual conflict beyond
    the contact tolerance therefore always costs more than the longest repair,
    which is the same lexicographic ordering J* = B F + K encodes. Not tuned.
    """
    return B / contact_tol, 1.0


def qubo(coeffs, lengths, kappa: float, lam: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(Q, q, c) of H = kappa G_2 + lambda K from Möbius coefficients."""
    P = len(lengths)
    if 2 ** P != len(coeffs):
        raise ValueError("coefficient table does not match the number of interventions")
    Q = np.zeros((P, P))
    for p in range(P):
        for r in range(p + 1, P):
            Q[p, r] = Q[r, p] = kappa * coefficient(coeffs, p, r) / 2.0
    q = np.array([kappa * coefficient(coeffs, p) + lam * lengths[p] for p in range(P)])
    return Q, q, kappa * coefficient(coeffs)


def qubo_energy(Q: np.ndarray, q: np.ndarray, c: float) -> np.ndarray:
    """x^T Q x + q^T x + c on every bitmask x."""
    X = binary_matrix(len(q)).astype(float)
    return np.einsum("xp,pr,xr->x", X, Q, X) + X @ q + c


def polynomial_energy(coeffs, lengths, kappa: float, lam: float) -> np.ndarray:
    """kappa G_2(x) + lambda K(x) evaluated directly from the polynomial."""
    K = binary_matrix(len(lengths)) @ np.asarray(lengths, dtype=float)
    return kappa * zeta(truncate(coeffs, 2)) + lam * K


def pair_sources(G_table, C_table, tol: float) -> list[dict]:
    """Attribute every pairwise term |beta_pq| > tol to the entities that produce it.

    With G = ||c||_1 the Möbius transform is additive over entities, so
    beta_pq = sum_i beta_pq^(i). Entity i's share is SUBSTITUTABLE when p and q
    each reduce c_i alone and together remove less than the sum of their
    individual effects (beta^(i) > 0); otherwise it is PHYSICAL: one
    intervention changes what the other does to entity i's geometry.
    """
    C = np.asarray(C_table, dtype=float)
    a = mobius(G_table)
    per_entity = np.stack([mobius(C[:, i]) for i in range(C.shape[1])], axis=1)
    P, out = n_vars(G_table), []
    for p in range(P):
        for r in range(p + 1, P):
            beta = coefficient(a, p, r)
            if abs(beta) <= tol:
                continue
            x_pr, shares = (1 << p) | (1 << r), []
            for i in np.flatnonzero(np.abs(per_entity[x_pr]) > tol):
                c0, cp, cr = C[0, i], C[1 << p, i], C[1 << r, i]
                both_reduce = cp < c0 - tol and cr < c0 - tol
                kind = SUBSTITUTABLE if per_entity[x_pr, i] > 0 and both_reduce else PHYSICAL
                shares.append({"entity": int(i), "beta_i": float(per_entity[x_pr, i]), "kind": kind})
            out.append({"p": p, "q": r, "beta": beta, "entities": shares})
    return out


def classify(reports: list[dict]) -> tuple[str, str]:
    """PoC interaction-order class from per-case reports (plan.md section 10).

    Each report needs: order (int), physical_pairs (int), unary_repairs_match (bool).
    C: some case needs order >= 3. B: some case needs a physical pairwise term
    without which the unary model returns the wrong minimal repair. A otherwise;
    substitutability-only pairwise terms are repair-choice structure and do not
    by themselves justify a QUBO.
    """
    if any(r["order"] >= 3 for r in reports):
        return "C", "at least one case requires third- or higher-order terms"
    physical = [r for r in reports if r["order"] == 2 and r["physical_pairs"] and not r["unary_repairs_match"]]
    if physical:
        return "B", f"{len(physical)} case(s) need physical pairwise terms to recover the minimal repair"
    return "A", "unary reproduces every minimal repair; no physical pairwise term is needed"
