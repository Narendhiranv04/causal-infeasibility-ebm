"""Stage 3: classical binary Hopfield network as a candidate QUBO solver.

Mapping. Put the QUBO H_Q(x) = x^T Q x + q^T x + c in canonical form (Q
symmetric, zero diagonal; x_p^2 = x_p folds the diagonal into q). With
x = (s + 1) / 2, s in {-1, +1}^P:

    x^T Q x = 1/4 s^T Q s + 1/2 (Q 1)^T s + 1/4 1^T Q 1
    q^T x   = 1/2 q^T s + 1/2 q^T 1

The Hopfield energy E(s) = -1/2 s^T W s - b^T s (W symmetric, zero diagonal)
therefore equals H_Q(x) + C for

    W = -Q / 2,    b = -(Q 1 + q) / 2,    C = -(1/4 1^T Q 1 + 1/2 1^T q + c).

Dynamics. Local field h_i = sum_j W_ij s_j + b_i. Flipping s_i changes E by
exactly 2 s_i h_i, so the asynchronous rule "flip iff s_i h_i < 0" (keep s_i
when h_i = 0) never increases E. A sweep without flips is a converged fixed
point: a one-flip local minimum, NOT a certified global QUBO optimum. Only
exhaustive enumeration certifies the global optimum in this PoC.
"""

from dataclasses import dataclass

import numpy as np

MAX_SWEEPS = 1000  # safety bound on full asynchronous sweeps per descent


def canonical_qubo(Q, q, c: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Same energy on {0,1}^P with Q symmetric and zero-diagonal."""
    Q = np.asarray(Q, dtype=float)
    Qs = (Q + Q.T) / 2.0
    return Qs - np.diag(np.diag(Qs)), np.asarray(q, dtype=float) + np.diag(Qs), float(c)


def rescale(Q, q, c: float, eta: float) -> tuple[np.ndarray, np.ndarray, float]:
    """H' = eta H for eta > 0: one global factor, all coefficient ratios and the argmin kept."""
    if not eta > 0:
        raise ValueError(f"eta must be > 0, got {eta}")
    return eta * np.asarray(Q, dtype=float), eta * np.asarray(q, dtype=float), eta * float(c)


def normalising_factor(Q, q) -> float:
    """eta = 1 / max |coefficient| of (Q, q); 1 for a constant energy."""
    scale = max(np.max(np.abs(Q), initial=0.0), np.max(np.abs(q), initial=0.0))
    return 1.0 / scale if scale > 0 else 1.0


@dataclass(frozen=True)
class Hopfield:
    W: np.ndarray  # symmetric, zero diagonal
    b: np.ndarray  # bias; threshold theta = -b
    C: float       # E(s) = H_Q(x) + C


def to_hopfield(Q, q, c: float) -> Hopfield:
    Q, q, c = canonical_qubo(Q, q, c)
    ones = np.ones(len(q))
    return Hopfield(W=-Q / 2.0, b=-(Q @ ones + q) / 2.0, C=-(ones @ Q @ ones / 4.0 + ones @ q / 2.0 + c))


def spins(x) -> np.ndarray:
    return 2 * np.asarray(x, dtype=np.int8) - 1


def bits(s) -> np.ndarray:
    return (np.asarray(s) + 1) // 2


def energy(net: Hopfield, s) -> np.ndarray:
    """E(s) for one state (P,) or a batch (n, P)."""
    s = np.asarray(s, dtype=float)
    return -0.5 * np.einsum("...i,ij,...j->...", s, net.W, s) - s @ net.b


def local_field(net: Hopfield, s, i: int) -> float:
    return float(net.W[i] @ s + net.b[i])


def flip_delta(net: Hopfield, s, i: int) -> float:
    """E(s with s_i flipped) - E(s) = 2 s_i h_i (zero diagonal)."""
    return 2.0 * s[i] * local_field(net, s, i)


def is_fixed_point(net: Hopfield, s) -> bool:
    """No single flip lowers E: s_i h_i >= 0 for every i."""
    s = np.asarray(s, dtype=float)
    return bool(np.all(s * (net.W @ s + net.b) >= 0.0))


@dataclass(frozen=True)
class Descent:
    s: np.ndarray       # final spin state
    energy: float       # final Hopfield energy E(s)
    sweeps: int
    flips: int
    converged: bool     # last sweep made no flip (fixed point), not global optimality
    trace: tuple[float, ...] = ()  # E after every single-neuron update when recorded


def descend(net: Hopfield, s0, rng: np.random.Generator | None = None, order: str = "random",
            max_sweeps: int = MAX_SWEEPS, record: bool = False) -> Descent:
    """Asynchronous single-neuron descent from s0.

    order="random": a fresh seeded permutation per sweep (needs rng);
    order="fixed": neurons 0..P-1 every sweep (deterministic).
    """
    if order not in ("random", "fixed"):
        raise ValueError(f"unknown order {order!r}")
    s = np.array(s0, dtype=float)
    E, trace, flips, P = float(energy(net, s)), [], 0, len(s)
    for sweep in range(1, max_sweeps + 1):
        flipped = 0
        for i in (rng.permutation(P) if order == "random" else range(P)):
            h = local_field(net, s, i)
            if s[i] * h < 0.0:
                E += 2.0 * s[i] * h
                s[i] = -s[i]
                flipped += 1
            if record:
                trace.append(E)
        flips += flipped
        if flipped == 0:
            return Descent(s.astype(np.int8), float(energy(net, s)), sweep, flips, True, tuple(trace))
    return Descent(s.astype(np.int8), float(energy(net, s)), max_sweeps, flips, False, tuple(trace))


def restarts(net: Hopfield, R: int, rng: np.random.Generator, order: str = "random") -> list[Descent]:
    """R independent descents from seeded uniformly random initial spin states."""
    P = len(net.b)
    return [descend(net, rng.choice(np.array([-1, 1], dtype=np.int8), size=P), rng, order) for _ in range(R)]


def best(descents: list[Descent]) -> Descent:
    """Lowest-energy descent (first one on exact ties)."""
    return min(descents, key=lambda d: d.energy)
