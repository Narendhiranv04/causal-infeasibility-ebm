"""PoC-2 Stage 6: structured repair optimization and final verdict rules (plan2.md sections 12, 13, 22, 23).

Why the measured structure is order <= 2 and macro-centred (a property of the CURRENT oracle
factorization, not evidence that real manipulation is universally pairwise): G = sum_i c_i; a
relocation changes only its own entity's conflict; only SHIFT_TARGET changes the shared envelope;
and every static body's pose is chosen by one exclusive group of options. Hence geometric pair terms
can only couple a repositioning macro with other options, and static validity factorizes into
products of two pose indicators.

EXACT QUADRATIC CONSTRAINTS. Pose indicator of body i: ind(orig) = 1 - sum_{p in opts(i)} x_p,
ind(p) = x_p. For every checked static body pair (i, j) and pose pair (a, b) with penetration beyond
CONTACT_TOL_3D, add ind_i(a) ind_j(b) to Vpen(x). On M = 1 states Vpen(x) is the number of violating
pairs, so Vpen = 0 iff V = 1; this is verified exhaustively on every scene (no validity_edge proxy).
Cpen(x) = sum over pairs in one choice group of x_p x_q >= 1 iff M = 0. On M = 0 states Vpen can be
negative but is bounded: |Vpen| <= W = sum v(a, b) mag(a) mag(b), mag(orig) = max(1, n_i - 1), mag(p) = 1.

PENALTY MAGNITUDES (explicit bounds, no tuning). Cost K(x) = sum l_p x_p (unit on the dataset, so
K = |S|), K_max = sum l_p, B = K_max + 1.
  H = kappa G_2(x) + lambda_V Vpen(x) + lambda_M Cpen(x) + K(x)
  kappa = B / G_FEAS_TOL: any infeasible valid state has G >= min positive G >= G_FEAS_TOL (checked per
          scene), so kappa G >= B > K_max.
  lambda_V = B: an invalid consistent state has Vpen >= 1 and G_2 = G >= 0, so H >= B.
  lambda_M = B + kappa max(0, -L_G) + lambda_V W, with L_G = G_0 + sum min(0, alpha) + sum min(0, beta)
          the lowest value of G_2 over all binary x: every inconsistent state has H >= B.
  Feasible valid states have H = K <= K_max < B, so argmin H = S* (ties within ENERGY_TIE_TOL).
  G_2 uses the exact compatible-domain coefficients of order <= 2 (base resolution); beta of
  choice-constrained pairs (undefined, NaN) is set to 0 because Cpen excludes those states.

SOLVERS compared against the exact oracle S*: unary structured selection (Stage-3 S*_1, exact M, V);
exact pairwise QUBO enumeration; branch-on-global-macro (per macro branch the geometry is a conditional
unary sum, but M and V still require a combinatorial search over relocations); PoC-1 Hopfield,
unchanged, R in {1, 4, 16}, checked against the exact QUBO optimum.

PRE-REGISTERED FINAL VERDICT RULES (plan2.md section 23; four separate conclusions)
  A causal diagnosis: STRONG iff C* is stable under refinement AND causal-set discovery is
    nontrivial (some scene has C* != {B0}); otherwise WEAK (localization B0 may still be stable).
  B corrective recourse: STRONG iff S* is stable, every S* is valid and feasible, and at most
    TRIVIAL_MAX of repairable scenes are obvious one-to-one blocker removals (every tied repair
    relocates exactly the B0 entities once each, no repositioning, V not binding); otherwise WEAK.
  C pairwise hypothesis: INSUFFICIENT if the pairwise decision-failure rate > HIGHER_ORDER_MAX;
    SUPPORTED if R_pair >= R_PAIR_MIN in >= 2 families and decision-changing scenes involve >= 2
    non-choice mechanism kinds (envelope-changing geometry, substitutable geometry, static
    compatibility); otherwise NOT NEEDED.
  D Hopfield: DROP if its R = 16 exact-QUBO-optimum rate < 0.95 or valid-feasible rate < 0.99; RETAIN
    iff accurate AND faster than both exact enumeration and branch-on-macro; otherwise OPTIONAL.
"""

import math
from itertools import combinations

import numpy as np

from poc import cases as cs
from poc import energy as en
from poc import hopfield as hf
from poc.mj_scene import IDENTITY, Entity3D, GeomWorld
from poc.oracle import conflict_3d
from poc.types import Entity, EntityRole, InterventionKind
from poc2 import structure as st
from poc2.oracle import RepairTable, apply_option, choice_masks

TRIVIAL_MAX, HIGHER_ORDER_MAX, R_PAIR_MIN = 0.5, 0.05, 0.05
HOPFIELD_EXACT_MIN, HOPFIELD_FEASIBLE_MIN = 0.95, 0.99
BUDGETS = (1, 4, 16)
SHIFT = InterventionKind.SHIFT_TARGET


def _bodies(scene, options):
    """Static bodies with their pose variants and the option index selecting each non-original pose."""
    bodies = []
    for i, ent in enumerate(scene.entities):
        opts = [p for p, o in enumerate(options) if o.intervention.kind is InterventionKind.RELOCATE
                and o.intervention.entity_id == ent.eid]
        poses = [ent] + [apply_option(scene, options[p].intervention).entities[i] for p in opts]
        bodies.append((ent.entity, poses, opts))
    if scene.fixture:
        opts = [p for p, o in enumerate(options) if o.intervention.kind is SHIFT]
        fix = Entity(cs.FIXTURE, EntityRole.TARGET)
        poses = [Entity3D(fix, scene.fixture)] + [Entity3D(fix, apply_option(scene, options[p].intervention).fixture)
                                                    for p in opts]
        bodies.append((fix, poses, opts))
    return bodies


def validity_encoding(scene, options) -> dict:
    """Quadratic Vpen(x) = const + unary . x + sum_{p<q} pair[p, q] x_p x_q, and its bound W."""
    P, bodies = len(options), _bodies(scene, options)
    flat = [Entity3D(Entity(f"vb{b}.{k}", ent.role), pose.boxes) for b, (ent, poses, _) in enumerate(bodies)
            for k, pose in enumerate(poses)]
    index, n = {}, 0
    for b, (_, poses, _) in enumerate(bodies):
        for k in range(len(poses)):
            index[(b, k)], n = n, n + 1
    world = GeomWorld(tuple(flat), scene.moving)
    world.set_pose(cs.PARK, IDENTITY)
    const, unary, pair, W = 0.0, np.zeros(P), np.zeros((P, P)), 0.0
    for (bi, (ei, pi, oi)), (bj, (ej, pj, oj)) in combinations(enumerate(bodies), 2):
        if not cs.checked_pair(ei, ej):
            continue
        for a in range(len(pi)):
            for b in range(len(pj)):
                ga, gb = world.entity_geoms[index[(bi, a)]], world.entity_geoms[index[(bj, b)]]
                d = min(world.signed_distance(u, w) for u in ga for w in gb)
                if conflict_3d([d])[0] <= 0.0:
                    continue
                W += max(1, len(oi) - 1) ** (a == 0) * max(1, len(oj) - 1) ** (b == 0)
                lin_a = {oi[a - 1]: 1.0} if a else {**{p: -1.0 for p in oi}, None: 1.0}
                lin_b = {oj[b - 1]: 1.0} if b else {**{p: -1.0 for p in oj}, None: 1.0}
                for p, u in lin_a.items():
                    for q, w in lin_b.items():
                        if p is None and q is None:
                            const += u * w
                        elif p is None or q is None or p == q:
                            unary[q if p is None else p] += u * w
                        else:
                            pair[min(p, q), max(p, q)] += u * w
    return {"const": const, "unary": unary, "pair": pair, "W": W}


def poly_energy(const: float, unary: np.ndarray, pair: np.ndarray) -> np.ndarray:
    """const + unary . x + sum_{p<q} pair x_p x_q on every bitmask x."""
    X = en.binary_matrix(len(unary)).astype(float)
    return const + X @ unary + np.einsum("xp,pq,xq->x", X, np.triu(pair, 1), X)


def choice_penalty(P: int, masks: list[int]) -> np.ndarray:
    pair = np.zeros((P, P))
    for m in masks:
        members = [p for p in range(P) if m >> p & 1]
        for p, q in combinations(members, 2):
            pair[p, q] = 1.0
    return pair


def build_qubo(table: RepairTable, options, venc: dict) -> dict:
    """Pre-registered H = kappa G_2 + lambda_V Vpen + lambda_M Cpen + K as (Q, q, c) with explicit bounds."""
    P = len(options)
    a = st.compatible_mobius(table.G, table.M)
    size = en.popcount(len(a))
    G0 = float(a[0])
    alpha = np.array([a[1 << p] for p in range(P)])
    beta = np.zeros((P, P))
    for p, q in combinations(range(P), 2):
        v = a[(1 << p) | (1 << q)]
        beta[p, q] = 0.0 if math.isnan(v) else float(v)
    cost = np.array([o.cost for o in options], dtype=float)
    B = int(cost.sum()) + 1
    kappa, lam_v = B / st.G_FEAS_TOL, float(B)
    L_G = G0 + np.minimum(alpha, 0).sum() + np.minimum(beta, 0).sum()
    lam_m = B + kappa * max(0.0, -L_G) + lam_v * venc["W"]
    cpen = choice_penalty(P, choice_masks(options))
    c = kappa * G0 + lam_v * venc["const"]
    q = kappa * alpha + lam_v * venc["unary"] + cost
    J = kappa * beta + lam_v * np.triu(venc["pair"], 1) + lam_m * cpen
    Q = (J + J.T) / 2.0
    assert np.all(size[np.isfinite(a)] <= P)
    return {"Q": Q, "q": q, "c": c, "kappa": kappa, "lambda_V": lam_v, "lambda_M": lam_m, "B": B, "L_G": float(L_G),
            "W": venc["W"], "G2": poly_energy(G0, alpha, beta), "cpen": poly_energy(0.0, np.zeros(P), cpen),
            "vpen": poly_energy(venc["const"], venc["unary"], venc["pair"])}


def constraint_check(table: RepairTable, qubo: dict) -> dict:
    """Exhaustive: penalty lambda_V Vpen + lambda_M Cpen is 0 iff M = V = 1, and >= B otherwise."""
    pen = qubo["lambda_V"] * qubo["vpen"] + qubo["lambda_M"] * qubo["cpen"]
    ok = (table.M == 1) & (table.V == 1)
    consistent = table.M == 1
    return {"zero_iff_M_and_V": bool(np.all((np.abs(pen) <= 1e-9) == ok)),
            "vpen_counts_violations": bool(np.all((qubo["vpen"][consistent] > 0.5) == (table.V[consistent] == 0))),
            "min_penalty_when_violated": float(pen[~ok].min(initial=np.inf)), "B": qubo["B"]}


def branch_on_macro(table: RepairTable, options) -> dict:
    """Branch over 'no repositioning' and each repositioning option; within a branch the geometry is the
    conditional unary sum G(m) + sum_p [G(m + p) - G(m)], but M and V still need a search over subsets."""
    P = len(options)
    macros = [None] + [p for p, o in enumerate(options) if o.intervention.kind is SHIFT]
    reloc = [p for p, o in enumerate(options) if o.intervention.kind is not SHIFT]
    unit = all(o.cost == 1 for o in options)  # unit cost: size-ordered search may stop at the first size
    found, work, unary_exact = {}, 0, True
    for m in macros:
        base = 0 if m is None else 1 << m
        g0 = table.G[base]
        alpha = {p: table.G[base | 1 << p] - g0 for p in reloc}
        for size in range(len(reloc) + 1):
            hits = []
            for R in combinations(reloc, size):
                x = base | sum(1 << p for p in R)
                work += 1
                if table.M[x] == 0:
                    continue
                g_hat = g0 + sum(alpha[p] for p in R)
                unary_exact &= bool(abs(g_hat - table.G[x]) <= 1e-9)
                if table.V[x] == 1 and g_hat <= st.G_FEAS_TOL:
                    hits.append(x)
            for x in hits:
                found.setdefault(int(table.K[x]), []).append(x)
            if hits and unit:
                break
    best = frozenset(found[min(found)]) if found else frozenset()
    return {"decision": best, "subsets_examined": work, "of_2P": 2 ** P, "conditional_unary_exact": unary_exact}


def hopfield_runs(qubo: dict, seed) -> tuple[list[int], list[float], float]:
    """Unchanged PoC-1 Hopfield on the globally rescaled QUBO; returns per-restart masks, energies, runtime."""
    import time
    Q, q, c = qubo["Q"], qubo["q"], qubo["c"]
    net = hf.to_hopfield(*hf.rescale(Q, q, c, hf.normalising_factor(Q, q)))
    t0 = time.perf_counter()
    runs = hf.restarts(net, max(BUDGETS), np.random.default_rng(seed))
    runtime = (time.perf_counter() - t0) / len(runs)
    weights = 1 << np.arange(len(q))
    return [int(hf.bits(d.s) @ weights) for d in runs], [d.energy for d in runs], runtime


def validity_levels(table: RepairTable) -> dict:
    """'binding': the full tied S* changes when V is ignored; 'critical': no V-ignoring optimum is a valid
    exact repair; 'cost_raising': V raises the minimal repair cost."""
    exact = st.exact_decision(table)
    from poc.oracle import admissible_minimal_repairs
    no_v = admissible_minimal_repairs(np.where(table.M == 1, table.F, 1), (table.M == 1).astype(int),
                                      table.K, int(table.K.max()))
    cost = lambda xs: min(int(table.K[x]) for x in xs) if xs else None  # noqa: E731
    return {"binding": no_v != exact, "critical": not (no_v & exact), "cost_raising": cost(no_v) != cost(exact)}


def verdicts(ev: dict) -> dict:
    """The four pre-registered final PoC-2 conclusions (kept separate; see the module docstring)."""
    A = "STRONG" if ev["C_star_stable"] and ev["nontrivial_causal_sets"] else "WEAK"
    B = ("STRONG" if ev["S_star_stable"] and ev["all_S_star_valid_feasible"]
         and ev["trivial_repair_fraction"] <= TRIVIAL_MAX else "WEAK")
    if ev["pairwise_failure_rate"] > HIGHER_ORDER_MAX:
        C = "INSUFFICIENT"
    elif ev["families_with_R_pair_min"] >= 2 and ev["n_mechanism_kinds"] >= 2:
        C = "SUPPORTED"
    else:
        C = "NOT NEEDED"
    if ev["hopfield_exact_R16"] < HOPFIELD_EXACT_MIN or ev["hopfield_feasible_R16"] < HOPFIELD_FEASIBLE_MIN:
        D = "DROP"
    else:
        D = "RETAIN" if ev["hopfield_faster_than_exact_and_branch"] else "OPTIONAL"
    return {"A_causal_diagnosis": A, "B_corrective_recourse": B, "C_pairwise_hypothesis": C, "D_hopfield": D}
