"""PoC-2 repair-space oracle (plan2.md sections 8.3-8.8).

All geometry, signed distances, envelope sweeps, conflict semantics (d, p, c,
G, F), static validity V and the repair objective J* = B F + K come from the
frozen PoC-1 package `poc`. This module owns only the new repair space:

  * choice groups and choice consistency M(S) (several destinations per object,
    mutually exclusive repositioning alternatives);
  * the original causal diagnosis B0 and all minimal causal sets C*, computed by
    diagnostic exclusion only (never with executable repair candidates);
  * the exhaustive table over all 2^P subsets: M, then V, F, G, K for M = 1;
  * the exact minimal valid corrective sets S* (M = V = 1, F = 0, minimal K).

Rows with M(S) = 0 carry no geometry: F = V = -1 and G = nan (never invented).
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from poc import cases as cs
from poc.envelope import ENVELOPE_STEP, sweep
from poc.mj_scene import IDENTITY, Entity3D, GeomWorld
from poc.oracle import Assessment, admissible_minimal_repairs, assess, feasibility_excluding, is_valid
from poc.types import Entity, EntityRole, InterventionKind
from poc2.types import CauseResult, RepairGroup, RepairOption, RepairOracleResult, RepairStatus

RELOCATE, SHIFT = InterventionKind.RELOCATE, InterventionKind.SHIFT_TARGET


def choice_groups(options: tuple[RepairOption, ...]) -> tuple[RepairGroup, ...]:
    """One group per object with >= 2 destinations and one for >= 2 repositioning alternatives.

    Every option belongs to exactly one key (its object, or the target), so no
    option can sit in two mutually exclusive groups.
    """
    keys: dict[str, list[str]] = {}
    for o in options:
        key = f"reposition:{o.intervention.entity_id}" if o.intervention.kind is SHIFT else f"object:{o.intervention.entity_id}"
        keys.setdefault(key, []).append(o.option_id)
    return tuple(RepairGroup(k, tuple(ids)) for k, ids in keys.items() if len(ids) >= 2)


def choice_masks(options: tuple[RepairOption, ...]) -> list[int]:
    index = {o.option_id: p for p, o in enumerate(options)}
    return [sum(1 << index[i] for i in g.option_ids) for g in choice_groups(options)]


def choice_consistent(x: int, masks: list[int]) -> bool:
    """M(S) = 1 iff at most one option of every group is selected."""
    return all((x & m).bit_count() <= 1 for m in masks)


def causes(a: Assessment) -> CauseResult:
    """B0 and all inclusion-minimal C within B0 with F^{do(ignore C)} = 0 (diagnostic only)."""
    if a.F == 0:
        return CauseResult(0, frozenset(), frozenset())
    B0 = sorted(a.blockers)
    minimal: list[frozenset[str]] = []
    for k in range(1, len(B0) + 1):
        for C in map(frozenset, combinations(B0, k)):
            if not any(m <= C for m in minimal) and feasibility_excluding(a, C) == 0:
                minimal.append(C)
    return CauseResult(1, frozenset(B0), frozenset(minimal))


@dataclass(frozen=True)
class RepairTable:
    """Exhaustive oracle table, rows indexed by bitmask x (bit p = option p)."""
    entity_ids: tuple[str, ...]
    option_ids: tuple[str, ...]
    M: np.ndarray       # (2^P,) choice consistency
    V: np.ndarray       # (2^P,) static validity, -1 where M = 0
    F: np.ndarray       # (2^P,) target-action infeasibility, -1 where M = 0
    G: np.ndarray       # (2^P,) graded conflict, nan where M = 0
    C: np.ndarray       # (2^P, N) conflict vectors, nan where M = 0
    K: np.ndarray       # (2^P,) repair cost
    original: Assessment

    def subset(self, x: int) -> frozenset[str]:
        return frozenset(i for p, i in enumerate(self.option_ids) if x >> p & 1)


def _variant_world(scene, options):
    """All pose variants: entity i -> [original, relocated by each of its options]; motion per repositioning."""
    variants = [[e] for e in scene.entities]
    where = {}
    ids = [e.eid for e in scene.entities]
    for p, o in enumerate(options):
        if o.intervention.kind is RELOCATE:
            i = ids.index(o.intervention.entity_id)
            variants[i].append(cs.apply_intervention(scene, o.intervention).entities[i])
            where[p] = (i, len(variants[i]) - 1)
    motions = {None: scene} | {p: cs.apply_intervention(scene, o.intervention)
                               for p, o in enumerate(options) if o.intervention.kind is SHIFT}
    flat = [Entity3D(Entity(f"{e.eid}.v{k}", e.entity.role), e.boxes) for vs in variants for k, e in enumerate(vs)]
    col = {}
    for i, vs in enumerate(variants):
        for k in range(len(vs)):
            col[(i, k)] = len(col)
    return where, motions, flat, col


def repair_table(scene: cs.Scene3D, options: tuple[RepairOption, ...], step: float = ENVELOPE_STEP) -> RepairTable:
    """Exact table over all 2^P subsets. Entity i's distance depends only on its own pose and the
    envelope, so each pose variant is swept once (checked against direct_row in the tests)."""
    P, masks = len(options), choice_masks(options)
    where, motions, flat, col = _variant_world(scene, options)
    sweeps = {m: sweep(GeomWorld(tuple(flat), scene.moving), sc.motion, scene.moving, step) for m, sc in motions.items()}
    fixtures = {m: Entity3D(Entity(f"{cs.FIXTURE}.{m}", EntityRole.TARGET), sc.fixture) for m, sc in motions.items()
                if sc.fixture}
    static = tuple(flat) + tuple(fixtures.values())
    fix_col = {m: len(flat) + k for k, m in enumerate(fixtures)}
    world = GeomWorld(static, scene.moving)
    world.set_pose(cs.PARK, IDENTITY)
    pair: dict[tuple[int, int], float] = {}
    n, N = 2 ** P, len(scene.entities)
    M, V, F, K = (np.zeros(n, dtype=int) for _ in range(4))
    G, C = np.full(n, np.nan), np.full((n, N), np.nan)
    original = None
    for x in range(n):
        S = [p for p in range(P) if x >> p & 1]
        K[x] = sum(options[p].cost for p in S)
        if not choice_consistent(x, masks):
            V[x] = F[x] = -1
            continue
        M[x] = 1
        m = next((p for p in S if p in motions), None)
        k_of = dict(where[p] for p in S if p in where)
        cols = [col[(i, k_of.get(i, 0))] for i in range(N)]
        sw = sweeps[m]
        a = assess([e.eid for e in scene.entities], sw.d_min[cols], sw.d_start[cols], sw.d_goal[cols])
        original = a if x == 0 else original
        F[x], G[x], C[x] = a.F, a.G, a.c
        bodies = cols + ([fix_col[m]] if m in fix_col else [])
        for u, w in combinations(bodies, 2):
            if (u, w) not in pair and cs.checked_pair(static[u].entity, static[w].entity):
                pair[(u, w)] = cs._pair_distance(world, world.entity_geoms[u], world.entity_geoms[w])
        V[x] = is_valid([pair[(u, w)] for u, w in combinations(bodies, 2) if (u, w) in pair])
    return RepairTable(tuple(e.eid for e in scene.entities), tuple(o.option_id for o in options),
                       M, V, F, G, C, K, original)


def pair_effect(table: RepairTable, p: int, q: int) -> float | None:
    """beta^G_pq = G(pq) - G(p) - G(q) + G({}) (plan2 9.2); None unless all four subsets are choice-consistent."""
    xs = (0, 1 << p, 1 << q, (1 << p) | (1 << q))
    if any(table.M[x] == 0 for x in xs):
        return None
    return float(table.G[xs[3]] - table.G[xs[1]] - table.G[xs[2]] + table.G[xs[0]])


def validity_edge(table: RepairTable, p: int, q: int) -> bool:
    """Static compatibility edge: p and q are each valid alone but jointly invalid (choice-consistent pair)."""
    x = (1 << p) | (1 << q)
    return bool(table.M[x] == 1 and table.V[1 << p] == 1 and table.V[1 << q] == 1 and table.V[x] == 0)


def direct_row(scene: cs.Scene3D, options: tuple[RepairOption, ...], x: int,
               step: float = ENVELOPE_STEP) -> tuple[Assessment, int]:
    """Reference evaluation of one choice-consistent subset: do(S), re-run the action, check V."""
    if not choice_consistent(x, choice_masks(options)):
        raise ValueError("contradictory selections have no geometric do(S)")
    repaired = cs.do(scene, [o.intervention for p, o in enumerate(options) if x >> p & 1])
    return cs.evaluate(repaired, step)[0], cs.validity(repaired)[0]


def repair_result(table: RepairTable) -> RepairOracleResult:
    """All tied minimal valid corrective sets: M = V = 1, F = 0, minimal K (J* from PoC-1)."""
    admissible = ((table.M == 1) & (table.V == 1)).astype(int)
    F = np.where(table.M == 1, table.F, 1)
    k_max = int(table.K.max())
    best = admissible_minimal_repairs(F, admissible, table.K, k_max)
    counts = dict(P=len(table.option_ids), n_choice_invalid=int((table.M == 0).sum()),
                  n_static_invalid=int(((table.M == 1) & (table.V == 0)).sum()))
    if not best:
        return RepairOracleResult(RepairStatus.NO_RECOURSE_IN_CATALOGUE, frozenset(), None, **counts)
    status = RepairStatus.FEASIBLE if table.F[0] == 0 else RepairStatus.REPAIRED
    cost = int(table.K[next(iter(best))])
    return RepairOracleResult(status, frozenset(table.subset(x) for x in best), cost, **counts)
