"""Exact recourse oracle and interventional dependency labels.

Search: deterministic breadth-first search over the repair-state graph.
  node  = pose index of every object (articulation fixed)
  edge  = one candidate intervention executable in the node (each object moves at most once)
  goal  = F(s, a_target) == 0
  cost  = 1 per intervention, max depth 4 (frozen)
All tied minimum-cost sequences are enumerated (never just the first one).

Directed labels at the initial state s0 (controlled interventions):
  D_enable[p->q]  = exec(p,s0) and not exec(q,s0) and exec(q, do(p,s0))
  D_disable[p->q] = exec(p,s0) and exec(q,s0) and not exec(q, do(p,s0))
Symmetric compatibility labels are stored separately.

Sequence proofs: for an optimal sequence pi and steps j < k, step j is an interventional
prerequisite of step k iff I_k is NOT executable in do(pi[:k] without I_j) while it IS
executable in do(pi[:k]); likewise for the target action. Every such edge carries the
blocker identity, the conflict type (swept volume vs destination occupancy) and tau*.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

from .interventions import RecourseProblem
from .sweep import PEN_TOL

MAX_DEPTH = 4


def conflict_type(pb: RecourseProblem, iv, blocker_obj: str, blocker_pose: int) -> dict:
    """Why intervention `iv` is blocked by `blocker_obj` at `blocker_pose`."""
    pr = pb.tab[iv.id][blocker_obj][blocker_pose]
    occ_dist = pb.eng.static_distance(iv.obj, pb.pose_tuple(iv.obj, iv.pose_idx), blocker_obj,
                                      pb.pose_tuple(blocker_obj, blocker_pose))
    occupancy = occ_dist <= -PEN_TOL
    sweep_phases = {"approach", "grasp", "lift", "transfer"}
    gripper_hit = any(p.startswith("gripper") for p in pr.parts_in_conflict)
    sweep = bool(set(pr.phases_in_conflict) & sweep_phases) or gripper_hit or (
        not occupancy and pr.status != "clear")
    kind = "occupancy+sweep" if occupancy and sweep else ("occupancy" if occupancy else "sweep")
    return {"type": kind, "swept_volume": sweep, "destination_occupied": occupancy,
            "final_pose_distance": round(float(occ_dist), 5), **pr.to_dict()}


def bfs(pb: RecourseProblem, max_depth=MAX_DEPTH):
    s0 = pb.s0
    layers = [{s0: []}]                 # state -> list of (prev_state, iv_id)
    goal_depth = None
    if not pb.F(s0):
        goal_depth = 0
    visited = {s0: 0}
    for depth in range(1, max_depth + 1):
        if goal_depth is not None:
            break
        nxt = defaultdict(list)
        for s in layers[-1]:
            for iv in pb.ivs:
                if pb.executable(iv, s):
                    s2 = pb.apply(s, iv)
                    if s2 in visited and visited[s2] < depth:
                        continue
                    visited[s2] = depth
                    nxt[s2].append((s, iv.id))
        layers.append(dict(nxt))
        if any(not pb.F(s) for s in nxt):
            goal_depth = depth
    seqs = []
    if goal_depth:
        goals = [s for s in layers[goal_depth] if not pb.F(s)]

        def back(s, d):
            if d == 0:
                yield []
                return
            for prev, ivid in layers[d][s]:
                for path in back(prev, d - 1):
                    yield path + [ivid]

        for g in sorted(goals):
            seqs.extend(back(g, goal_depth))
        seqs = sorted(seqs)
    n_states = sum(len(l) for l in layers)
    return goal_depth, seqs, n_states


def all_solution_sets(pb: RecourseProblem, max_depth=MAX_DEPTH):
    """Irreducible repair sets (any executable ordering reaches the goal, no proper subset does)."""
    sols = []
    ivs = [iv for iv in pb.ivs if iv.admissible]
    for L in range(1, max_depth + 1):
        for combo in itertools.combinations(ivs, L):
            if len({iv.obj for iv in combo}) < L:
                continue
            if any(set(s) <= {iv.id for iv in combo} for s, _ in sols):
                continue
            for perm in itertools.permutations(combo):
                s = pb.s0
                ok = True
                for iv in perm:
                    if not pb.executable(iv, s):
                        ok = False
                        break
                    s = pb.apply(s, iv)
                if ok and not pb.F(s):
                    sols.append(({iv.id for iv in combo}, [iv.id for iv in perm]))
                    break
    return [{"interventions": sorted(s), "example_order": order, "length": len(s)} for s, order in sols]


def state_after(pb, ids, base=None):
    s = base or pb.s0
    for i in ids:
        s = pb.apply(s, pb.iv_by_id[i])
    return s


def directed_labels(pb: RecourseProblem):
    s0 = pb.s0
    ex0 = {iv.id: pb.executable(iv, s0) for iv in pb.ivs}
    enable, disable = [], []
    for p in pb.ivs:
        if not ex0[p.id]:
            continue
        s1 = pb.apply(s0, p)
        for q in pb.ivs:
            if q.obj == p.obj:
                continue
            after = pb.executable(q, s1)
            if not ex0[q.id] and after:
                why = conflict_type(pb, q, p.obj, 0)
                enable.append({"from": p.id, "to": q.id, "blocker": p.obj, "blocked_before": True,
                               "executable_after": True, "cause": why})
            elif ex0[q.id] and not after:
                why = conflict_type(pb, q, p.obj, p.pose_idx)
                disable.append({"from": p.id, "to": q.id, "blocker": p.obj, "cause": why})
    return enable, disable


def compatibility_labels(pb: RecourseProblem):
    out = {"same_object": [], "destination_overlap": [], "final_placement_collision": [], "support_incompatibility": []}
    for p, q in itertools.combinations(pb.ivs, 2):
        pair = sorted([p.id, q.id])
        if p.obj == q.obj:
            out["same_object"].append(pair)
            continue
        d = pb.eng.static_distance(p.obj, pb.pose_tuple(p.obj, p.pose_idx), q.obj, pb.pose_tuple(q.obj, q.pose_idx))
        if d <= -PEN_TOL:
            out["destination_overlap"].append(pair + [round(float(d), 4)])
        d1 = pb.eng.static_distance(p.obj, pb.pose_tuple(p.obj, p.pose_idx), q.obj, pb.pose_tuple(q.obj, 0))
        d2 = pb.eng.static_distance(q.obj, pb.pose_tuple(q.obj, q.pose_idx), p.obj, pb.pose_tuple(p.obj, 0))
        if min(d1, d2) <= -PEN_TOL:
            out["final_placement_collision"].append(pair + [round(float(min(d1, d2)), 4)])
        if p.placement.slot == q.placement.slot:
            out["support_incompatibility"].append(pair + [p.placement.slot])
    return out


def sequence_proof(pb: RecourseProblem, seq: list[str]) -> dict:
    """Interventional prerequisite edges inside one executable sequence (+ target)."""
    edges = []
    for k, ivk in enumerate(seq):
        iv = pb.iv_by_id[ivk]
        before = state_after(pb, seq[:k])
        assert pb.executable(iv, before), "sequence not executable"
        for j in range(k):
            cf = state_after(pb, [x for x in seq[:k] if x != seq[j]])
            ok_cf, bl = pb.exec_info(iv, cf)
            if not ok_cf:
                bj = pb.iv_by_id[seq[j]]
                if any(o == bj.obj for o, _ in bl):
                    edges.append({"from": seq[j], "to": ivk, "blocker": bj.obj,
                                  "proof": {f"{ivk} executable without {seq[j]}": False,
                                            f"{ivk} executable after {seq[j]}": True},
                                  "cause": conflict_type(pb, iv, bj.obj, cf[pb.obj_index(bj.obj)])})
    final = state_after(pb, seq)
    for j in range(len(seq)):
        cf = state_after(pb, [x for x in seq if x != seq[j]])
        F_cf, bl = pb.target_info(cf)
        if F_cf:
            bj = pb.iv_by_id[seq[j]]
            pr = pb.ttab[bj.obj][0]
            edges.append({"from": seq[j], "to": "TARGET", "blocker": bj.obj,
                          "proof": {f"TARGET feasible without {seq[j]}": False, "TARGET feasible after sequence": True},
                          "cause": {"type": "target_blocker" if pr.status != "clear" else "indirect", **pr.to_dict()}})
    return {"sequence": seq, "edges": edges, "final_F": pb.F(final)}


def solve(pb: RecourseProblem) -> dict:
    pb.track = set()
    s0 = pb.s0
    F0, blockers0 = pb.target_info(s0)
    depth, seqs, n_states = bfs(pb)
    no_recourse = bool(F0) and depth is None
    enable, disable = directed_labels(pb)
    compat = compatibility_labels(pb)
    proofs = [sequence_proof(pb, s) for s in seqs] if seqs else []
    alts = all_solution_sets(pb) if F0 else []
    return {
        "F_initial": int(F0),
        "direct_target_blockers": sorted({o for o, _ in blockers0}),
        "direct_target_blocker_details": {o: pr.to_dict() for o, pr in blockers0},
        "optimal_cost": depth if depth is not None else None,
        "sequence_length": depth if depth is not None else None,
        "optimal_sequences": seqs,
        "n_optimal_sequences": len(seqs),
        "no_recourse": no_recourse,
        "search": {"algorithm": "BFS (unit cost)", "max_depth": MAX_DEPTH, "states_expanded": n_states},
        "enable_edges": enable,
        "disable_edges": disable,
        "compatibility_edges": compat,
        "sequence_proofs": proofs,
        "irreducible_solutions": alts,
        "executable_initially": sorted(iv.id for iv in pb.ivs if pb.executable(iv, s0)),
        "target_clearance": {o: round(float(pb.ttab[o][0].min_dist), 5) for o in pb.obj_keys},
        "intervention_object": {iv.id: iv.obj for iv in pb.ivs},
        "consulted_ambiguities": pb.consulted_ambiguities(),
    }
