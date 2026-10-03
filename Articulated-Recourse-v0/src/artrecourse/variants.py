"""Structural variant definitions, evaluated on the oracle output (never on generation intent).

A scene is accepted as variant V only if the geometry, through the oracle, actually
produced V's structure. Definitions follow the dataset specification; "meaningful" repair
dependency edges are the interventional prerequisite edges inside optimal sequences
(oracle.sequence_proof), each backed by an explicit before/after executability proof.
"""

from __future__ import annotations

import networkx as nx

NEAR_MISS = 0.03   # C0: some object passes within 3 cm of the target's swept envelope


def repair_edges(res: dict, seq_index: int | None = None) -> list[dict]:
    proofs = res["sequence_proofs"] if seq_index is None else [res["sequence_proofs"][seq_index]]
    out, seen = [], set()
    for p in proofs:
        for e in p["edges"]:
            if e["to"] == "TARGET":
                continue
            k = (e["from"], e["to"])
            if k not in seen:
                seen.add(k)
                out.append(e)
    return out


def _seq_graph(proof):
    g = nx.DiGraph()
    g.add_nodes_from(proof["sequence"] + ["TARGET"])
    for e in proof["edges"]:
        g.add_edge(e["from"], e["to"], **e)
    return g


def check_variant(variant: str, res: dict, scene: dict) -> tuple[bool, list[str]]:
    """Returns (satisfied, list of unmet conditions)."""
    why = []
    F0, L = res["F_initial"], res["optimal_cost"]
    D = res["direct_target_blockers"]
    proofs = res["sequence_proofs"]
    n_obj = len(scene["objects"])
    n_cand = len(scene["candidate_interventions"])

    def need(cond, msg):
        if not cond:
            why.append(msg)

    if variant == "C0":
        need(F0 == 0, "target must be feasible")
        near = [o for o, d in res["target_clearance"].items() if d < NEAR_MISS]
        need(len(near) >= 1, f"no object within {NEAR_MISS} m of the target sweep")
        return not why, why
    need(F0 == 1, "target must be initially infeasible")
    if variant == "C1":
        need(res["no_recourse"], "a repair exists within depth 4")
        return not why, why
    need(L is not None, "no recourse within depth 4")
    if L is None:
        return False, why
    edges_any = repair_edges(res)
    if variant == "V0":
        need(L == 1, f"optimal length {L} != 1")
        need(len(D) == 1, f"{len(D)} direct blockers != 1")
        need(not edges_any, "repair dependency edge present")
    elif variant == "V1":
        need(len(D) >= 2, f"{len(D)} direct blockers < 2")
        need(L >= 2, f"optimal length {L} < 2")
        need(not edges_any, "repairs are not independent (dependency edge present)")
        need(any(all(s in res["executable_initially"] for s in p["sequence"]) for p in proofs),
             "no optimal sequence whose repairs are all executable in s0")
    elif variant == "V2":
        need(L >= 2, f"optimal length {L} < 2")
        need(any(e["cause"]["destination_occupied"] for e in edges_any), "no occupancy dependency edge")
    elif variant == "V3":
        need(L >= 2, f"optimal length {L} < 2")
        need(any(e["cause"]["swept_volume"] and not e["cause"]["destination_occupied"] for e in edges_any),
             "no pure swept-volume dependency edge (destination free, trajectory blocked)")
    elif variant == "V4":
        need(L >= 3, f"optimal length {L} < 3")
        ok = False
        for i, p in enumerate(proofs):
            es = repair_edges(res, i)
            g = nx.Graph()
            g.add_edges_from((e["from"], e["to"]) for e in es)
            comps = [c for c in nx.connected_components(g) if len(c) >= 2]
            objs = {e["blocker"] for e in es} | {res["intervention_object"][iv] for c in comps for iv in c}
            if len(es) >= 2 and len(comps) >= 2 and len(objs) >= 4:
                ok = True
        need(ok, "no optimal sequence with >= 2 distinct dependency chains over >= 4 objects")
    elif variant == "V5":
        need(L == 3, f"optimal length {L} != 3")
        ok = False
        for p in proofs:
            s = p["sequence"]
            if len(s) != 3:
                continue
            g = _seq_graph(p)
            chain = g.has_edge(s[0], s[1]) and g.has_edge(s[1], s[2]) and g.has_edge(s[2], "TARGET")
            sweeps = sum(g.edges[a, b]["cause"]["swept_volume"] for a, b in ((s[0], s[1]), (s[1], s[2]))
                         if g.has_edge(a, b))
            if chain and sweeps >= 2:
                ok = True
        need(ok, "no optimal sequence forming a C->B->A->target chain with >= 2 swept-volume edges")
    elif variant == "V6":
        sols = res["irreducible_solutions"]
        lens = {s["length"] for s in sols}
        need(len(sols) >= 2, f"{len(sols)} irreducible repair plans < 2")
        need(len(lens) >= 2, "alternative repair plans do not differ in length/cost")
    elif variant == "V7":
        need(7 <= n_obj <= 9, f"{n_obj} objects not in [7, 9]")
        need(12 <= n_cand <= 20, f"{n_cand} candidate interventions not in [12, 20]")
        need(len(D) >= 2, f"{len(D)} direct blockers < 2")
        need(L in (3, 4), f"optimal length {L} not in {{3, 4}}")
        need(len(edges_any) >= 3, f"{len(edges_any)} meaningful enablement edges < 3")
        need(sum(e["cause"]["swept_volume"] for e in edges_any) >= 2, "< 2 swept-volume enablement edges")
        comp = res["compatibility_edges"]
        n_comp = len(comp["destination_overlap"]) + len(comp["final_placement_collision"]) + len(comp["support_incompatibility"])
        need(n_comp >= 1, "no compatibility / destination-conflict edge")
        need(not all(e["cause"]["destination_occupied"] and not e["cause"]["swept_volume"] for e in edges_any),
             "all dependencies are occupancy-only")
    else:
        raise KeyError(variant)
    return not why, why


def classify_all(res: dict, scene: dict) -> dict:
    return {v: check_variant(v, res, scene)[0] for v in ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "C0", "C1")}
