"""PoC-3 Stage 1: feature contract and one-scene energy plumbing (plan3.md section 48). No training.

Run from the repository root:

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s1_features.py

Writes PoC-3/out/s1/feature_contract.json and example_tensors.npz for the six PoC-2 Stage-1 make-space
cases and the three hand-built task regression cases (no dataset is generated). The untrained model is
initialised with an explicit Torch seed, so every number is reproducible.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from poc import energy as en
from poc2 import scenes, tasks
from poc3 import TRAINING_SEEDS, RunEnvironment
from poc3 import features as ft
from poc3 import model as md

OUT = Path(__file__).resolve().parents[1] / "out" / "s1"
SEED = TRAINING_SEEDS[0]


def cases() -> dict:
    out = {c.name: (c.scene, c.regions, c.options) for c in scenes.stage1_cases()}
    out.update({k: tasks.build(spec) for k, (spec, _, _) in tasks.regression_cases().items()})
    return out


def seeded(pairwise: bool) -> md.EnergyModel:
    torch.manual_seed(SEED)
    return md.EnergyModel(pairwise=pairwise).eval()


def check(model, scene, regions, options) -> tuple[dict, dict]:
    f = ft.extract(scene, regions, options)
    with torch.no_grad():
        q, Q = (t.double() for t in model.coefficients(f))
        E = md.state_energies(q, Q)
        rev = model.coefficients(ft.extract(replace(scene, entities=scene.entities[::-1]), regions, options))
        perm = np.random.default_rng(SEED).permutation(len(options))
        q2, Q2 = model.coefficients(ft.extract(scene, regions, tuple(options[p] for p in perm)))
    t = torch.as_tensor(perm)
    poc1 = en.qubo_energy(Q.numpy() / 2, q.numpy(), 0.0)
    record = {
        "N_entities": len(f.entities), "P": len(options), "n_states": len(E), "fixture": bool(scene.fixture),
        "n_no_entity_candidates": int((f.affected == ft.NO_ENTITY).sum()),
        "Q_symmetric_exact": bool(torch.equal(Q, Q.T)), "Q_zero_diagonal": bool((torch.diagonal(Q) == 0).all()),
        "E_empty_set": float(E[0]), "E_range": [float(E.min()), float(E.max())],
        "max_dev_entity_reorder": float(max((rev[0].double() - q).abs().max(), (rev[1].double() - Q).abs().max())),
        "max_dev_candidate_reorder": float(max((q2.double() - q[t]).abs().max(), (Q2.double() - Q[t][:, t]).abs().max())),
        "max_dev_vs_poc1_qubo_energy": float(np.abs(E.numpy() - poc1).max()),
    }
    arrays = {"entities": f.entities, "action": f.action, "moving": f.moving, "candidates": f.candidates,
              "affected": f.affected, "q": q.numpy(), "Q": Q.numpy(), "E": E.numpy()}
    return record, arrays


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pair, unary = seeded(True), seeded(False)
    records, arrays = {}, {}
    for name, (scene, regions, options) in cases().items():
        records[name], arr = check(pair, scene, regions, options)
        arrays.update({f"{name}/{k}": v for k, v in arr.items()})
    out = {
        "stage": 1, "torch_seed": SEED, "environment": RunEnvironment.capture().to_json(),
        "feature_contract": ft.contract(),
        "model": {"architecture": md.__doc__.strip(), "hidden": md.HIDDEN,
                  "n_parameters_pairwise": md.n_parameters(pair), "n_parameters_unary": md.n_parameters(unary),
                  "max_parameters": md.MAX_PARAMS, "trained": False},
        "leakage": "inputs are SceneFeatures only (entities, action, moving, candidates, affected); extraction "
                   "never calls the oracle; ids, names, family, seeds and costs do not enter (tests/test_features.py)",
        "examples": records,
        "sanity": {"all_Q_symmetric": all(r["Q_symmetric_exact"] for r in records.values()),
                   "all_E_empty_zero": all(r["E_empty_set"] == 0 for r in records.values()),
                   "max_dev_entity_reorder": max(r["max_dev_entity_reorder"] for r in records.values()),
                   "max_dev_candidate_reorder": max(r["max_dev_candidate_reorder"] for r in records.values()),
                   "max_dev_vs_poc1_qubo_energy": max(r["max_dev_vs_poc1_qubo_energy"] for r in records.values())},
    }
    (OUT / "feature_contract.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT / "example_tensors.npz", **arrays)
    summary = {k: v for k, v in out["model"].items() if k != "architecture"}
    print(json.dumps({"model": summary, "sanity": out["sanity"], "n_examples": len(records)}, indent=1))


if __name__ == "__main__":
    main()
