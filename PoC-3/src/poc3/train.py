"""PoC-3 training: Stage 3 independent baseline (sections 20, 32, 33A, 50) and Stage 4 exact structured set
likelihood for the unary / pairwise learned energies (sections 21, 22, 33B-C, 36, 51; see the Stage-4 block).

  Model       the unchanged Stage-1 EnergyModel(pairwise=False); its unary outputs q_p are independent
              candidate logits, pi_p = sigmoid(q_p). Q, structured energy inference, M / V projection and
              repair-set optimization are never used.
  Inputs      the unchanged Stage-1 features.extract on each frozen Stage-2 scene spec (in-memory only).
  Target      soft marginal y_p = (1 / |S*|) sum_{x in S*} x_p over ALL stored tied optima.
  Loss        BCEWithLogits against y, averaged over the scene's own P candidates, then equally over
              scenes (no padding exists: scenes are processed at their own P). No focal loss, positive
              weighting or reweighting of any kind.
  Optimiser   AdamW, lr 1e-3, weight decay 1e-5, at most 200 epochs, early-stopping patience 20 on the
              validation scene-balanced BCE only (the test split is never seen here). Minibatches of
              BATCH_SCENES = 32 scenes in a seeded per-epoch permutation.
  Determinism Python / NumPy / Torch seeded from the training seed; torch.use_deterministic_algorithms(True);
              cuDNN deterministic, benchmark off; CPU with a fixed thread count.
  Inference   x_hat_p = 1[pi_p >= 0.5], used exactly as predicted (no repair of M, V or cardinality).
No learned standardization beyond the fixed L0 scaling of the feature contract (no normalization.json).
"""

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as fnn

from poc2 import dataset as pd
from poc3 import TRAINING_SEEDS
from poc3 import dataset as ds
from poc3 import features as ft
from poc3 import model as md
from poc3.metrics import soft_target

LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE = 1e-3, 1e-5, 200, 20
SEEDS = TRAINING_SEEDS
BATCH_SCENES = 32
THRESHOLD = 0.5
HYPERPARAMETERS = {"optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
                   "patience": PATIENCE, "batch_scenes": BATCH_SCENES, "seeds": list(SEEDS), "threshold": THRESHOLD,
                   "loss": "BCEWithLogits on soft marginals, mean over P then over scenes", "selection": "val BCE"}


@dataclass(frozen=True)
class Sample:
    inputs: dict              # Stage-1 model tensors only
    y: torch.Tensor           # (P,) soft marginal target
    optimal: frozenset[int]   # oracle S* as state indices (evaluation only)
    meta: dict                # scene_id / family / intent (evaluation only, never an input)


def make_sample(scene, regions, options, optimal, meta: dict | None = None) -> Sample:
    f = ft.extract(scene, regions, options)
    y = torch.as_tensor(soft_target(optimal, len(options)), dtype=torch.float32)
    return Sample(md.as_tensors(f), y, frozenset(optimal), meta or {})


def samples_from_dataset(records: list[dict], arrays: dict, rows) -> list[Sample]:
    out = []
    for j in rows:
        r, P = records[j], int(arrays["P"][j])
        meta = {k: r[k] for k in ("scene_id", "family", "intent", "split")}
        out.append(make_sample(*ds.build(pd.spec_from_json(r["spec"])), ds.unpack_states(arrays["optimal"][j], P), meta))
    return out


def split_rows(records: list[dict], split: str) -> list[int]:
    return [j for j, r in enumerate(records) if r["split"] == split]


def set_determinism(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logits(model: md.EnergyModel, s: Sample) -> torch.Tensor:
    q, _ = model(**s.inputs)  # Q of the unary model is identically zero and is ignored
    return q


def scene_balanced_bce(model: md.EnergyModel, batch: list[Sample]) -> torch.Tensor:
    return torch.stack([fnn.binary_cross_entropy_with_logits(logits(model, s), s.y) for s in batch]).mean()


class EarlyStopping:
    """Best = strictly lowest validation loss; stop after `patience` epochs without improvement."""

    def __init__(self, patience: int):
        self.patience, self.best, self.best_epoch, self.bad = patience, float("inf"), -1, 0

    def update(self, epoch: int, value: float) -> bool:
        if value < self.best:
            self.best, self.best_epoch, self.bad = value, epoch, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self) -> bool:
        return self.bad >= self.patience


def train(train_set: list, val_set: list, seed: int, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
          batch: int = BATCH_SCENES, threads: int = 1, pairwise: bool = False, loss_fn=None, name: str = "bce",
          make_model=None) -> dict:
    """Train one seed; returns the best-validation state dict and the full loss history.

    Defaults are the frozen Stage-3 baseline (unary model, scene-balanced BCE). Stage 4 passes
    pairwise / loss_fn = structured_loss / name = "set_nll"; everything else is shared. make_model (Stage-4.5
    diagnostic hook) builds another (q, Q) model at the same point, after seeding; None keeps EnergyModel."""
    loss_fn = loss_fn or scene_balanced_bce
    set_determinism(seed, threads)
    model = make_model() if make_model is not None else md.EnergyModel(pairwise=pairwise)
    assert model.pairwise == pairwise
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order_rng = np.random.default_rng(seed)
    stopper, history, best_state = EarlyStopping(patience), {f"train_{name}": [], f"val_{name}": []}, None
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        order, total = order_rng.permutation(len(train_set)), 0.0
        for k in range(0, len(order), batch):
            chunk = [train_set[i] for i in order[k:k + batch]]
            opt.zero_grad()
            loss = loss_fn(model, chunk)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(chunk)
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model, val_set))
        history[f"train_{name}"].append(total / len(train_set))
        history[f"val_{name}"].append(val)
        if stopper.update(epoch, val):
            best_state = copy.deepcopy(model.state_dict())
        if stopper.stop:
            break
    return {"state_dict": best_state, "best_epoch": stopper.best_epoch, f"best_val_{name}": stopper.best,
            "epochs_run": len(history[f"val_{name}"]), "history": history, "runtime_s": time.perf_counter() - t0,
            "seed": seed, "threads": threads, "pairwise": pairwise}


def load_model(state_dict: dict, pairwise: bool = False) -> md.EnergyModel:
    model = md.EnergyModel(pairwise=pairwise)
    model.load_state_dict(state_dict)
    return model.eval()


def predict(model: md.EnergyModel, s: Sample) -> tuple[np.ndarray, int]:
    """(pi, x_hat): independent probabilities and the thresholded bitmask, with no post-processing."""
    with torch.no_grad():
        pi = torch.sigmoid(logits(model, s)).numpy().astype(float)
    return pi, int(sum(1 << p for p in range(len(pi)) if pi[p] >= THRESHOLD))


# ------------------------------------------------------------ Stage 4: exact structured set likelihood
# E(x) = q^T x + sum_{p<r} Q_pr x_p x_r + K(x) on the exact admissible domain A = {M = 1, V = 1} (hard mask,
# from the stored Stage-2 bitset); K uses the stored candidate costs and is never an input. Loss per scene:
#   L_set = (1 / |S*|) sum_{x in S*} E(x) / T + logsumexp_{y in A} (-E(y) / T),  T_train = 1,
# averaged equally over scenes. Top-1 = argmin over A, ties -> lowest state index; learned minimum set
# = {x in A : E(x) <= E_min + ENERGY_TIE_TOL}. Temperature: one scalar per checkpoint on the fixed grid,
# minimising validation set NLL (ties -> smallest grid index).
ENERGY_TIE_TOL = 1e-6
T_GRID = np.logspace(np.log10(0.05), np.log10(20.0), 241)
GATE_SCENES_PER_FAMILY, GATE_EPOCHS, GATE_BATCH, GATE_MIN_HITS = 5, 400, 20, 19


@dataclass(frozen=True)
class StructuredSample:
    inputs: dict              # Stage-1 model tensors only
    cost: torch.Tensor        # (P,) exact stored candidate costs (float64), never an input
    states: np.ndarray        # (A,) admissible state indices, ascending
    bits: torch.Tensor        # (A, P) float64 bits of the admissible states
    target: torch.Tensor      # (A,) bool, state in S*
    optimal: frozenset[int]   # complete tied oracle S*
    meta: dict                # evaluation only


def make_structured_sample(scene, regions, options, admissible, optimal, cost, meta: dict | None = None):
    P, states = len(options), np.array(sorted(admissible), dtype=np.int64)
    if not optimal or not set(optimal) <= set(admissible):
        raise ValueError("S* must be a non-empty subset of the admissible domain")
    bits = torch.as_tensor((states[:, None] >> np.arange(P)[None, :]) & 1, dtype=torch.float64)
    target = torch.as_tensor(np.isin(states, sorted(optimal)))
    return StructuredSample(md.as_tensors(ft.extract(scene, regions, options)),
                            torch.as_tensor(np.asarray(cost, dtype=float)), states, bits, target, frozenset(optimal),
                            meta or {})


def structured_samples_from_dataset(records: list[dict], arrays: dict, rows) -> list[StructuredSample]:
    out = []
    for j in rows:
        r, P = records[j], int(arrays["P"][j])
        meta = {k: r[k] for k in ("scene_id", "family", "intent", "split")}
        out.append(make_structured_sample(*ds.build(pd.spec_from_json(r["spec"])),
                                          ds.unpack_states(arrays["admissible"][j], P),
                                          ds.unpack_states(arrays["optimal"][j], P), arrays["cost"][j][:P], meta))
    return out


def admissible_energies(q: torch.Tensor, Q: torch.Tensor, s: StructuredSample) -> torch.Tensor:
    """E(x) for every admissible x, in float64; the symmetric Q counts each pair p < r once."""
    X = s.bits
    return X @ q.double() + 0.5 * ((X @ Q.double()) * X).sum(dim=1) + X @ s.cost


def set_nll(E: torch.Tensor, target: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return E[target].mean() / T + torch.logsumexp(-E / T, dim=0)


def structured_loss(model: md.EnergyModel, batch: list[StructuredSample]) -> torch.Tensor:
    return torch.stack([set_nll(admissible_energies(*model(**s.inputs), s), s.target) for s in batch]).mean()


def energies(model: md.EnergyModel, s: StructuredSample) -> np.ndarray:
    with torch.no_grad():
        return admissible_energies(*model(**s.inputs), s).numpy()


def exact_inference(E: np.ndarray, s: StructuredSample) -> tuple[int, frozenset[int]]:
    """(top-1 state: exact argmin, lowest state index on ties; learned minimum set within ENERGY_TIE_TOL)."""
    top = int(s.states[np.flatnonzero(E == E.min())[0]])
    return top, frozenset(int(x) for x in s.states[E <= E.min() + ENERGY_TIE_TOL])


def nll_curve(E: np.ndarray, target: np.ndarray, temps) -> np.ndarray:
    """Exact set NLL of one scene at each temperature (stable logsumexp, float64)."""
    T = np.atleast_1d(np.asarray(temps, dtype=float))
    Z = -E[None, :] / T[:, None]
    m = Z.max(axis=1, keepdims=True)
    return E[target].mean() / T + m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))


def calibrate_temperature(val_energies: list[np.ndarray], val_targets: list[np.ndarray]) -> dict:
    """Validation-only scalar temperature on the fixed grid (lowest mean set NLL; ties -> smallest index)."""
    curve = np.mean([nll_curve(E, t, T_GRID) for E, t in zip(val_energies, val_targets)], axis=0)
    i = int(np.flatnonzero(curve == curve.min())[0])
    t1 = float(np.mean([nll_curve(E, t, 1.0)[0] for E, t in zip(val_energies, val_targets)]))
    return {"T": float(T_GRID[i]), "index": i, "at_boundary": i in (0, len(T_GRID) - 1),
            "val_nll_T1": t1, "val_nll_cal": float(curve[i])}


def gate_rows(records: list[dict]) -> list[int]:
    """First 5 repairable TRAIN scenes of each family (fixed row order): the 20-scene overfit gate."""
    out = []
    for fam in ds.FAMILIES:
        rows = [j for j, r in enumerate(records) if r["split"] == "train" and r["family"] == fam
                and r["intent"] == "repairable"]
        out += rows[:GATE_SCENES_PER_FAMILY]
    return out


def overfit_gate(samples: list[StructuredSample], seed: int = 7, epochs: int = GATE_EPOCHS) -> dict:
    """Pairwise model fit and evaluated (exact inference) on the same scenes; pass iff >= 19 / 20 hits."""
    run = train(samples, samples, seed, max_epochs=epochs, patience=epochs, batch=GATE_BATCH, pairwise=True,
                loss_fn=structured_loss, name="set_nll")
    model = load_model(run["state_dict"], pairwise=True)
    hits = [exact_inference(energies(model, s), s)[0] in s.optimal for s in samples]
    return {"hits": int(sum(hits)), "n": len(samples), "passed": sum(hits) >= GATE_MIN_HITS * len(samples) / 20,
            "best_epoch": run["best_epoch"], "best_train_set_nll": run["best_val_set_nll"],
            "runtime_s": run["runtime_s"]}
