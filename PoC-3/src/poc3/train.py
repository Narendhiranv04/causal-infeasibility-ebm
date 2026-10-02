"""PoC-3 Stage 3: independent non-energy candidate baseline (plan3.md sections 20, 32, 33A, 50).

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


def train(train_set: list[Sample], val_set: list[Sample], seed: int, max_epochs: int = MAX_EPOCHS,
          patience: int = PATIENCE, batch: int = BATCH_SCENES, threads: int = 1) -> dict:
    """Train one seed; returns the best-validation state dict and the full loss history."""
    set_determinism(seed, threads)
    model = md.EnergyModel(pairwise=False)
    assert not model.pairwise
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order_rng = np.random.default_rng(seed)
    stopper, history, best_state = EarlyStopping(patience), {"train_bce": [], "val_bce": []}, None
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        order, total = order_rng.permutation(len(train_set)), 0.0
        for k in range(0, len(order), batch):
            chunk = [train_set[i] for i in order[k:k + batch]]
            opt.zero_grad()
            loss = scene_balanced_bce(model, chunk)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(chunk)
        model.eval()
        with torch.no_grad():
            val = float(scene_balanced_bce(model, val_set))
        history["train_bce"].append(total / len(train_set))
        history["val_bce"].append(val)
        if stopper.update(epoch, val):
            best_state = copy.deepcopy(model.state_dict())
        if stopper.stop:
            break
    return {"state_dict": best_state, "best_epoch": stopper.best_epoch, "best_val_bce": stopper.best,
            "epochs_run": len(history["val_bce"]), "history": history, "runtime_s": time.perf_counter() - t0,
            "seed": seed, "threads": threads}


def load_model(state_dict: dict) -> md.EnergyModel:
    model = md.EnergyModel(pairwise=False)
    model.load_state_dict(state_dict)
    return model.eval()


def predict(model: md.EnergyModel, s: Sample) -> tuple[np.ndarray, int]:
    """(pi, x_hat): independent probabilities and the thresholded bitmask, with no post-processing."""
    with torch.no_grad():
        pi = torch.sigmoid(logits(model, s)).numpy().astype(float)
    return pi, int(sum(1 << p for p in range(len(pi)) if pi[p] >= THRESHOLD))
