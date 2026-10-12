"""
Shared training / evaluation utilities for both experiments.

A sample ("item") is a dict:
    snaps       list of PyG Data (length T for temporal models, 1 for static)
    y           FloatTensor (N,) labels, may contain NaN for unknown labels
    train_mask  BoolTensor (N,) nodes that contribute to the loss
    eval_mask   BoolTensor (N,) nodes that are scored
Static models only see the last snapshot.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch_geometric.data import Batch

from models.temporal_gatv2 import TemporalGATv2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing F1 — choose it on validation, apply it to test."""
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = p >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "base_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Forward / predict
# ---------------------------------------------------------------------------

def forward_batch(model, items: List[dict]) -> torch.Tensor:
    """Run a model on a batch of items; returns concatenated node probabilities."""
    if isinstance(model, TemporalGATv2):
        T = len(items[0]["snaps"])
        return model([Batch.from_data_list([it["snaps"][t] for it in items]) for t in range(T)])
    b = Batch.from_data_list([it["snaps"][-1] for it in items])
    return model(b.x, b.edge_index, b.edge_attr)


@torch.no_grad()
def predict(model, items: List[dict], batch_size: int = 32):
    """Returns (y, p) over eval_mask nodes of all items, as numpy arrays."""
    model.eval()
    ys, ps = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i: i + batch_size]
        p = forward_batch(model, chunk)
        y = torch.cat([it["y"] for it in chunk])
        m = torch.cat([it["eval_mask"] for it in chunk]) & ~torch.isnan(y)
        ys.append(y[m].numpy())
        ps.append(p[m].numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_items: List[dict], val_items: List[dict], *,
                epochs: int = 40, lr: float = 1e-3, weight_decay: float = 1e-4,
                batch_size: int = 16, patience: int = 8, seed: int = 0,
                log: Optional[Callable[[str], None]] = None) -> Dict[str, list]:
    """
    Weighted BCE (positives up-weighted by neg/pos), Adam, gradient clipping,
    early stopping on validation ROC-AUC. Restores the best weights in place.
    """
    ys = torch.cat([it["y"][it["train_mask"]] for it in train_items])
    ys = ys[~torch.isnan(ys)]
    pos_weight = float((ys == 0).sum() / max((ys == 1).sum(), 1))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "val_roc_auc": []}
    best_auc, best_state, bad = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_items))
        losses = []
        for i in range(0, len(order), batch_size):
            chunk = [train_items[j] for j in order[i: i + batch_size]]
            p = forward_batch(model, chunk).clamp(1e-6, 1 - 1e-6)
            y = torch.cat([it["y"] for it in chunk])
            m = torch.cat([it["train_mask"] for it in chunk]) & ~torch.isnan(y)
            w = torch.where(y[m] == 1, pos_weight, 1.0)
            loss = F.binary_cross_entropy(p[m], y[m], weight=w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        yv, pv = predict(model, val_items)
        auc = roc_auc_score(yv, pv)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_roc_auc"].append(float(auc))
        if log:
            log(f"    epoch {epoch + 1:3d}  loss {np.mean(losses):.4f}  val AUC {auc:.4f}")

        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return history


def summarize_runs(runs: List[Dict[str, float]], keys=("roc_auc", "pr_auc", "f1", "precision", "recall")):
    """Mean and std across seeds for each metric."""
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs])
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out
