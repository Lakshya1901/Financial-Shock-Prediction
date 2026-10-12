"""
Finance-sector comparison: GATv2 vs other GNNs vs non-graph baselines.

Re-evaluates the original capstone configuration (10 US financials, 1-day
horizon, 3% impact threshold, supply-chain + shock-star graph, 8 features)
with a chronological train / val / test split, several seeds, and baselines
that ignore the graph. Model selection (early stopping, F1 threshold) uses
validation only; the numbers reported are on the held-out test events.

It also saves the seed-0 GATv2 as the model served by the Streamlit app.

Usage (from the project root):
    python -m experiments.run_static_comparison            # 5 seeds
    python -m experiments.run_static_comparison --quick    # 1 seed, 5 epochs

Outputs:
    experiments/results/static_comparison.csv       mean ± std per model (test)
    experiments/results/static_comparison_runs.csv  one row per (model, seed)
    trained_models/finance_gatv2.pth                deployed app model + metadata
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from data.finance_universe import (CONFIG, FEATURE_NAMES, FINANCE_TICKERS, VOL_FEATURE,
                                   build_items, load_prices)
from experiments.run_ablation import flat_last_snapshot, score
from experiments.training import compute_metrics, predict, set_seed, summarize_runs, train_model
from models.baselines import GATv1Net, GCNNet, MLPNet, SAGENet, StaticGATv2, count_parameters

RESULTS_DIR = "experiments/results"
APP_MODEL_PATH = "trained_models/finance_gatv2.pth"

MODELS = {
    "GATv2": StaticGATv2,
    "GATv1": GATv1Net,
    "GCN": GCNNet,
    "GraphSAGE": SAGENet,
    "MLP": MLPNet,
}


def main(seeds: int, epochs: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(APP_MODEL_PATH), exist_ok=True)
    items = build_items(load_prices())
    for s in ("train", "val", "test"):
        dates = [it["date"] for it in items[s]]
        print(f"{s:5s}: {len(items[s]):4d} events  {min(dates).date()} → {max(dates).date()}")

    Xtr, ytr = flat_last_snapshot(items["train"])
    Xva, yva = flat_last_snapshot(items["val"])
    Xte, yte = flat_last_snapshot(items["test"])
    runs = [{"model": "VolatilityHeuristic", "seed": 0, "params": 0,
             **score(yva, Xva[:, VOL_FEATURE], yte, Xte[:, VOL_FEATURE])}]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    runs.append({"model": "LogisticRegression", "seed": 0, "params": Xtr.shape[1] + 1,
                 **score(yva, lr.predict_proba(Xva)[:, 1], yte, lr.predict_proba(Xte)[:, 1])})

    for name, cls in MODELS.items():
        for seed in range(seeds):
            set_seed(seed)
            model = cls(len(FEATURE_NAMES))
            train_model(model, items["train"], items["val"], epochs=epochs, lr=5e-3,
                        batch_size=8, patience=15, seed=seed)
            yv, pv = predict(model, items["val"])
            yt, pt = predict(model, items["test"])
            m = score(yv, pv, yt, pt)
            runs.append({"model": name, "seed": seed, "params": count_parameters(model), **m})
            print(f"{name:10s} seed {seed}: test AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}  "
                  f"F1 {m['f1']:.4f}")
            if name == "GATv2" and seed == 0:
                torch.save({
                    "state_dict": model.state_dict(),
                    "tickers": FINANCE_TICKERS,
                    "feature_names": FEATURE_NAMES,
                    "config": CONFIG,
                    "threshold": m["threshold"],
                    "test_metrics": m,
                    "trained_at": datetime.now().isoformat(timespec="seconds"),
                }, APP_MODEL_PATH)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(os.path.join(RESULTS_DIR, "static_comparison_runs.csv"), index=False)
    summary = []
    for name in dict.fromkeys(runs_df.model):
        rs = runs_df[runs_df.model == name].to_dict("records")
        summary.append({"model": name, "seeds": len(rs), "params": rs[0]["params"],
                        "base_rate": rs[0]["base_rate"], "n_test_nodes": rs[0]["n"],
                        **summarize_runs(rs)})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "static_comparison.csv"), index=False)
    cols = ["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std"]
    print("\nTest set (shock source excluded):")
    print(summary_df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Positive base rate on test: {summary_df.base_rate.iloc[0]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs = 1, 5
    main(a.seeds, a.epochs)
