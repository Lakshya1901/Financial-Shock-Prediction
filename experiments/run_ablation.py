"""
Multi-market ablation study on real data from four markets.

Each configuration adds one component on top of the previous one:

    StaticGATv2            single snapshot at the shock date, trailing Pearson
                           edges, S&P 500 nodes only
    TemporalGATv2_Static   + 12 weekly snapshots and temporal attention
                           (same Pearson graph in every snapshot)
    TemporalGATv2_DCC      + a DCC-GARCH correlation graph per snapshot
    MultiMarket_Full        + NIFTY / FTSE / Nikkei stocks as extra nodes
                           (and foreign shock sources become visible)

Non-graph reference points, evaluated on the same events and nodes:
    VolatilityHeuristic    score = trailing 20-day volatility (no training)
    LogisticRegression     on the shock-date node features
    MLP                    2-layer per-node MLP on the shock-date node features

Every model is scored on the same targets: S&P 500 stocks other than the shock
source, on held-out test events that come strictly after the validation period.
The F1 threshold is chosen on validation and then applied to test. Trained
models are repeated over several seeds; results report mean ± std.

Usage (from the project root):
    python -m experiments.run_ablation                # 5 seeds, full run
    python -m experiments.run_ablation --quick        # 1 seed, 2 epochs (smoke test)

Outputs:
    experiments/results/ablation_results.csv          mean ± std per model (test)
    experiments/results/ablation_runs.csv             one row per (model, seed)
    experiments/results/ablation_histories.json       training curves
    experiments/results/ablation_test_predictions.npz seed-0 test labels/scores
    trained_models/<config>_seed0.pth                 seed-0 checkpoints
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from data.shock_dataset import N_FEATURES, VOL_FEATURE, load_or_build
from experiments.training import (best_f1_threshold, compute_metrics, predict, set_seed,
                                  summarize_runs, train_model)
from models.baselines import MLPNet, StaticGATv2, count_parameters
from models.temporal_gatv2 import TemporalGATv2

RESULTS_DIR = "experiments/results"
MODELS_DIR = "trained_models"

GNN_CONFIGS = [
    {"name": "StaticGATv2",          "temporal": False, "graph": "pearson", "nodes": "us"},
    {"name": "TemporalGATv2_Static", "temporal": True,  "graph": "pearson", "nodes": "us"},
    {"name": "TemporalGATv2_DCC",    "temporal": True,  "graph": "dcc",     "nodes": "us"},
    {"name": "MultiMarket_Full",      "temporal": True,  "graph": "dcc",     "nodes": "all"},
]


def build_items(ds, graph: str, nodes: str, temporal: bool):
    ev = ds.events
    return {s: [ds.sample(row, graph, nodes, temporal) for row in ev[ev.split == s].itertuples()]
            for s in ("train", "val", "test")}


def flat_last_snapshot(items):
    """(X, y) over eval-mask nodes using shock-date features — for sklearn / heuristics."""
    X, y = [], []
    for it in items:
        m = it["eval_mask"] & ~torch.isnan(it["y"])
        X.append(it["snaps"][-1].x[m].numpy())
        y.append(it["y"][m].numpy())
    return np.concatenate(X), np.concatenate(y)


def score(y_val, p_val, y_test, p_test):
    return compute_metrics(y_test, p_test, best_f1_threshold(y_val, p_val))


NEURAL = [{"name": "MLP", "temporal": False, "graph": "pearson", "nodes": "us", "mlp": True}] + GNN_CONFIGS

_DS = None
_ITEMS = {}


def _init_worker(threads: int):
    global _DS
    torch.set_num_threads(threads)
    _DS = load_or_build()


def _items_for(cfg):
    key = (cfg["graph"], cfg["nodes"], cfg["temporal"])
    if key not in _ITEMS:
        _ITEMS[key] = build_items(_DS, *key)
    return _ITEMS[key]


def run_neural(task):
    """One (config, seed) training run. Executed in a worker process."""
    cfg, seed, epochs, patience = task
    items = _items_for(cfg)
    set_seed(seed)
    if cfg.get("mlp"):
        model = MLPNet(N_FEATURES)
    elif cfg["temporal"]:
        model = TemporalGATv2(input_dim=N_FEATURES)
    else:
        model = StaticGATv2(N_FEATURES)
    t0 = time.time()
    hist = train_model(model, items["train"], items["val"], epochs=epochs, patience=patience, seed=seed)
    yv, pv = predict(model, items["val"])
    yt, pt = predict(model, items["test"])
    m = score(yv, pv, yt, pt)
    run = {"model": cfg["name"], "seed": seed, "params": count_parameters(model),
           "epochs_run": len(hist["train_loss"]), "train_time_s": round(time.time() - t0, 1), **m}
    print(f"  {cfg['name']:22s} seed {seed}: test AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}  "
          f"F1 {m['f1']:.4f}  ({run['epochs_run']} epochs, {run['train_time_s']:.0f}s)", flush=True)
    if seed == 0:
        torch.save({"state_dict": model.state_dict(), "config": cfg,
                    "dataset_config": _DS.config.__dict__},
                   os.path.join(MODELS_DIR, f"{cfg['name']}_seed0.pth"))
    return run, hist, (yt, pt) if seed == 0 else None


def main(seeds: int, epochs: int, patience: int, rebuild: bool, workers: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    ds = load_or_build(rebuild=rebuild)
    ev = ds.events
    print(f"Events: {dict(ev.split.value_counts())}  |  DCC a={ds.dcc_params['a']:.4f} "
          f"b={ds.dcc_params['b']:.4f}")

    runs, histories, test_preds = [], {}, {}

    # ---- Non-graph references (US nodes, shock-date features) --------------
    static_us = build_items(ds, "pearson", "us", temporal=False)
    Xtr, ytr = flat_last_snapshot(static_us["train"])
    Xva, yva = flat_last_snapshot(static_us["val"])
    Xte, yte = flat_last_snapshot(static_us["test"])

    m = score(yva, Xva[:, VOL_FEATURE], yte, Xte[:, VOL_FEATURE])
    runs.append({"model": "VolatilityHeuristic", "seed": 0, "params": 0, **m})
    test_preds["VolatilityHeuristic"] = (yte, Xte[:, VOL_FEATURE])

    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    m = score(yva, lr.predict_proba(Xva)[:, 1], yte, lr.predict_proba(Xte)[:, 1])
    runs.append({"model": "LogisticRegression", "seed": 0, "params": N_FEATURES + 1, **m})
    test_preds["LogisticRegression"] = (yte, lr.predict_proba(Xte)[:, 1])

    # ---- Trained neural models, one process per (config, seed) -------------
    # Most expensive configs first so the pool stays busy
    tasks = [(cfg, seed, epochs, patience) for cfg in reversed(NEURAL) for seed in range(seeds)]
    print(f"Training {len(tasks)} runs on {workers} worker processes...")
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(1,)) as pool:
        results = pool.map(run_neural, tasks, chunksize=1)
    for (cfg, seed, _, _), (run, hist, preds) in sorted(
            zip(tasks, results), key=lambda r: ([c["name"] for c in NEURAL].index(r[0][0]["name"]), r[0][1])):
        runs.append(run)
        histories[f"{cfg['name']}/seed{seed}"] = hist
        if preds is not None:
            test_preds[cfg["name"]] = preds

    # ---- Save ----------------------------------------------------------------
    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(os.path.join(RESULTS_DIR, "ablation_runs.csv"), index=False)
    order = list(dict.fromkeys(runs_df.model))
    summary = []
    for name in order:
        rs = runs_df[runs_df.model == name].to_dict("records")
        summary.append({"model": name, "seeds": len(rs), "params": rs[0]["params"],
                        "base_rate": rs[0]["base_rate"], "n_test_nodes": rs[0]["n"],
                        **summarize_runs(rs)})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "ablation_histories.json"), "w") as f:
        json.dump(histories, f)
    np.savez(os.path.join(RESULTS_DIR, "ablation_test_predictions.npz"),
             **{f"{k}__y": v[0] for k, v in test_preds.items()},
             **{f"{k}__p": v[1] for k, v in test_preds.items()})

    print("\nTest set (S&P 500 targets, shock source excluded):")
    cols = ["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std"]
    print(summary_df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Positive base rate on test: {summary_df.base_rate.iloc[0]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cached dataset")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true", help="1 seed, 2 epochs")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs = 1, 2
    main(a.seeds, a.epochs, a.patience, a.rebuild, a.workers)
