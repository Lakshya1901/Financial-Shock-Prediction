"""
Paper / README figures, generated only from real experiment outputs.

Requires (run these first):
    python -m experiments.run_static_comparison
    python -m experiments.run_ablation

There is no synthetic fallback: if an input is missing the script stops and
says which experiment to run.

Usage (from the project root):
    python -m experiments.generate_figures

Writes PDF + PNG to experiments/figures/:
    static_comparison     finance-sector test ROC-AUC / PR-AUC by model
    ablation_metrics      CascadeNet ablation test ROC-AUC / PR-AUC (mean ± std)
    ablation_roc          seed-0 test ROC curves
    training_curves       validation ROC-AUC per epoch (seed 0)
    temporal_attention    CascadeNet last-query attention over the 12 weeks, by market
    cross_market_attention  mean first-layer GATv2 edge attention by market pair
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from analysis.attention_attribution import EdgeImportanceAnalyzer, TemporalAttentionAnalyzer  # noqa: E402
from data.shock_dataset import load_or_build  # noqa: E402
from models.temporal_gatv2 import TemporalGATv2  # noqa: E402

RESULTS = "experiments/results"
FIGURES = "experiments/figures"
MARKETS = ["SP500", "NIFTY", "FTSE", "NIKKEI"]
BASELINE_COLOR, GNN_COLOR = "#9e9e9e", "#1f77b4"
NON_GRAPH = {"VolatilityHeuristic", "LogisticRegression", "MLP"}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.grid": True, "grid.color": "#dddddd",
    "grid.linewidth": 0.5, "figure.facecolor": "white", "savefig.bbox": "tight",
})


def _require(path: str, how: str) -> str:
    if not os.path.exists(path):
        sys.exit(f"Missing {path}. Run `{how}` first.")
    return path


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


def _metric_bars(df: pd.DataFrame, title: str, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    colors = [BASELINE_COLOR if m in NON_GRAPH else GNN_COLOR for m in df.model]
    for ax, metric, label in [(axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC")]:
        ax.bar(range(len(df)), df[f"{metric}_mean"], yerr=df[f"{metric}_std"], color=colors,
               capsize=3, edgecolor="#333333", linewidth=0.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.model, rotation=35, ha="right")
        ax.set_ylabel(f"Test {label}")
        ref = 0.5 if metric == "roc_auc" else df.base_rate.iloc[0]
        ax.axhline(ref, color="#d62728", ls="--", lw=1,
                   label="random" if metric == "roc_auc" else "base rate")
        lo = min(ref, (df[f"{metric}_mean"] - df[f"{metric}_std"]).min())
        hi = (df[f"{metric}_mean"] + df[f"{metric}_std"]).max()
        ax.set_ylim(max(0, lo - 0.05), min(1, hi + 0.05))
        ax.legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, name)


def static_comparison() -> None:
    df = pd.read_csv(_require(f"{RESULTS}/static_comparison.csv",
                              "python -m experiments.run_static_comparison"))
    _metric_bars(df, "Finance sector (10 banks, 1-day horizon) — held-out test events",
                 "static_comparison")


def ablation_metrics() -> None:
    df = pd.read_csv(_require(f"{RESULTS}/ablation_results.csv", "python -m experiments.run_ablation"))
    _metric_bars(df, "CascadeNet ablation — S&P 500 targets, held-out test events",
                 "ablation_metrics")


def ablation_roc() -> None:
    npz = np.load(_require(f"{RESULTS}/ablation_test_predictions.npz",
                           "python -m experiments.run_ablation"))
    names = [k[:-3] for k in npz.files if k.endswith("__y")]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name in names:
        fpr, tpr, _ = roc_curve(npz[f"{name}__y"], npz[f"{name}__p"])
        ax.plot(fpr, tpr, ls="--" if name in NON_GRAPH else "-", label=name)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC curves (seed 0)")
    ax.legend(fontsize=8)
    _save(fig, "ablation_roc")


def training_curves() -> None:
    with open(_require(f"{RESULTS}/ablation_histories.json", "python -m experiments.run_ablation")) as f:
        hist = json.load(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, h in hist.items():
        if key.endswith("/seed0"):
            ax.plot(range(1, len(h["val_roc_auc"]) + 1), h["val_roc_auc"], marker="o", ms=3,
                    label=key.split("/")[0])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title("Validation ROC-AUC during training (seed 0)")
    ax.legend(fontsize=8)
    _save(fig, "training_curves")


def attention_figures() -> None:
    ckpt = torch.load(_require("trained_models/CascadeNet_Full_seed0.pth",
                               "python -m experiments.run_ablation"), weights_only=False)
    ds = load_or_build()
    model = TemporalGATv2(input_dim=ds.X.shape[2])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    temporal, edges = TemporalAttentionAnalyzer(model), EdgeImportanceAnalyzer(model)

    nodes = ds.node_sets["all"]
    tickers = [ds.tickers[n] for n in nodes]
    node_market = np.array([MARKETS.index(ds.markets[n]) for n in nodes])
    t2m = {t: ds.markets[n] for t, n in zip(tickers, nodes)}

    attn_sum = np.zeros((4, ds.config.T))
    flow = np.zeros((4, 4))
    test = ds.events[ds.events.split == "test"]
    for row in test.itertuples():
        snaps = ds.sample(row, "dcc", "all")["snaps"]
        last_q = temporal.get_attention_maps(snaps)["attention_weights"][:, -1, :]
        for m in range(4):
            attn_sum[m] += last_q[node_market == m].mean(0)
        flow += edges.cross_market_edge_heatmap(snaps, tickers, t2m)
    attn, flow = attn_sum / len(test), flow / len(test)

    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_yticks(range(4))
    ax.set_yticklabels(MARKETS)
    T = ds.config.T
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"t-{T - 1 - k}w" if k < T - 1 else "shock\nday" for k in range(T)], fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="attention")
    ax.set_title(f"CascadeNet temporal attention from the shock-day position "
                 f"(mean over {len(test)} test events)")
    _save(fig, "temporal_attention")

    fig, ax = plt.subplots(figsize=(4.8, 4))
    im = ax.imshow(flow, cmap="magma")
    ax.set_xticks(range(4))
    ax.set_xticklabels(MARKETS)
    ax.set_yticks(range(4))
    ax.set_yticklabels(MARKETS)
    ax.set_xlabel("target market")
    ax.set_ylabel("source market")
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{flow[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean edge attention")
    ax.set_title("GATv2 layer-1 edge attention by market pair")
    _save(fig, "cross_market_attention")


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    static_comparison()
    ablation_metrics()
    ablation_roc()
    training_curves()
    attention_figures()
