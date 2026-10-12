"""
Single-snapshot shock dataset for 10 large US financials.

This is the original capstone setup ("OptimalGNN" configuration), with the
evaluation problems fixed:
    * shocks are detected with a z-score against the *trailing* 60-day
      volatility (the old code used full-sample mean/std, i.e. future data)
    * events are sorted by date before the chronological train/val/test split
      (the old code split a list ordered by ticker, so it was not temporal)
    * the shock source is excluded from scoring

The same feature and graph functions are used for training and by the
Streamlit app, so the deployed model sees exactly the inputs it was trained on.

Sample:  event (date d, source s) → graph of the 10 banks at the close of d
Label:   y_i = 1 if |P_i(d + horizon) / P_i(d) − 1| > impact_threshold
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from data.multi_market import DATA_END, DATA_START, MultiMarketFetcher

FINANCE_TICKERS = ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SCHW", "USB"]

# Documented interbank / peer relationships (directed). Unweighted in training.
SUPPLY_CHAIN = {
    "JPM": ["BAC", "GS", "MS"],
    "GS": ["JPM", "MS"],
    "MS": ["JPM", "GS"],
    "BAC": ["JPM", "C"],
    "BLK": ["JPM", "GS", "MS"],
}

# Configuration chosen by the original hyperparameter search
CONFIG = {
    "horizon": 1,             # trading days
    "impact_threshold": 0.03,
    "shock_z": 3.0,
    "lookback": 10,
    "vol_window": 60,
    "max_events_per_day": 3,
    "train_frac": 0.6,
    "val_frac": 0.2,
}

FEATURE_NAMES = [
    "mean_return_pct", "volatility_pct", "latest_return_pct", "avg_5d_return_pct",
    "price_vs_mean", "volume_vs_mean", "is_shock_source", "in_supply_chain",
]
VOL_FEATURE = 1


def load_prices(start: str = DATA_START, end: str = DATA_END, cache_dir: str = "data/.cache",
                force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    fetcher = MultiMarketFetcher(start, end, cache_dir)
    path = fetcher._cache_path("FINANCE")
    if os.path.exists(path) and not force_refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    data = {t: df for t in FINANCE_TICKERS if (df := fetcher._download(t)) is not None}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


def node_features(hist: pd.DataFrame, ticker: str, shock_ticker: Optional[str],
                  lookback: int = CONFIG["lookback"]) -> np.ndarray:
    """8 features from the last `lookback` rows of `hist` (OHLCV up to and including today)."""
    h = hist.tail(lookback)
    r = h["Close"].pct_change().dropna()
    feat = [
        r.mean() * 100,
        r.std() * 100,
        r.iloc[-1] * 100,
        r.iloc[-5:].mean() * 100,
        h["Close"].iloc[-1] / h["Close"].mean() - 1,
        h["Volume"].iloc[-1] / h["Volume"].mean(),
        float(ticker == shock_ticker),
        float(ticker in SUPPLY_CHAIN),
    ]
    return np.nan_to_num(np.array(feat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_graph(tickers: List[str], features: Dict[str, np.ndarray],
                shock_ticker: Optional[str]) -> Data:
    """Supply-chain edges (weight 1.0) plus a star from the shock source (weight 0.5)."""
    idx = {t: i for i, t in enumerate(tickers)}
    edges, weights = [], []
    for src, dsts in SUPPLY_CHAIN.items():
        for dst in dsts:
            if src in idx and dst in idx:
                edges.append((idx[src], idx[dst]))
                weights.append(1.0)
    if shock_ticker in idx:
        s = idx[shock_ticker]
        for i in range(len(tickers)):
            if i != s:
                edges.append((s, i))
                weights.append(0.5)
    return Data(
        x=torch.tensor(np.stack([features[t] for t in tickers])),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(weights, dtype=torch.float).unsqueeze(1),
    )


def build_items(prices: Dict[str, pd.DataFrame], cfg: dict = CONFIG) -> Dict[str, List[dict]]:
    """Detect shocks, build one graph per event, split chronologically."""
    tickers = [t for t in FINANCE_TICKERS if t in prices]
    close = pd.DataFrame({t: prices[t]["Close"] for t in tickers}).dropna()
    r = close.pct_change()
    z = r / r.rolling(cfg["vol_window"]).std().shift(1)
    fwd = close.shift(-cfg["horizon"]) / close - 1
    dates = close.index

    events = []
    for d in range(cfg["vol_window"] + 1, len(dates) - cfg["horizon"]):
        hits = [t for t in tickers if abs(z.iloc[d][t]) > cfg["shock_z"]]
        hits = sorted(hits, key=lambda t: -abs(z.iloc[d][t]))[: cfg["max_events_per_day"]]
        events += [(d, t) for t in hits]

    days = sorted({d for d, _ in events})
    train_end = days[int(len(days) * cfg["train_frac"]) - 1]
    val_end = days[int(len(days) * (cfg["train_frac"] + cfg["val_frac"])) - 1]

    items = {"train": [], "val": [], "test": []}
    for d, src in events:
        if d <= train_end:
            split = "train"
        elif train_end + cfg["horizon"] < d <= val_end:
            split = "val"
        elif d > val_end + cfg["horizon"]:
            split = "test"
        else:
            continue  # purge gap
        date = dates[d]
        feats = {t: node_features(prices[t].loc[:date], t, src, cfg["lookback"]) for t in tickers}
        g = build_graph(tickers, feats, src)
        not_src = torch.tensor([t != src for t in tickers])
        items[split].append({
            "snaps": [g],
            "y": torch.tensor((fwd.iloc[d].abs() > cfg["impact_threshold"]).astype(float).values,
                              dtype=torch.float),
            "train_mask": not_src,
            "eval_mask": not_src,
            "date": date, "source": src,
        })
    return items
