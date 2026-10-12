"""
Real-data shock-event dataset for CascadeNet.

Turns the USD price panel from `data.multi_market.load_usd_panel()` into
supervised samples:

    sample = (event date d, shock source s)
    input  = T graph snapshots at trading days d-(T-1)*step, ..., d-step, d
    label  = y_i = 1 if |P_i(d + horizon) / P_i(d) - 1| > impact_threshold

Everything a sample sees is known at the close of day d:
    * shock detection uses a z-score against the *trailing* 60-day volatility
    * node features are trailing rolling statistics
    * Pearson edges use the trailing `corr_window` days
    * DCC-GARCH parameters and feature standardization are estimated on the
      training period only, then applied forward (see DCCGARCHEstimator.correlation_path)

Events are split chronologically into train / val / test, with a `horizon`-day
purge gap so no training label overlaps the evaluation period.

Node features (14, same layout as TEMPORAL_GATV2_DESIGN.md):
    [0]  20d mean return        [7]  is non-US stock
    [1]  20d volatility         [8-11] market one-hot (SP500, NIFTY, FTSE, NIKKEI)
    [2]  latest daily return    [12] sector index / 11
    [3]  5d mean return         [13] log(volume / 5d mean volume), clipped ±3
    [4]  price / 20d mean - 1
    [5]  log(volume / 20d mean volume)
    [6]  is shock source (set only on the last snapshot of the sample)
Continuous features (0-5, 13) are z-scored with training-period statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from data.dynamic_correlation import DCCGARCHEstimator
from data.multi_market import MARKET_TO_IDX, SECTOR_MAP, load_usd_panel

N_FEATURES = 14
SHOCK_FLAG = 6
CONTINUOUS = [0, 1, 2, 3, 4, 5, 13]
VOL_FEATURE = 1  # used by the volatility heuristic baseline


@dataclass
class DatasetConfig:
    T: int = 12                    # snapshots per sample
    step: int = 5                  # trading days between snapshots (weekly)
    horizon: int = 5               # label horizon in trading days
    impact_threshold: float = 0.03
    shock_z: float = 3.0           # |return| / trailing vol threshold
    shock_min_return: float = 0.03
    max_events_per_day: int = 3    # keep only the largest shocks on crowded days
    vol_window: int = 60
    corr_window: int = 120
    top_k: int = 10                # neighbours per node in each snapshot graph
    train_frac: float = 0.6
    val_frac: float = 0.2


@dataclass
class ShockDataset:
    tickers: List[str]
    markets: List[str]
    dates: pd.DatetimeIndex
    X: np.ndarray                  # (D, N, F) standardized features (flag = 0)
    raw_vol: np.ndarray            # (D, N) unstandardized 20d volatility
    Y: np.ndarray                  # (D, N) labels, NaN where horizon runs past the data
    events: pd.DataFrame           # date_idx, src_idx, ticker, z, ret, split
    edges: Dict[str, Dict[int, tuple]] = field(default_factory=dict)
    # edges[graph_name][date_idx] -> (edge_index LongTensor, edge_attr FloatTensor)
    node_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    config: DatasetConfig = field(default_factory=DatasetConfig)
    dcc_params: Optional[dict] = None

    def snapshot_days(self, date_idx: int) -> List[int]:
        c = self.config
        return [date_idx - (c.T - 1 - k) * c.step for k in range(c.T)]

    def sample(self, event_row, graph: str, node_set: str, temporal: bool = True) -> dict:
        """
        Build model inputs for one event (an item for experiments.training).

        graph:    "pearson" (one trailing-window graph reused for every snapshot)
                  or "dcc" (the DCC correlation graph at each snapshot date)
        node_set: "us" (S&P 500 only) or "all" (4 markets)
        Training uses every labelled node except the shock source; evaluation
        scores only US stocks other than the source, so all configs are
        compared on the same targets.
        """
        nodes = self.node_sets[node_set]
        d, src = int(event_row.date_idx), int(event_row.src_idx)
        days = self.snapshot_days(d) if temporal else [d]
        is_src = nodes == src

        snaps = []
        for day in days:
            x = torch.from_numpy(self.X[day][nodes])
            if day == d and is_src.any():
                x = x.clone()
                x[torch.from_numpy(is_src), SHOCK_FLAG] = 1.0
            ei, ea = self.edges[f"{graph}:{node_set}"][day if graph == "dcc" else d]
            snaps.append(Data(x=x, edge_index=ei, edge_attr=ea))

        us = np.array([self.markets[n] == "SP500" for n in nodes])
        return {
            "snaps": snaps,
            "y": torch.from_numpy(self.Y[d][nodes]).float(),
            "train_mask": torch.from_numpy(~is_src),
            "eval_mask": torch.from_numpy(us & ~is_src),
        }


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _feature_panel(close: pd.DataFrame, volume: pd.DataFrame, markets: List[str]) -> np.ndarray:
    r = close.pct_change()
    vol = volume.replace(0, np.nan)
    feats = [
        r.rolling(20).mean() * 100,
        r.rolling(20).std() * 100,
        r * 100,
        r.rolling(5).mean() * 100,
        close / close.rolling(20).mean() - 1,
        np.log(vol / vol.rolling(20).mean()),
    ]
    D, N = close.shape
    X = np.zeros((D, N, N_FEATURES), dtype=np.float32)
    for k, f in enumerate(feats):
        X[:, :, k] = f.values
    X[:, :, 13] = np.clip(np.log(vol / vol.rolling(5).mean()).values, -3, 3)
    for j, (t, m) in enumerate(zip(close.columns, markets)):
        X[:, j, 7] = float(m != "SP500")
        X[:, j, 8 + MARKET_TO_IDX[m]] = 1.0
        X[:, j, 12] = SECTOR_MAP.get(t.split(".")[0], 11) / 11.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _detect_events(r: pd.DataFrame, cfg: DatasetConfig, first: int, last: int) -> pd.DataFrame:
    trailing_vol = r.rolling(cfg.vol_window).std().shift(1)
    z = (r / trailing_vol).values
    rv = r.values
    rows = []
    for d in range(first, last + 1):
        hits = np.where((np.abs(z[d]) > cfg.shock_z) & (np.abs(rv[d]) > cfg.shock_min_return))[0]
        hits = sorted(hits, key=lambda j: -abs(z[d, j]))[: cfg.max_events_per_day]
        for j in hits:
            rows.append({"date_idx": d, "src_idx": int(j), "z": float(z[d, j]), "ret": float(rv[d, j])})
    return pd.DataFrame(rows)


def _assign_splits(events: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    days = np.sort(events.date_idx.unique())
    train_end = days[int(len(days) * cfg.train_frac) - 1]
    val_end = days[int(len(days) * (cfg.train_frac + cfg.val_frac)) - 1]
    split = np.full(len(events), "", dtype=object)
    d = events.date_idx.values
    split[d <= train_end] = "train"
    split[(d > train_end + cfg.horizon) & (d <= val_end)] = "val"
    split[d > val_end + cfg.horizon] = "test"
    events = events.assign(split=split)
    return events[events.split != ""].reset_index(drop=True)


def _top_k_edges(corr: np.ndarray, k: int) -> tuple:
    """Symmetric top-k |corr| graph. Returns (edge_index, edge_attr)."""
    a = np.abs(np.nan_to_num(corr))
    np.fill_diagonal(a, -1.0)
    k = min(k, a.shape[0] - 1)
    nbrs = np.argpartition(-a, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(a.shape[0]), k)
    cols = nbrs.ravel()
    pairs = np.unique(np.concatenate([np.stack([rows, cols]), np.stack([cols, rows])], axis=1), axis=1)
    ei = torch.from_numpy(pairs).long()
    ea = torch.from_numpy(a[pairs[0], pairs[1]]).float().unsqueeze(1)
    return ei, ea


def build_dataset(cfg: Optional[DatasetConfig] = None, panel: Optional[Dict] = None,
                  verbose: bool = True) -> ShockDataset:
    cfg = cfg or DatasetConfig()
    panel = panel or load_usd_panel()
    close, volume, markets = panel["close"], panel["volume"], panel["markets"]
    tickers = list(close.columns)
    r = close.pct_change().fillna(0.0)
    D, N = close.shape
    log = print if verbose else (lambda *a, **k: None)

    # Labels
    fwd = close.shift(-cfg.horizon) / close - 1
    Y = (fwd.abs() > cfg.impact_threshold).astype(float).where(fwd.notna()).values

    # Events need a full snapshot history plus a correlation window behind them
    first = max(cfg.vol_window + 1, cfg.corr_window, (cfg.T - 1) * cfg.step + 20)
    last = D - 1 - cfg.horizon
    events = _assign_splits(_detect_events(r, cfg, first, last), cfg)
    events["ticker"] = [tickers[j] for j in events.src_idx]
    train_end = int(events.loc[events.split == "train", "date_idx"].max())
    log(f"[dataset] {len(events)} events: " + ", ".join(
        f"{s}={n}" for s, n in events.split.value_counts().reindex(["train", "val", "test"]).items()))
    log(f"[dataset] train ≤ {close.index[train_end].date()}, test from "
        f"{close.index[events.loc[events.split == 'test', 'date_idx'].min()].date()}")

    # Features, standardized on the training period
    X = _feature_panel(close, volume, markets)
    raw_vol = X[:, :, VOL_FEATURE].copy()
    ref = X[first: train_end + 1]
    mu = ref[..., CONTINUOUS].reshape(-1, len(CONTINUOUS)).mean(0)
    sd = ref[..., CONTINUOUS].reshape(-1, len(CONTINUOUS)).std(0) + 1e-8
    X[..., CONTINUOUS] = np.clip((X[..., CONTINUOUS] - mu) / sd, -10, 10)

    node_sets = {
        "us": np.array([i for i, m in enumerate(markets) if m == "SP500"]),
        "all": np.arange(N),
    }
    ds = ShockDataset(tickers, markets, close.index, X, raw_vol, Y, events,
                      node_sets=node_sets, config=cfg)

    # Pearson edges: one graph per event date, from the trailing window
    for d in sorted(events.date_idx.unique()):
        C = np.corrcoef(r.values[d - cfg.corr_window + 1: d + 1].T)
        for name, nodes in node_sets.items():
            ds.edges.setdefault(f"pearson:{name}", {})[d] = _top_k_edges(C[np.ix_(nodes, nodes)], cfg.top_k)

    # DCC edges: fit on the training window, filter forward over every date
    log("[dataset] fitting DCC-GARCH on the training window...")
    est = DCCGARCHEstimator(min_obs=120, dcc_max_iter=100)
    est.fit(r.iloc[1: train_end + 1])
    assert est.fitted_tickers_ == tickers, "DCC dropped tickers; node order would break"
    needed = {day for d in events.date_idx.unique() for day in ds.snapshot_days(d)}
    for d, (_, R) in enumerate(est.correlation_path(r)):
        if d in needed:
            for name, nodes in node_sets.items():
                ds.edges.setdefault(f"dcc:{name}", {})[d] = _top_k_edges(R[np.ix_(nodes, nodes)], cfg.top_k)
    ds.dcc_params = {"a": est.dcc_a_, "b": est.dcc_b_}
    log(f"[dataset] DCC a={est.dcc_a_:.4f} b={est.dcc_b_:.4f}")
    return ds


def load_or_build(path: str = "data/processed/shock_dataset.pkl", rebuild: bool = False,
                  cfg: Optional[DatasetConfig] = None) -> ShockDataset:
    """Cache the built dataset (DCC fitting takes a few minutes)."""
    import os
    import pickle
    if os.path.exists(path) and not rebuild:
        with open(path, "rb") as f:
            ds = pickle.load(f)
        if cfg is None or ds.config == cfg:
            return ds
    ds = build_dataset(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ds, f)
    return ds
