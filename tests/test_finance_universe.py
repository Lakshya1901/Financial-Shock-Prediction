import numpy as np
import pandas as pd

from data.finance_universe import FEATURE_NAMES, FINANCE_TICKERS, build_graph, build_items, node_features


def _prices(n_days=700, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    out = {}
    for t in FINANCE_TICKERS:
        r = rng.normal(0, 0.015, n_days)
        r[rng.random(n_days) < 0.01] += -0.09
        close = 50 * np.cumprod(1 + r)
        out[t] = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                               "Volume": rng.integers(1e6, 2e6, n_days).astype(float)}, index=dates)
    return out


def test_node_features_and_graph():
    prices = _prices()
    feats = {t: node_features(prices[t], t, "JPM") for t in FINANCE_TICKERS}
    assert all(f.shape == (len(FEATURE_NAMES),) for f in feats.values())
    assert feats["JPM"][6] == 1.0 and feats["BAC"][6] == 0.0
    g = build_graph(FINANCE_TICKERS, feats, "JPM")
    assert g.x.shape == (10, 8)
    assert g.edge_index.shape[1] == g.edge_attr.shape[0]
    # star edges from the source to every other bank
    src = FINANCE_TICKERS.index("JPM")
    assert set(g.edge_index[1][g.edge_index[0] == src].tolist()) >= set(range(10)) - {src}


def test_items_are_chronological_and_exclude_source():
    items = build_items(_prices())
    last_train = max(it["date"] for it in items["train"])
    first_val = min(it["date"] for it in items["val"])
    last_val = max(it["date"] for it in items["val"])
    first_test = min(it["date"] for it in items["test"])
    assert last_train < first_val and last_val < first_test
    it = items["train"][0]
    src = FINANCE_TICKERS.index(it["source"])
    assert not it["eval_mask"][src]
