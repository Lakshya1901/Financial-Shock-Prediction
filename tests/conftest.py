import numpy as np
import pandas as pd
import pytest

from data.multi_market import FTSE_TOP50, NIFTY50, NIKKEI_TOP50, SP500_TOP50


def make_panel(n_days=400, seed=0, per_market=(4, 2, 2, 2)):
    """Synthetic USD price/volume panel with a common factor and occasional jumps."""
    rng = np.random.default_rng(seed)
    tickers = (SP500_TOP50[: per_market[0]] + NIFTY50[: per_market[1]]
               + FTSE_TOP50[: per_market[2]] + NIKKEI_TOP50[: per_market[3]])
    markets = (["SP500"] * per_market[0] + ["NIFTY"] * per_market[1]
               + ["FTSE"] * per_market[2] + ["NIKKEI"] * per_market[3])
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    factor = rng.normal(0, 0.01, n_days)
    rets = factor[:, None] * 0.8 + rng.normal(0, 0.012, (n_days, len(tickers)))
    jumps = rng.random((n_days, len(tickers))) < 0.01
    rets[jumps] += rng.choice([-0.08, 0.08], jumps.sum())
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=tickers)
    volume = pd.DataFrame(rng.integers(1e6, 2e6, close.shape).astype(float), index=dates, columns=tickers)
    return {"close": close, "volume": volume, "markets": markets}


@pytest.fixture
def panel():
    return make_panel()
