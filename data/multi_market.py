"""
Price data for the multi-market model (S&P 500, NIFTY 50, FTSE 100, Nikkei 225).

Universe: ~200 large-cap stocks across 4 markets
    - S&P 500    — top 50 by market cap (USD)
    - NIFTY 50   — all 50 constituents (INR)
    - FTSE 100   — top 50 by market cap (GBP)
    - Nikkei 225 — top 50 by market cap (JPY)

`load_usd_panel()` downloads (or loads from cache) daily OHLCV for every
ticker plus the three FX rates, converts closes to USD and aligns everything
on the NYSE trading calendar:

    close_usd[t, i] = local_close[t, i] * usd_per_local_unit[t]

Non-US markets are forward-filled (max 5 days) on days they were closed, so a
local holiday shows up as a 0% return rather than a missing value.

Timing: Tokyo, Mumbai and London all close before New York on the same
calendar date, so using a non-US close dated `t` to predict a US outcome
after the NYSE close on `t` introduces no look-ahead.

Usage:
    >>> from data.multi_market import load_usd_panel
    >>> panel = load_usd_panel()           # cached under data/.cache/
    >>> panel["close"].shape               # (n_days, n_tickers)
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fixed window so results are reproducible (yfinance `end` is exclusive).
DATA_START = "2021-10-01"
DATA_END = "2026-09-30"

# ---------------------------------------------------------------------------
# Market configuration
# ---------------------------------------------------------------------------

# Top 50 S&P 500 stocks by market cap (as of mid-2026 approx)
SP500_TOP50 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "BRK-B", "JPM",
    "LLY", "UNH", "XOM", "V", "MA", "JNJ", "PG", "HD", "MRK", "COST",
    "ABBV", "CVX", "WMT", "BAC", "NFLX", "CRM", "KO", "AMD", "PEP", "TMO",
    "ACN", "ORCL", "LIN", "MCD", "ADBE", "CSCO", "GE", "ABT", "DHR", "TXN",
    "PM", "WFC", "ISRG", "IBM", "AMGN", "NEE", "RTX", "GS", "NOW", "QCOM",
]

# NIFTY 50 — append .NS for yfinance
NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "HCLTECH.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "POWERGRID.NS",
    "NTPC.NS", "ONGC.NS", "TATAMOTORS.NS", "COALINDIA.NS", "TECHM.NS",
    "ULTRACEMCO.NS", "DIVISLAB.NS", "BAJAJFINSV.NS", "DRREDDY.NS", "CIPLA.NS",
    "EICHERMOT.NS", "APOLLOHOSP.NS", "ADANIPORTS.NS", "NESTLEIND.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "GRASIM.NS", "HEROMOTOCO.NS", "TATACONSUM.NS", "BPCL.NS",
    "SHRIRAMFIN.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "ADANIENT.NS",
    "M&M.NS", "INDUSINDBK.NS", "TRENT.NS", "BEL.NS", "HINDALCO.NS",
]

# FTSE 100 top 50 — append .L for yfinance
FTSE_TOP50 = [
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L", "GSK.L", "DGE.L",
    "REL.L", "BATS.L", "NG.L", "VOD.L", "LSEG.L", "GLEN.L", "NWG.L", "LLOY.L",
    "BT-A.L", "AAL.L", "IMB.L", "EXPN.L", "ABF.L", "PSN.L", "MNG.L", "ANTO.L",
    "WTB.L", "SGE.L", "PRU.L", "ADM.L", "CPG.L", "CNA.L", "BKG.L", "BARC.L",
    "MNDI.L", "RKT.L", "AHT.L", "JMAT.L", "JD.L", "KGF.L", "SBRY.L", "MRO.L",
    "HLN.L", "SKG.L", "PHNX.L", "EZJ.L", "AUTO.L", "SMT.L", "BA.L", "RR.L", "SSE.L", "INF.L",
]

# Nikkei 225 top 50 — append .T for yfinance
NIKKEI_TOP50 = [
    "7203.T", "6758.T", "8306.T", "9432.T", "6861.T", "9984.T", "8316.T", "6098.T",
    "6367.T", "8035.T", "9433.T", "6594.T", "4063.T", "6902.T", "7267.T", "4503.T",
    "9022.T", "9613.T", "8031.T", "6702.T", "6645.T", "6326.T", "7741.T", "4543.T",
    "3382.T", "8801.T", "9020.T", "4911.T", "8766.T", "7011.T", "6857.T", "5108.T",
    "6501.T", "7733.T", "4005.T", "2914.T", "6752.T", "7201.T", "8411.T", "9101.T",
    "9202.T", "5401.T", "7751.T", "6770.T", "4452.T", "8802.T", "9303.T", "4568.T", "6971.T", "7974.T",
]

# Forex tickers on yfinance (all to USD)
FOREX_PAIRS = {
    "NIFTY":  "USDINR=X",   # USD per INR → invert to get INR→USD
    "FTSE":   "GBPUSD=X",   # USD per GBP → use directly
    "NIKKEI": "USDJPY=X",   # USD per JPY → invert to get JPY→USD
    "SP500":  None,          # Already USD
}

# Sector mappings (simplified; in production use yfinance .info)
SECTOR_MAP: Dict[str, int] = {
    # SP500 sectors
    "AAPL": 0, "MSFT": 0, "NVDA": 0, "GOOGL": 0, "AMZN": 1, "META": 0,
    "TSLA": 2, "AVGO": 0, "JPM": 3, "LLY": 4, "UNH": 4, "XOM": 5, "V": 3,
    "MA": 3, "JNJ": 4, "PG": 6, "HD": 1, "MRK": 4, "COST": 1, "ABBV": 4,
    "CVX": 5, "WMT": 1, "BAC": 3, "NFLX": 7, "CRM": 0, "KO": 6, "AMD": 0,
    "PEP": 6, "TMO": 4, "ACN": 0, "ORCL": 0, "MCD": 6, "ADBE": 0, "CSCO": 0,
    "GE": 8, "ABT": 4, "DHR": 4, "TXN": 0, "PM": 6, "WFC": 3, "GS": 3,
    "RTX": 8, "IBM": 0, "AMGN": 4, "NEE": 9, "NOW": 0, "QCOM": 0,
    "LIN": 10, "ISRG": 4, "BRK-B": 3,
    # Default all others to sector 11 (other)
}
# Sector names for reference
SECTOR_NAMES = {
    0: "Technology", 1: "Consumer", 2: "Automotive", 3: "Finance",
    4: "Healthcare", 5: "Energy", 6: "Consumer Staples", 7: "Media",
    8: "Industrial", 9: "Utilities", 10: "Materials", 11: "Other",
}

MARKET_TO_IDX = {"SP500": 0, "NIFTY": 1, "FTSE": 2, "NIKKEI": 3}

MARKETS = {"SP500": SP500_TOP50, "NIFTY": NIFTY50, "FTSE": FTSE_TOP50, "NIKKEI": NIKKEI_TOP50}

ALL_TICKERS = SP500_TOP50 + NIFTY50 + FTSE_TOP50 + NIKKEI_TOP50
TICKER_TO_MARKET = (
    {t: "SP500"  for t in SP500_TOP50}
    | {t: "NIFTY"  for t in NIFTY50}
    | {t: "FTSE"   for t in FTSE_TOP50}
    | {t: "NIKKEI" for t in NIKKEI_TOP50}
)


# ---------------------------------------------------------------------------
# Data fetcher
# ---------------------------------------------------------------------------

def _normalize_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Strip timezone and time-of-day so every market indexes by calendar date."""
    idx = pd.to_datetime(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


class MultiMarketFetcher:
    """
    Downloads daily OHLCV for all 4 markets and FX rates from yfinance,
    caching each group as a pickle so later runs work offline.

    Args:
        start, end: Date window passed to yfinance (`end` is exclusive).
        cache_dir:  Directory for the pickled downloads.
    """

    def __init__(self, start: str = DATA_START, end: str = DATA_END,
                 cache_dir: str = "data/.cache"):
        self.start = start
        self.end = end
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}_{self.start}_{self.end}.pkl")

    def _download(self, ticker: str) -> Optional[pd.DataFrame]:
        import yfinance as yf
        try:
            df = yf.Ticker(ticker).history(start=self.start, end=self.end, auto_adjust=True)
        except Exception as e:  # network errors, delisted tickers
            logger.warning(f"Failed to fetch {ticker}: {e}")
            return None
        if df.empty or len(df) < 30:
            return None
        df.index = _normalize_index(df.index)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def _fetch_group(self, tickers, market_name: str,
                     force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        path = self._cache_path(market_name)
        if os.path.exists(path) and not force_refresh:
            with open(path, "rb") as f:
                return pickle.load(f)

        print(f"  [{market_name}] downloading {len(tickers)} tickers...")
        data = {}
        for t in tickers:
            df = self._download(t)
            if df is not None:
                data[t] = df
        missing = sorted(set(tickers) - set(data))
        if missing:
            print(f"  [{market_name}] no data for {len(missing)} tickers: {missing}")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return data

    def _fetch_forex(self, force_refresh: bool = False) -> Dict[str, pd.Series]:
        """Returns {market: USD per one unit of local currency}."""
        path = self._cache_path("FOREX")
        if os.path.exists(path) and not force_refresh:
            with open(path, "rb") as f:
                return pickle.load(f)

        fx = {}
        for market, pair in FOREX_PAIRS.items():
            if pair is None:
                continue
            df = self._download(pair)
            if df is None:
                raise RuntimeError(f"Could not download FX rate {pair}")
            s = df["Close"]
            # USDINR=X / USDJPY=X quote local units per USD → invert
            fx[market] = 1.0 / s if pair.startswith("USD") else s
        with open(path, "wb") as f:
            pickle.dump(fx, f)
        return fx

    def fetch_all(self, force_refresh: bool = False
                  ) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, pd.Series]]:
        market_data = {
            "SP500":  self._fetch_group(SP500_TOP50,  "SP500",  force_refresh),
            "NIFTY":  self._fetch_group(NIFTY50,      "NIFTY",  force_refresh),
            "FTSE":   self._fetch_group(FTSE_TOP50,   "FTSE",   force_refresh),
            "NIKKEI": self._fetch_group(NIKKEI_TOP50, "NIKKEI", force_refresh),
        }
        return market_data, self._fetch_forex(force_refresh)


# ---------------------------------------------------------------------------
# USD panel on the NYSE calendar
# ---------------------------------------------------------------------------

def load_usd_panel(start: str = DATA_START, end: str = DATA_END,
                   cache_dir: str = "data/.cache", force_refresh: bool = False,
                   max_missing_frac: float = 0.05) -> Dict:
    """
    Returns a dict with
        close:   DataFrame (dates × tickers) of USD close prices
        volume:  DataFrame (dates × tickers) of share volume
        markets: list of market names aligned with the columns
    Tickers missing more than `max_missing_frac` of NYSE days are dropped.
    """
    market_data, fx = MultiMarketFetcher(start, end, cache_dir).fetch_all(force_refresh)

    us_frames = market_data["SP500"].values()
    calendar = sorted(set().union(*[df.index for df in us_frames]))
    calendar = pd.DatetimeIndex(calendar)

    closes, volumes, markets = {}, {}, []
    for market in ["SP500", "NIFTY", "FTSE", "NIKKEI"]:
        rate = fx.get(market)
        for ticker in MARKETS[market]:
            df = market_data[market].get(ticker)
            if df is None:
                continue
            close = df["Close"]
            if rate is not None:
                aligned_rate = rate.reindex(close.index.union(rate.index)).ffill().reindex(close.index)
                close = close * aligned_rate
            close = close[~close.index.duplicated()]
            vol = df["Volume"][~df.index.duplicated()]

            # Align to NYSE days: forward-fill closes across local holidays
            union = close.index.union(calendar)
            c = close.reindex(union).ffill(limit=5).reindex(calendar)
            v = vol.reindex(union).ffill(limit=5).reindex(calendar)
            if c.isna().mean() > max_missing_frac:
                logger.info(f"Dropping {ticker}: {c.isna().mean():.1%} missing")
                continue
            closes[ticker], volumes[ticker] = c, v
            markets.append(market)

    close_df = pd.DataFrame(closes).bfill().ffill()
    volume_df = pd.DataFrame(volumes).bfill().ffill()
    return {"close": close_df, "volume": volume_df, "markets": markets}
