"""Volatility-regime features.

Market-wide (full history since 1990): VIX level, VIX 1-year percentile, and the variance
risk premium proxy VIX/100 - realized vol of SPY.

Per-stock implied vol (only from the day options recording started, NaN before):
iv30_atm, iv_rv_spread = iv30 - 21d realized vol, iv_rank_252, skew30, term_slope,
put/call ratios. A snapshot recorded after the close of day d is usable at d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = np.sqrt(252)
IV_RANK_MIN_OBS = 20


def market(ms: pd.DataFrame) -> pd.DataFrame:
    """ms: rows from `market_series` -> one row per date."""
    vix = ms[ms["symbol"] == "^VIX"].set_index("date")["close"].sort_index()
    spy = ms[ms["symbol"] == "SPY"].set_index("date")["adj_close"].sort_index()
    spy_rv21 = np.log(spy / spy.shift(1)).rolling(21).std() * ANN
    out = pd.DataFrame({"vix": vix})
    out["vix_pct_252"] = vix.rolling(252, min_periods=126).rank(pct=True)
    out["vix_chg_5"] = vix / vix.shift(5) - 1
    out = out.join(spy_rv21.rename("spy_rv21"), how="left")
    out["vrp"] = out["vix"] / 100 - out["spy_rv21"]
    return out.drop(columns=["spy_rv21"]).reset_index().rename(columns={"index": "date"})


def stock_iv(iv: pd.DataFrame) -> pd.DataFrame:
    """iv: rows from `iv_snapshots` -> per (ticker, date) IV features (without the RV spread)."""
    if iv.empty:
        return pd.DataFrame(columns=["ticker", "date", "iv30_atm", "iv_rank_252", "skew30",
                                     "term_slope", "pc_vol_ratio", "pc_oi_ratio"])
    iv = iv.sort_values(["ticker", "date"]).copy()
    g = iv.groupby("ticker")["iv30_atm"]
    lo = g.transform(lambda s: s.rolling(252, min_periods=IV_RANK_MIN_OBS).min())
    hi = g.transform(lambda s: s.rolling(252, min_periods=IV_RANK_MIN_OBS).max())
    iv["iv_rank_252"] = ((iv["iv30_atm"] - lo) / (hi - lo)).where(hi > lo)
    return iv[["ticker", "date", "iv30_atm", "iv_rank_252", "skew30", "term_slope",
               "pc_vol_ratio", "pc_oi_ratio"]]


def attach(features: pd.DataFrame, ms: pd.DataFrame, iv: pd.DataFrame) -> pd.DataFrame:
    """Left-join market and per-stock IV features onto candle features."""
    out = features.merge(market(ms), on="date", how="left")
    out = out.merge(stock_iv(iv), on=["ticker", "date"], how="left")
    out["iv_rv_spread"] = out["iv30_atm"] - out["rv_cc_21"]
    # DCA sizing input: per-stock IV rank when available, else market VIX percentile
    out["vol_rank"] = out["iv_rank_252"].fillna(out["vix_pct_252"])
    return out
