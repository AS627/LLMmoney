"""Daily candlestick features, one row per (ticker, date).

Every feature at date t uses only rows with date <= t (rolling windows / shifts into the
past), so a row is computable at t's close. OHLC are split/dividend-adjusted with the
factor adj_close/close so multi-day comparisons are not broken by splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = np.sqrt(252)
RET_WINDOWS = (1, 5, 21, 63, 252)
VOL_WINDOWS = (10, 21, 63)


def adjust_ohlc(px: pd.DataFrame) -> pd.DataFrame:
    """Add adjusted open/high/low/close columns (a_open, a_high, a_low, a_close)."""
    f = (px["adj_close"] / px["close"]).where(px["close"] > 0)
    out = px.copy()
    for col in ("open", "high", "low", "close"):
        out[f"a_{col}"] = px[col] * f
    return out


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def _one_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date")
    o, h, l, c, v = g["a_open"], g["a_high"], g["a_low"], g["a_close"], g["volume"].astype(float)
    f = pd.DataFrame(index=g.index)
    f["ticker"], f["date"] = g["ticker"], g["date"]

    # returns and position vs 52-week range
    for n in RET_WINDOWS:
        f[f"ret_{n}"] = c.pct_change(n, fill_method=None)
    f["dist_52w_high"] = c / h.rolling(252, min_periods=126).max() - 1
    f["dist_52w_low"] = c / l.rolling(252, min_periods=126).min() - 1
    f["gap"] = o / c.shift(1) - 1

    # candle shape
    rng = (h - l).replace(0, np.nan)
    f["range_pct"] = (h - l) / c
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["close_pos"] = (c - l) / rng

    # realized volatility (annualized)
    lr = np.log(c / c.shift(1))
    hl2 = np.log(h / l) ** 2
    co2 = np.log(c / o) ** 2
    for n in VOL_WINDOWS:
        f[f"rv_cc_{n}"] = lr.rolling(n, min_periods=n).std() * ANN
        f[f"rv_park_{n}"] = np.sqrt(hl2.rolling(n, min_periods=n).mean() / (4 * np.log(2))) * ANN
        gk = (0.5 * hl2 - (2 * np.log(2) - 1) * co2).rolling(n, min_periods=n).mean()
        f[f"rv_gk_{n}"] = np.sqrt(gk.clip(lower=0)) * ANN
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f["atr14_pct"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c

    # volume
    lv = np.log1p(v)
    ldv = np.log1p(v * g["close"])
    f["vol_z21"] = (lv - lv.rolling(21).mean()) / lv.rolling(21).std()
    f["dvol_z21"] = (ldv - ldv.rolling(21).mean()) / ldv.rolling(21).std()

    # momentum
    f["rsi14"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    return f


def build(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: rows from the `prices` table -> candle features."""
    px = adjust_ohlc(prices)
    parts = [_one_ticker(g) for _, g in px.groupby("ticker", sort=False)]
    out = pd.concat(parts, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan)
