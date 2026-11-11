"""Short-horizon label: h-day return in excess of SPY, bucketed into down / flat / up.

Uses adjusted closes from t to t+h trading days of the stock's own calendar, and SPY's
return over the same calendar dates. Classes: 0 = down, 1 = flat, 2 = up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DOWN, FLAT, UP = 0, 1, 2


def build(prices: pd.DataFrame, market: pd.DataFrame, horizon: int = 5, flat_band: float = 0.01) -> pd.DataFrame:
    spy = (market[market["symbol"] == "SPY"].set_index("date")["adj_close"].sort_index())
    out = []
    for t, g in prices.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        end_date = g["date"].shift(-horizon)
        r = g["adj_close"].shift(-horizon) / g["adj_close"] - 1
        spy_r = end_date.map(spy).to_numpy() / g["date"].map(spy).to_numpy() - 1
        out.append(pd.DataFrame({"ticker": t, "date": g["date"], "fwd_ret": r,
                                 "fwd_excess": r.to_numpy() - spy_r, "outcome_date": end_date}))
    df = pd.concat(out, ignore_index=True)
    x = df["fwd_excess"]
    df["y_fwd"] = np.where(x.isna(), np.nan, np.where(x > flat_band, UP, np.where(x < -flat_band, DOWN, FLAT)))
    return df
