"""Long-horizon barrier label, ported from research/max_logistic.ipynb (get_current_state).

For an entry on day t at the midpoint price m_t = (high + low) / 2:
    y = 1  if the midpoint reaches  up * m_t  strictly before it reaches  down * m_t
           within the next `horizon` trading days,
    y = 0  otherwise (down first, both on the same day, or neither).
The notebook's formula y = max_w1 * (1 - max_w2 * (1 - w3)) with w3 = [first up < first down]
is exactly this. Same-day hits count as 0, as in the notebook.

Unlike the notebook, prices are split-adjusted and windows are counted in trading days.
A label is "known" once a barrier is hit or the full window has elapsed; rows whose
window is still open with no hit get NaN so they are never trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llmmoney.features.candles import adjust_ohlc


def label_series(mid: np.ndarray, starts: np.ndarray, up: float, down: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (label, days_to_outcome) for each start index into `mid`."""
    n = len(mid)
    y = np.full(len(starts), np.nan)
    days = np.full(len(starts), np.nan)
    for k, i in enumerate(starts):
        entry = mid[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        w = mid[i + 1 : i + 1 + horizon]
        hit_up = np.flatnonzero(w >= up * entry)
        hit_dn = np.flatnonzero(w <= down * entry)
        first_up = hit_up[0] if hit_up.size else np.inf
        first_dn = hit_dn[0] if hit_dn.size else np.inf
        if np.isfinite(first_up) or np.isfinite(first_dn):
            y[k] = 1.0 if first_up < first_dn else 0.0
            days[k] = min(first_up, first_dn) + 1
        elif i + horizon < n:  # full window observed, no hit
            y[k] = 0.0
            days[k] = horizon
    return y, days


def month_starts(dates: pd.Series) -> np.ndarray:
    """Positions of the first trading day of each month."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    first = d.groupby([d.dt.year, d.dt.month]).head(1)
    return first.index.to_numpy()


def build(prices: pd.DataFrame, up: float = 1.4, down: float = 0.7, horizon: int = 252,
          monthly: bool = True) -> pd.DataFrame:
    px = adjust_ohlc(prices)
    rows = []
    for t, g in px.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        mid = ((g["a_high"] + g["a_low"]) / 2).to_numpy()
        starts = month_starts(g["date"]) if monthly else np.arange(len(g))
        y, days = label_series(mid, starts, up, down, horizon)
        rows.append(pd.DataFrame({"ticker": t, "date": g["date"].iloc[starts].to_numpy(),
                                  "y_barrier": y, "barrier_days": days}))
    return pd.concat(rows, ignore_index=True)
