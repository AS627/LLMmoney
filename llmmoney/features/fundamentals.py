"""Point-in-time fundamental ratios (the notebook's debt / equity / sales / FCF, scaled by price).

Each quarter becomes usable on its publish_date (as-of join: the latest quarter published
on or before t). Ratios are taken against the market cap at t, which makes them
comparable across companies and over time, unlike the notebook's raw per-share values.
TTM sums need 4 quarters; companies with thin SimFin history get NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ITEMS = {
    "Revenue": "revenue", "Net Income": "net_income", "Shares (Diluted)": "shares",
    "Total Equity": "equity", "Long Term Debt": "lt_debt", "Short Term Debt": "st_debt",
    "Net Cash from Operating Activities": "cfo", "Change in Fixed Assets & Intangibles": "capex",
}
OUT_COLS = ["sales_yield", "earnings_yield", "fcf_yield", "book_to_price", "debt_to_equity", "fund_age_days"]


def quarterly(fund: pd.DataFrame) -> pd.DataFrame:
    """Long fundamentals -> one row per (ticker, quarter) with TTM flows, latest stocks."""
    f = fund[fund["item"].isin(ITEMS)]
    q = (f.pivot_table(index=["ticker", "fiscal_year", "fiscal_period"], columns="item",
                       values="value", aggfunc="last")
          .rename(columns=ITEMS).reset_index())
    pub = f.groupby(["ticker", "fiscal_year", "fiscal_period"])["publish_date"].max().reset_index()
    q = q.merge(pub, on=["ticker", "fiscal_year", "fiscal_period"])
    for c in ITEMS.values():
        if c not in q:
            q[c] = np.nan
    q = q.sort_values(["ticker", "fiscal_year", "fiscal_period"])
    q["fcf"] = q["cfo"] + q["capex"].fillna(0)  # capex is reported negative
    for c in ("revenue", "net_income", "fcf"):
        q[f"{c}_ttm"] = q.groupby("ticker")[c].transform(lambda s: s.rolling(4, min_periods=4).sum())
    q["debt"] = q[["lt_debt", "st_debt"]].sum(axis=1, min_count=1)
    return q


def build(fund: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """As-of join quarterly fundamentals onto daily prices -> ratio features per (ticker, date)."""
    if fund.empty:
        return prices[["ticker", "date"]].assign(**{c: np.nan for c in OUT_COLS})
    q = quarterly(fund)
    q["publish_date"] = pd.to_datetime(q["publish_date"]).astype("datetime64[ns]")
    px = prices[["ticker", "date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
    m = pd.merge_asof(px.sort_values("date"), q.sort_values("publish_date"),
                      left_on="date", right_on="publish_date", by="ticker", direction="backward")
    mcap = (m["shares"] * m["close"]).where(lambda s: s > 0)
    out = pd.DataFrame({"ticker": m["ticker"], "date": m["date"].dt.date})
    out["sales_yield"] = m["revenue_ttm"] / mcap
    out["earnings_yield"] = m["net_income_ttm"] / mcap
    out["fcf_yield"] = m["fcf_ttm"] / mcap
    out["book_to_price"] = m["equity"] / mcap
    out["debt_to_equity"] = (m["debt"] / m["equity"]).where(m["equity"] > 0)
    out["fund_age_days"] = (m["date"] - m["publish_date"]).dt.days
    return out.replace([np.inf, -np.inf], np.nan)
