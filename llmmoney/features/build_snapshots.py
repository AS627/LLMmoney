"""Assemble the daily feature table and both label tables in DuckDB.

Tables written (replaced on each run):
  features_daily   candle + volatility-regime + fundamental features per (ticker, date)
  labels_barrier   +40% / -30% within 252 trading days, monthly entries (notebook target)
  labels_fwd       5-day excess return vs SPY, bucketed down / flat / up
"""
from __future__ import annotations

import duckdb
import pandas as pd

from llmmoney.config.settings import get_settings
from llmmoney.features import candles, fundamentals, volatility
from llmmoney.labels import barrier, forward_returns


def _replace(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    con.register("incoming", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM incoming")
    con.unregister("incoming")


def run(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    s = get_settings()
    prices = con.execute("SELECT * FROM prices ORDER BY ticker, date").df()
    ms = con.execute("SELECT * FROM market_series").df()
    iv = con.execute("SELECT * FROM iv_snapshots").df()
    fund = con.execute("SELECT * FROM fundamentals").df()
    for df in (prices, ms, iv):
        df["date"] = pd.to_datetime(df["date"]).dt.date

    feats = candles.build(prices)
    feats = volatility.attach(feats, ms, iv)
    feats = feats.merge(fundamentals.build(fund, prices), on=["ticker", "date"], how="left")
    _replace(con, "features_daily", feats)

    lb = barrier.build(prices)
    _replace(con, "labels_barrier", lb)
    lf = forward_returns.build(prices, ms, horizon=s.horizon_days, flat_band=s.flat_band)
    _replace(con, "labels_fwd", lf)
    return {"features_daily": len(feats), "feature_cols": feats.shape[1] - 2,
            "labels_barrier_known": int(lb["y_barrier"].notna().sum()),
            "labels_fwd_known": int(lf["y_fwd"].notna().sum())}
