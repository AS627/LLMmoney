"""Market-wide daily series from yfinance: VIX (market IV), SPY (benchmark), ^IRX (T-bill rate).

Symbols are fetched one at a time: yfinance's threaded multi-download can fail with
"database is locked" on its internal timezone cache.
"""
from __future__ import annotations

import logging

import duckdb
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

MARKET_SYMBOLS = ["^VIX", "SPY", "^IRX"]
COLS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=30), reraise=True)
def _history(symbol: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(symbol).history(start=start, auto_adjust=False, actions=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")
    return df


def history_frame(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """yfinance history -> rows for `market_series`."""
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.DataFrame({
        "symbol": symbol,
        "date": idx.date,
        "open": df["Open"].to_numpy(),
        "high": df["High"].to_numpy(),
        "low": df["Low"].to_numpy(),
        "close": df["Close"].to_numpy(),
        "adj_close": df.get("Adj Close", df["Close"]).to_numpy(),
        "volume": pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64").to_numpy(),
    })
    return out.dropna(subset=["close"]).drop_duplicates(subset=["symbol", "date"], keep="last")[COLS]


def ingest(con: duckdb.DuckDBPyConnection, symbols: list[str] | None = None, start: str = "1990-01-01") -> dict[str, int]:
    counts: dict[str, int] = {}
    for sym in symbols or MARKET_SYMBOLS:
        df = history_frame(sym, _history(sym, start))
        con.execute("DELETE FROM market_series WHERE symbol = ?", [sym])
        con.register("incoming", df)
        con.execute(f"INSERT INTO market_series ({', '.join(COLS)}) SELECT {', '.join(COLS)} FROM incoming")
        con.unregister("incoming")
        counts[sym] = len(df)
        log.info("%s: %d rows %s → %s", sym, len(df), df["date"].min(), df["date"].max())
    return counts
