"""Load SimFin bulk data (quarterly fundamentals + daily share prices) into DuckDB.

SimFin bulk downloads cover the whole US market; we cache them under data/simfin and
filter to the requested tickers before writing. Banks and insurers use separate
statement templates, so each statement is the union of its general/banks/insurance
datasets.

Point-in-time note: `publish_date` is when the filing became public and is what
feature joins must use. SimFin values may have been restated later
(`restated_date`); that column is kept so features can be conservative about it.
"""
from __future__ import annotations

import logging
from typing import Iterable

import duckdb
import pandas as pd

from llmmoney.config.settings import get_settings

log = logging.getLogger(__name__)

STATEMENTS = {
    "income": ["income", "income-banks", "income-insurance"],
    "balance": ["balance", "balance-banks", "balance-insurance"],
    "cashflow": ["cashflow", "cashflow-banks", "cashflow-insurance"],
}
ID_COLS = ["Ticker", "SimFinId", "Currency", "Fiscal Year", "Fiscal Period",
           "Report Date", "Publish Date", "Restated Date"]

# Our ticker -> SimFin's ticker, for share classes SimFin lists differently. Data is stored
# under our ticker. BRK-A prices differ in level from BRK.B but returns and per-share
# ratios are consistent, which is all the features use.
SIMFIN_ALIASES = {"BRK.B": "BRK-A", "GOOGL": "GOOG"}

PRICE_COLS = {
    "Ticker": "ticker", "Date": "date", "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Adj. Close": "adj_close", "Volume": "volume",
    "Shares Outstanding": "shares_out",
}


def _setup_simfin():
    import simfin as sf

    s = get_settings()
    if not s.simfin_api_key:
        raise RuntimeError("SIMFIN_API_KEY is not set (add it to .env)")
    sf.set_api_key(s.simfin_api_key)
    s.simfin_dir.mkdir(parents=True, exist_ok=True)
    sf.set_data_dir(str(s.simfin_dir))
    return sf


def _load_dataset(sf, dataset: str, refresh_days: int) -> pd.DataFrame | None:
    try:
        df = sf.load(dataset=dataset, variant="quarterly", market="us", index=None,
                      parse_dates=["Report Date", "Publish Date", "Restated Date"],
                      refresh_days=refresh_days)
    except Exception as e:  # dataset may be unavailable on the account's tier
        log.warning("SimFin dataset %s unavailable: %s", dataset, e)
        return None
    return df


def fundamentals_long(df: pd.DataFrame, statement: str) -> pd.DataFrame:
    """Wide SimFin statement frame -> long rows matching the `fundamentals` table."""
    id_cols = [c for c in ID_COLS if c in df.columns]
    value_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="item", value_name="value")
    long = long.dropna(subset=["value", "Publish Date"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    out = pd.DataFrame({
        "ticker": long["Ticker"],
        "statement": statement,
        "fiscal_year": long["Fiscal Year"].astype(int),
        "fiscal_period": long["Fiscal Period"],
        "report_date": long["Report Date"].dt.date,
        "publish_date": long["Publish Date"].dt.date,
        "restated_date": long["Restated Date"].dt.date if "Restated Date" in long else None,
        "item": long["item"],
        "value": long["value"].astype(float),
    })
    return out.drop_duplicates(subset=["ticker", "statement", "fiscal_year", "fiscal_period", "item"], keep="last")


def prices_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[c for c in PRICE_COLS if c in df.columns]].rename(columns=PRICE_COLS)
    for col in PRICE_COLS.values():
        if col not in out:
            out[col] = None
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("Int64")
    return out[list(PRICE_COLS.values())].drop_duplicates(subset=["ticker", "date"], keep="last")


def _upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, tickers: list[str]) -> int:
    """Replace all rows for `tickers` in `table` with `df` (idempotent reloads)."""
    con.execute(f"DELETE FROM {table} WHERE ticker IN (SELECT UNNEST(?))", [tickers])
    con.register("incoming", df)
    cols = ", ".join(df.columns)
    con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM incoming")
    con.unregister("incoming")
    return len(df)


def ingest(con: duckdb.DuckDBPyConnection, tickers: Iterable[str], refresh_days: int = 1) -> dict[str, int]:
    """Download (or reuse cached) SimFin bulk data and write the given tickers to DuckDB."""
    sf = _setup_simfin()
    tickers = sorted(set(tickers))
    to_ours = {SIMFIN_ALIASES.get(t, t): t for t in tickers}
    counts: dict[str, int] = {}

    def select(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Ticker"].isin(to_ours)].copy()
        df["Ticker"] = df["Ticker"].map(to_ours)
        return df

    frames = []
    for statement, datasets in STATEMENTS.items():
        for ds in datasets:
            df = _load_dataset(sf, ds, refresh_days)
            if df is None:
                continue
            df = select(df)
            if not df.empty:
                frames.append(fundamentals_long(df, statement))
    fund = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    counts["fundamentals"] = _upsert(con, "fundamentals", fund, tickers) if not fund.empty else 0

    px = sf.load_shareprices(variant="daily", market="us", index=None, refresh_days=refresh_days)
    px = prices_frame(select(px))
    counts["prices"] = _upsert(con, "prices", px, tickers)

    found = set(px["ticker"]) | (set(fund["ticker"]) if not fund.empty else set())
    counts["missing_tickers"] = len(set(tickers) - found)
    if set(tickers) - found:
        log.warning("No SimFin data for: %s", ", ".join(sorted(set(tickers) - found)))
    return counts
