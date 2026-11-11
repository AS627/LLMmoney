"""Daily option-chain recorder (yfinance) -> per-stock implied-volatility features.

yfinance only serves *current* chains, so this must run every trading day after the
close; history accumulates from the first run. Raw chains are kept as Parquet under
data/options/YYYY-MM-DD/ so features can be recomputed with better logic later.

Feature definitions (all from yfinance's per-contract `impliedVolatility`):
  iv30_atm / iv60_atm  ATM IV interpolated to 30 / 60 calendar days
  skew30               IV of 0.9*spot puts minus IV of 1.1*spot calls, at ~30d
  term_slope           iv60_atm - iv30_atm
  pc_vol_ratio         put volume / call volume, expiries within 60d
  pc_oi_ratio          put open interest / call open interest, expiries within 60d
Only contracts with a positive bid and a sane IV are used: deep-ITM and stale quotes
produce meaningless IVs.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from llmmoney.config.settings import get_settings

log = logging.getLogger(__name__)

MAX_DTE = 75          # fetch expiries out to this many days (covers the 60d point)
MIN_IV, MAX_IV = 0.01, 5.0
SNAPSHOT_COLS = ["ticker", "date", "asof_ts", "spot", "iv30_atm", "iv60_atm", "skew30",
                 "term_slope", "pc_vol_ratio", "pc_oi_ratio", "n_contracts"]


@dataclass
class Chain:
    """All contracts for one ticker: columns type(C/P), expiry, dte, strike, iv, bid, volume, open_interest."""
    ticker: str
    spot: float
    asof: datetime
    contracts: pd.DataFrame


# ---------- pure feature maths (unit-tested) ----------

def _clean(c: pd.DataFrame) -> pd.DataFrame:
    return c[(c["bid"] > 0) & c["iv"].between(MIN_IV, MAX_IV)]


def _iv_at_strike(side: pd.DataFrame, k: float) -> float:
    """Linear interpolation of IV across strikes at strike k (nan if k is outside the quoted range)."""
    side = side.sort_values("strike")
    if side.empty or k < side["strike"].iloc[0] or k > side["strike"].iloc[-1]:
        return np.nan
    return float(np.interp(k, side["strike"], side["iv"]))


def _per_expiry(c: pd.DataFrame, spot: float) -> pd.DataFrame:
    """ATM IV and skew for each expiry, using out-of-the-money quotes on each side."""
    rows = []
    for (expiry, dte), g in c.groupby(["expiry", "dte"]):
        calls, puts = g[g["type"] == "C"], g[g["type"] == "P"]
        # ATM: average of call and put IV interpolated at spot
        sides = [v for v in (_iv_at_strike(calls, spot), _iv_at_strike(puts, spot)) if np.isfinite(v)]
        atm = float(np.mean(sides)) if sides else np.nan
        skew = _iv_at_strike(puts, 0.9 * spot) - _iv_at_strike(calls, 1.1 * spot)
        rows.append({"expiry": expiry, "dte": dte, "atm": atm, "skew": skew})
    return pd.DataFrame(rows, columns=["expiry", "dte", "atm", "skew"])


def _interp_dte(per_exp: pd.DataFrame, col: str, target: int) -> float:
    """Interpolate a per-expiry value to `target` days in total-variance space for IV, linearly otherwise.

    Falls back to the nearest expiry when target is not bracketed."""
    d = per_exp.dropna(subset=[col]).sort_values("dte")
    d = d[d["dte"] > 0]
    if d.empty:
        return np.nan
    below, above = d[d["dte"] <= target], d[d["dte"] >= target]
    if below.empty or above.empty:
        return float(d.iloc[(d["dte"] - target).abs().argmin()][col])
    lo, hi = below.iloc[-1], above.iloc[0]
    if lo["dte"] == hi["dte"]:
        return float(lo[col])
    w = (target - lo["dte"]) / (hi["dte"] - lo["dte"])
    if col == "atm":  # interpolate total variance sigma^2 * t, then convert back
        var = (1 - w) * lo[col] ** 2 * lo["dte"] + w * hi[col] ** 2 * hi["dte"]
        return float(np.sqrt(var / target))
    return float((1 - w) * lo[col] + w * hi[col])


def compute_features(chain: Chain) -> dict:
    raw = chain.contracts
    c = _clean(raw)
    per_exp = _per_expiry(c, chain.spot)
    iv30 = _interp_dte(per_exp, "atm", 30)
    iv60 = _interp_dte(per_exp, "atm", 60)
    near = raw[raw["dte"] <= 60]
    def ratio(col: str) -> float:
        p = near.loc[near["type"] == "P", col].fillna(0).sum()
        k = near.loc[near["type"] == "C", col].fillna(0).sum()
        return float(p / k) if k > 0 else np.nan
    return {
        "ticker": chain.ticker,
        "date": chain.asof.date(),
        "asof_ts": chain.asof.replace(tzinfo=None),
        "spot": chain.spot,
        "iv30_atm": iv30,
        "iv60_atm": iv60,
        "skew30": _interp_dte(per_exp, "skew", 30),
        "term_slope": iv60 - iv30,
        "pc_vol_ratio": ratio("volume"),
        "pc_oi_ratio": ratio("open_interest"),
        "n_contracts": int(len(c)),
    }


# ---------- fetching ----------

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=30), reraise=True)
def fetch_chain(ticker: str, max_dte: int = MAX_DTE) -> Chain:
    import yfinance as yf

    yt = yf.Ticker(ticker.replace(".", "-"))  # yfinance uses BRK-B
    asof = datetime.now(timezone.utc)
    today = asof.date()
    frames, spot = [], np.nan
    for exp in yt.options:
        dte = (date.fromisoformat(exp) - today).days
        if dte < 1 or dte > max_dte:
            continue
        oc = yt.option_chain(exp)
        spot = oc.underlying.get("regularMarketPrice", spot) if getattr(oc, "underlying", None) else spot
        for kind, df in (("C", oc.calls), ("P", oc.puts)):
            frames.append(pd.DataFrame({
                "type": kind, "expiry": exp, "dte": dte,
                "strike": df["strike"], "iv": df["impliedVolatility"], "bid": df["bid"],
                "volume": df["volume"], "open_interest": df["openInterest"],
            }))
    if not frames or not np.isfinite(spot):
        raise RuntimeError(f"no usable option chain for {ticker}")
    return Chain(ticker, float(spot), asof, pd.concat(frames, ignore_index=True))


def record(con: duckdb.DuckDBPyConnection, tickers: list[str], pause_s: float = 1.0) -> dict[str, int]:
    """Fetch chains, save raw Parquet, upsert one iv_snapshots row per ticker. Failures are logged and skipped."""
    out_dir = get_settings().data_dir / "options"
    rows, failed = [], []
    for t in tickers:
        try:
            chain = fetch_chain(t)
        except Exception as e:
            log.warning("options %s failed: %s", t, e)
            failed.append(t)
            continue
        day_dir = out_dir / chain.asof.date().isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        chain.contracts.assign(spot=chain.spot, asof=chain.asof).to_parquet(day_dir / f"{t}.parquet", index=False)
        rows.append(compute_features(chain))
        time.sleep(pause_s)  # be gentle with Yahoo's rate limits
    if rows:
        df = pd.DataFrame(rows)[SNAPSHOT_COLS]
        con.register("incoming", df)
        con.execute("""
            DELETE FROM iv_snapshots USING incoming
            WHERE iv_snapshots.ticker = incoming.ticker AND iv_snapshots.date = incoming.date
        """)
        con.execute(f"INSERT INTO iv_snapshots ({', '.join(SNAPSHOT_COLS)}) SELECT {', '.join(SNAPSHOT_COLS)} FROM incoming")
        con.unregister("incoming")
    if failed:
        log.warning("failed tickers: %s", ", ".join(failed))
    return {"recorded": len(rows), "failed": len(failed)}
