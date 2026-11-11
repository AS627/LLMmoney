"""Dollar-cost-averaging backtest driven by the logistic signals and volatility regime.

Each month (first trading day) the long-horizon classifier picks the top N tickers with
out-of-sample P(hit +40% before -30%) >= tau. Each pick's budget is bought in K tranches
spaced `spacing` trading days apart. Strategies compared on the same picks:

  lump         whole budget on day 0
  dca          equal tranches
  signal_dca   tranche k is scaled by m_k = clip(1 + a*s_k + b*(vol_rank_k - 0.5), lo, hi)
               s_k = P(up) - P(down) from the 5-day classifier on that tranche's day
               vol_rank_k = per-stock IV rank, or VIX 1y percentile before IV history exists
               Sized sequentially with no lookahead: spend min(remaining, base*m_k); the
               last tranche spends whatever is left, so total spend equals the budget.
  universe_dca equal-tranche DCA into *every* ticker (no classifier) - shows model value

Exit (notebook rule): first day the midpoint (H+L)/2 is >= up*entry or <= down*entry, sold
at that day's close; otherwise sold at the close `horizon` trading days after entry.
Tranches scheduled on or after the exit day are not bought.

Portfolio: capital is split into 12 sleeves; the cohort entered in month m uses sleeve
m % 12 and holds until exit or the sleeve's next turn, so returns compound realistically.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from llmmoney.features.candles import adjust_ohlc

STRATEGIES = ("lump", "dca", "signal_dca", "universe_dca")


@dataclass(frozen=True)
class DCAConfig:
    top_n: int = 5
    tau: float = 0.35          # minimum calibrated P(hit) to enter
    tranches: int = 4
    spacing: int = 5           # trading days between tranches (weekly)
    a: float = 1.0             # weight on short-horizon signal
    b: float = 1.0             # weight on volatility rank (buy more when vol is high)
    m_lo: float = 0.25
    m_hi: float = 2.0
    up: float = 1.4
    down: float = 0.7
    horizon: int = 252
    cost_bps: float = 5.0


def _load(con: duckdb.DuckDBPyConnection):
    prices = adjust_ohlc(con.execute("SELECT * FROM prices").df())
    prices["date"] = pd.to_datetime(prices["date"])
    prices["mid"] = (prices["a_high"] + prices["a_low"]) / 2
    sig = con.execute("SELECT ticker, date, model, p_down, p_up FROM signals "
                      "WHERE model IN ('logistic_long', 'logistic_short')").df()
    sig["date"] = pd.to_datetime(sig["date"])
    longs = sig[sig["model"] == "logistic_long"][["ticker", "date", "p_up"]].rename(columns={"p_up": "p_hit"})
    shorts = sig[sig["model"] == "logistic_short"].assign(s=lambda d: d["p_up"] - d["p_down"])[["ticker", "date", "s"]]
    vr = con.execute("SELECT ticker, date, vol_rank FROM features_daily").df()
    vr["date"] = pd.to_datetime(vr["date"])
    spy = con.execute("SELECT date, adj_close FROM market_series WHERE symbol = 'SPY'").df()
    spy["date"] = pd.to_datetime(spy["date"])
    return prices, longs, shorts, vr, spy.set_index("date")["adj_close"].sort_index()


def simulate_position(close: np.ndarray, mid: np.ndarray, i0: int, i_end: int, weights: np.ndarray | None,
                      mult: np.ndarray | None, cfg: DCAConfig) -> tuple[np.ndarray, dict]:
    """Value path (per $1 budget) from i0..i_end for one pick.

    weights: fixed tranche fractions (lump / dca). mult: per-tranche multipliers m_k
    (signal_dca, sequential sizing). Exactly one of them is given."""
    entry = mid[i0]
    exit_i = i_end
    for j in range(i0 + 1, i_end + 1):
        if mid[j] >= cfg.up * entry or mid[j] <= cfg.down * entry:
            exit_i = j
            break
    k_n = len(weights) if weights is not None else len(mult)
    tranche_days = [i0 + k * cfg.spacing for k in range(k_n)]
    cost = cfg.cost_bps / 1e4
    cash, shares, spent_px, spent = 1.0, 0.0, 0.0, 0.0
    remaining, base = 1.0, 1.0 / k_n
    buys = {}
    for k, t in enumerate(tranche_days):
        if t >= exit_i and k > 0:
            break
        if weights is not None:
            amt = weights[k]
        else:
            last = k == k_n - 1 or tranche_days[k + 1] >= exit_i
            amt = remaining if last else min(remaining, base * mult[k])
        remaining -= amt
        buys[t] = amt
    path = np.empty(i_end - i0 + 1)
    for idx, t in enumerate(range(i0, i_end + 1)):
        if t in buys and t <= exit_i:
            amt = buys[t]
            shares += amt * (1 - cost) / close[t]
            cash -= amt
            spent += amt
            spent_px += amt * close[t]
        if t < exit_i:
            path[idx] = cash + shares * close[t]
        else:
            if t == exit_i:
                cash += shares * close[t] * (1 - cost)
                shares = 0.0
            path[idx] = cash
    info = {"ret": path[-1] - 1, "avg_cost": spent_px / spent if spent else np.nan,
            "invested": spent, "exit_days": exit_i - i0}
    return path, info


def run(con: duckdb.DuckDBPyConnection, cfg: DCAConfig = DCAConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices, longs, shorts, vr, spy = _load(con)
    by_ticker = {t: g.sort_values("date").reset_index(drop=True) for t, g in prices.groupby("ticker")}
    s_map = shorts.set_index(["ticker", "date"])["s"]
    v_map = vr.set_index(["ticker", "date"])["vol_rank"]
    calendar = pd.DatetimeIndex(sorted(prices["date"].unique()))
    cohorts = sorted(longs["date"].unique())
    month_ix = {d: i for i, d in enumerate(cohorts)}

    nav_parts = {s: [] for s in STRATEGIES}
    trades = []
    for d in cohorts:
        sleeve_next = cohorts[month_ix[d] + 12] if month_ix[d] + 12 < len(cohorts) else calendar[-1]
        day = longs[longs["date"] == d].sort_values("p_hit", ascending=False)
        picks = day[day["p_hit"] >= cfg.tau].head(cfg.top_n)["ticker"].tolist()
        universe = day["ticker"].tolist()
        cal = calendar[(calendar >= d) & (calendar <= sleeve_next)]
        for strat in STRATEGIES:
            names = universe if strat == "universe_dca" else picks
            paths = []
            for t in names:
                g = by_ticker[t]
                pos = g.index[g["date"] == d]
                if pos.empty:
                    continue
                i0 = int(pos[0])
                i_end = min(i0 + cfg.horizon, int(g.index[g["date"] <= sleeve_next][-1]))
                if i_end <= i0:
                    continue
                close, mid = g["a_close"].to_numpy(), g["mid"].to_numpy()
                if strat == "lump":
                    path, info = simulate_position(close, mid, i0, i_end, np.array([1.0]), None, cfg)
                elif strat in ("dca", "universe_dca"):
                    path, info = simulate_position(close, mid, i0, i_end, np.full(cfg.tranches, 1 / cfg.tranches), None, cfg)
                else:
                    mult = []
                    for k in range(cfg.tranches):
                        tk = min(i0 + k * cfg.spacing, len(g) - 1)
                        key = (t, g["date"].iloc[tk])
                        s = s_map.get(key, 0.0)
                        v = v_map.get(key, 0.5)
                        s = 0.0 if pd.isna(s) else s
                        v = 0.5 if pd.isna(v) else v
                        mult.append(np.clip(1 + cfg.a * s + cfg.b * (v - 0.5), cfg.m_lo, cfg.m_hi))
                    path, info = simulate_position(close, mid, i0, i_end, None, np.array(mult), cfg)
                series = pd.Series(path, index=g["date"].iloc[i0:i_end + 1].to_numpy()).reindex(cal).ffill().fillna(1.0)
                paths.append(series)
                if strat != "universe_dca":
                    trades.append({"cohort": d, "ticker": t, "strategy": strat, **info})
            cohort_val = pd.concat(paths, axis=1).mean(axis=1) if paths else pd.Series(1.0, index=cal)
            nav_parts[strat].append((month_ix[d] % 12, cohort_val))

    nav = pd.DataFrame({s: _sleeve_nav(nav_parts[s], calendar) for s in STRATEGIES})
    nav = nav.loc[nav.index >= cohorts[0]]
    nav["spy"] = spy.reindex(nav.index).ffill() / spy.reindex(nav.index).ffill().iloc[0]
    trades_df = pd.DataFrame(trades)
    return _summary(nav, trades_df), nav, trades_df


def _sleeve_nav(parts: list[tuple[int, pd.Series]], calendar: pd.DatetimeIndex) -> pd.Series:
    """Chain each sleeve's cohorts multiplicatively; sum 12 sleeves of 1/12 each."""
    start = min(p[1].index[0] for p in parts)
    cal = calendar[calendar >= start]
    total = pd.Series(0.0, index=cal)
    for sleeve in range(12):
        val = pd.Series(np.nan, index=cal)
        level = 1 / 12
        for _, cohort in (p for p in parts if p[0] == sleeve):
            seg = cohort / cohort.iloc[0] * level
            val.loc[seg.index] = seg.to_numpy()
            level = float(seg.iloc[-1])
        total = total.add(val.ffill().fillna(1 / 12), fill_value=0)
    return total


def _summary(nav: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    for col in nav.columns:
        v = nav[col]
        r = v.pct_change().dropna()
        row = {
            "CAGR": v.iloc[-1] ** (1 / years) - 1,
            "vol": r.std() * np.sqrt(252),
            "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "max_dd": (v / v.cummax() - 1).min(),
        }
        t = trades[trades["strategy"] == col] if not trades.empty else trades
        if len(t):
            row["trades"] = len(t)
            row["hit_rate"] = (t["ret"] > 0).mean()
            row["mean_trade_ret"] = t["ret"].mean()
        rows[col] = row
    out = pd.DataFrame(rows).T
    if not trades.empty:
        piv = trades.pivot_table(index=["cohort", "ticker"], columns="strategy", values="avg_cost")
        for s in ("lump", "signal_dca"):
            if s in piv and "dca" in piv:
                out.loc[s, "cost_vs_dca"] = (piv[s] / piv["dca"] - 1).mean()
    return out
