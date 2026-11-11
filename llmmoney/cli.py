"""Command-line entry point: `llmmoney <command>`."""
from __future__ import annotations

import typer

app = typer.Typer(help="LLMmoney: market signals + research assistant.", no_args_is_help=True)


@app.command("init-db")
def init_db() -> None:
    """Create the DuckDB file and tables."""
    from llmmoney.storage.db import connect

    with connect() as con:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    typer.echo(f"DB ready with tables: {', '.join(tables)}")


def _todo(milestone: str) -> None:
    typer.echo(f"Not implemented yet (milestone: {milestone}).")
    raise typer.Exit(1)


@app.command("ingest-simfin")
def ingest_simfin(tickers: str = typer.Option("", help="Comma-separated subset; default = universe")) -> None:
    """Bulk-load SimFin fundamentals and daily prices."""
    import logging

    from llmmoney.config.settings import load_universe
    from llmmoney.ingest.simfin_loader import ingest
    from llmmoney.storage.db import connect

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    selected = [t.strip().upper() for t in tickers.split(",") if t.strip()] or load_universe()
    with connect() as con:
        counts = ingest(con, selected)
        summary = con.execute("""
            SELECT ticker, min(date), max(date), count(*) FROM prices
            WHERE ticker IN (SELECT UNNEST(?)) GROUP BY ticker ORDER BY ticker
        """, [selected]).fetchall()
    typer.echo(f"Loaded {counts}")
    for t, lo, hi, n in summary:
        typer.echo(f"  {t:6s} prices {lo} → {hi} ({n} rows)")


@app.command("ingest-market")
def ingest_market(start: str = typer.Option("1990-01-01", help="History start date")) -> None:
    """Load ^VIX, SPY and ^IRX daily history from yfinance."""
    import logging

    from llmmoney.ingest.yf_loader import ingest
    from llmmoney.storage.db import connect

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with connect() as con:
        counts = ingest(con, start=start)
    typer.echo(f"Loaded {counts}")


@app.command("record-options")
def record_options(tickers: str = typer.Option("", help="Comma-separated subset; default = universe")) -> None:
    """Snapshot today's option chains and store IV features. Run daily after the close."""
    import logging

    from llmmoney.config.settings import load_universe
    from llmmoney.ingest.options_recorder import record
    from llmmoney.storage.db import connect

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    selected = [t.strip().upper() for t in tickers.split(",") if t.strip()] or load_universe()
    with connect() as con:
        counts = record(con, selected)
        rows = con.execute("""
            SELECT ticker, date, round(spot, 2), round(iv30_atm, 4), round(iv60_atm, 4),
                   round(skew30, 4), round(pc_oi_ratio, 2), n_contracts
            FROM iv_snapshots WHERE date = (SELECT max(date) FROM iv_snapshots)
              AND ticker IN (SELECT UNNEST(?)) ORDER BY ticker
        """, [selected]).fetchall()
    typer.echo(f"{counts}")
    typer.echo("ticker  date        spot     iv30    iv60    skew30  pc_oi  n")
    for r in rows:
        typer.echo("  ".join(str(x) for x in r))


@app.command("run-grokbot")
def run_grokbot(once: bool = typer.Option(False, help="Single poll then exit")) -> None:
    """Poll investor voices and news via the Grok API."""
    _todo("3 - Grokbot harness")


@app.command("build-features")
def build_features() -> None:
    """Build point-in-time daily features and both label tables."""
    from llmmoney.features.build_snapshots import run
    from llmmoney.storage.db import connect

    with connect() as con:
        counts = run(con)
        stats = con.execute("""
            SELECT round(avg(y_barrier), 3), count(y_barrier) FROM labels_barrier
        """).fetchone()
        fwd = con.execute("""
            SELECT y_fwd, count(*) FROM labels_fwd WHERE y_fwd IS NOT NULL GROUP BY 1 ORDER BY 1
        """).fetchall()
    typer.echo(f"{counts}")
    typer.echo(f"barrier label: positive rate {stats[0]} over {stats[1]} known monthly entries")
    typer.echo("5d label counts (0=down,1=flat,2=up): " + ", ".join(f"{int(k)}={n}" for k, n in fwd))


@app.command("backtest")
def backtest(model: str = typer.Option("logistic_long", help="logistic_long | logistic_short")) -> None:
    """Walk-forward train/evaluate, write out-of-sample signals, save the final model."""
    import pandas as pd

    from llmmoney.backtest.walkforward import SPECS, reliability, run
    from llmmoney.storage.db import connect

    if model not in SPECS:
        typer.echo(f"Unknown model {model!r}; choose from {', '.join(SPECS)}")
        raise typer.Exit(1)
    with connect() as con:
        report = run(con, model)
        calib = reliability(con, model) if model == "logistic_long" else None
    pd.set_option("display.width", 160)
    typer.echo(report.round(4).to_string(index=False))
    typer.echo("\nmean over years:\n" + report.drop(columns=["year", "C"]).mean().round(4).to_string())
    if calib is not None:
        typer.echo("\nreliability (out-of-sample, by predicted decile):\n" + calib.to_string())
    typer.echo(f"\nSaved models_out/{model}.joblib; out-of-sample predictions in signals (model='{model}').")


@app.command("backtest-dca")
def backtest_dca(
    top_n: int = typer.Option(5), tau: float = typer.Option(0.35), tranches: int = typer.Option(4),
    spacing: int = typer.Option(5), a: float = typer.Option(1.0), b: float = typer.Option(1.0),
) -> None:
    """Compare lump-sum, plain DCA, signal+IV DCA and no-model DCA. Needs both logistic backtests run first."""
    import pandas as pd

    from llmmoney.backtest.dca import DCAConfig, run
    from llmmoney.storage.db import connect

    cfg = DCAConfig(top_n=top_n, tau=tau, tranches=tranches, spacing=spacing, a=a, b=b)
    with connect() as con:
        summary, nav, _ = run(con, cfg)
    pd.set_option("display.width", 160)
    typer.echo(f"{nav.index[0].date()} → {nav.index[-1].date()}  config={cfg}")
    typer.echo(summary.round(4).to_string())
    typer.echo("\ncost_vs_dca: average purchase price relative to plain DCA (negative = bought cheaper)")


@app.command("serve")
def serve() -> None:
    """Run the signals + chat API."""
    _todo("7 - Ensemble and API")


if __name__ == "__main__":
    app()
