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
    """Build point-in-time daily snapshots and labels."""
    _todo("4 - Features and labels")


@app.command("backtest")
def backtest(model: str = typer.Option("baseline")) -> None:
    """Walk-forward backtest for a model."""
    _todo("5 - Baseline and backtest")


@app.command("serve")
def serve() -> None:
    """Run the signals + chat API."""
    _todo("7 - Ensemble and API")


if __name__ == "__main__":
    app()
