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
    _todo("2 - SimFin ingest")


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
