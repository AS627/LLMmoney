"""DuckDB connection helper and schema bootstrap."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from llmmoney.config.settings import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_PATH.read_text())


@contextmanager
def connect(path: str | Path | None = None, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the project DB (or `path`, e.g. ':memory:' in tests) with schema applied."""
    target = str(path) if path is not None else str(get_settings().db_path)
    con = duckdb.connect(target, read_only=read_only)
    try:
        if not read_only:
            init_schema(con)
        yield con
    finally:
        con.close()
