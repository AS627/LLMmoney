from llmmoney.storage.db import connect


def test_schema_creates_tables():
    with connect(":memory:") as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"prices", "fundamentals", "posts", "news", "signals"} <= tables
