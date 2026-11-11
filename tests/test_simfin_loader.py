import pandas as pd

from llmmoney.ingest.simfin_loader import _upsert, fundamentals_long, prices_frame
from llmmoney.storage.db import connect


def _income_wide():
    return pd.DataFrame({
        "Ticker": ["AAPL", "AAPL"],
        "SimFinId": [1, 1],
        "Currency": ["USD", "USD"],
        "Fiscal Year": [2026, 2026],
        "Fiscal Period": ["Q1", "Q2"],
        "Report Date": pd.to_datetime(["2025-12-31", "2026-03-31"]),
        "Publish Date": pd.to_datetime(["2026-01-30", "2026-05-01"]),
        "Restated Date": pd.to_datetime(["2026-01-30", "2026-05-01"]),
        "Revenue": [100.0, 110.0],
        "Net Income": [20.0, None],
    })


def test_fundamentals_long_melts_and_drops_nulls():
    out = fundamentals_long(_income_wide(), "income")
    assert len(out) == 3  # 2 revenue + 1 net income (null dropped)
    assert set(out["item"]) == {"Revenue", "Net Income"}
    assert (out["statement"] == "income").all()
    assert str(out["publish_date"].iloc[0]) == "2026-01-30"


def test_upsert_is_idempotent():
    fund = fundamentals_long(_income_wide(), "income")
    px = prices_frame(pd.DataFrame({
        "Ticker": ["AAPL"], "Date": ["2026-01-02"], "Open": [1.0], "High": [1.0], "Low": [1.0],
        "Close": [1.0], "Adj. Close": [1.0], "Volume": [10], "Shares Outstanding": [5.0],
    }))
    with connect(":memory:") as con:
        for _ in range(2):
            _upsert(con, "fundamentals", fund, ["AAPL"])
            _upsert(con, "prices", px, ["AAPL"])
        assert con.execute("SELECT count(*) FROM fundamentals").fetchone()[0] == 3
        assert con.execute("SELECT count(*) FROM prices").fetchone()[0] == 1
