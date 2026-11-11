from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from llmmoney.features import candles
from llmmoney.labels.barrier import label_series
from llmmoney.labels.forward_returns import DOWN, FLAT, UP, build as fwd_build


# ---------- barrier label ----------

@pytest.mark.parametrize("path,expected", [
    ([100, 110, 141, 60], 1.0),        # up first
    ([100, 90, 69, 150], 0.0),         # down first
    ([100, 105, 95, 100, 101], 0.0),   # neither, full window observed
])
def test_barrier_outcomes(path, expected):
    y, _ = label_series(np.array(path, float), np.array([0]), 1.4, 0.7, horizon=len(path) - 2)
    assert y[0] == expected


def test_barrier_unknown_when_window_open():
    y, _ = label_series(np.array([100, 105, 110], float), np.array([0]), 1.4, 0.7, horizon=10)
    assert np.isnan(y[0])


def test_barrier_known_early_if_hit():
    y, days = label_series(np.array([100, 150, 100], float), np.array([0]), 1.4, 0.7, horizon=10)
    assert y[0] == 1.0 and days[0] == 1


# ---------- candle features ----------

def _prices(n=300, seed=0, ticker="T"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    d0 = date(2020, 1, 1)
    return pd.DataFrame({"ticker": ticker, "date": [d0 + timedelta(days=i) for i in range(n)],
                         "open": open_, "high": high, "low": low, "close": close, "adj_close": close,
                         "volume": rng.integers(1_000, 10_000, n), "shares_out": np.nan})


def test_candle_shape_on_known_bar():
    px = pd.DataFrame({"ticker": "T", "date": [date(2020, 1, 1)], "open": [10.0], "high": [14.0],
                       "low": [8.0], "close": [12.0], "adj_close": [12.0], "volume": [100], "shares_out": [np.nan]})
    f = candles.build(px).iloc[0]
    assert f["body"] == pytest.approx(2 / 6)
    assert f["upper_wick"] == pytest.approx(2 / 6)
    assert f["lower_wick"] == pytest.approx(2 / 6)
    assert f["close_pos"] == pytest.approx(4 / 6)


def test_split_adjustment_removes_fake_crash():
    px = _prices(30)
    px.loc[20:, ["open", "high", "low", "close"]] /= 2  # 2:1 split on day 20, adj_close unchanged
    f = candles.build(px)
    assert abs(f.loc[20, "ret_1"]) < 0.05


def test_no_lookahead():
    """Changing the future must not change features at earlier dates."""
    px = _prices()
    base = candles.build(px)
    bumped = px.copy()
    bumped.loc[250:, ["open", "high", "low", "close", "adj_close"]] *= 3
    bumped.loc[250:, "volume"] *= 50
    after = candles.build(bumped)
    cols = [c for c in base.columns if c not in ("ticker", "date")]
    pd.testing.assert_frame_equal(base.loc[:249, cols], after.loc[:249, cols])


# ---------- forward-return label ----------

def test_fwd_label_buckets():
    d = [date(2024, 1, i) for i in range(1, 5)]
    px = pd.DataFrame({"ticker": "T", "date": d, "adj_close": [100, 103, 100, 99.5]})
    spy = pd.DataFrame({"symbol": "SPY", "date": d, "adj_close": [100.0] * 4})
    out = fwd_build(px, spy, horizon=1, flat_band=0.01)
    assert out["y_fwd"].tolist()[:3] == [UP, DOWN, FLAT]
    assert np.isnan(out["y_fwd"].iloc[3])
