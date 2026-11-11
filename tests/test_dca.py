import numpy as np
import pytest

from llmmoney.backtest.dca import DCAConfig, simulate_position

CFG = DCAConfig(tranches=4, spacing=1, cost_bps=0.0, horizon=10)


def test_lump_hits_take_profit():
    close = mid = np.array([100, 120, 145, 150, 150], float)
    path, info = simulate_position(close, mid, 0, 4, np.array([1.0]), None, CFG)
    assert info["exit_days"] == 2
    assert info["ret"] == pytest.approx(0.45)
    assert path[-1] == path[2]  # flat in cash after exit


def test_dca_skips_tranches_after_exit():
    close = mid = np.array([100, 100, 60, 60, 60], float)  # stop-loss on day 2
    path, info = simulate_position(close, mid, 0, 4, np.full(4, 0.25), None, CFG)
    assert info["invested"] == pytest.approx(0.5)  # only days 0 and 1 bought
    assert info["ret"] == pytest.approx(0.5 * 0.6 - 0.5)


def test_signal_sizing_spends_exact_budget_without_lookahead():
    close = mid = np.array([100, 99, 98, 97, 96, 95], float)
    _, info = simulate_position(close, mid, 0, 5, None, np.array([2.0, 2.0, 0.25, 0.25]), CFG)
    assert info["invested"] == pytest.approx(1.0)  # first two tranches spend 0.5 each, rest get 0


def test_signal_last_tranche_takes_remainder():
    close = mid = np.full(6, 100.0)
    _, info = simulate_position(close, mid, 0, 5, None, np.array([0.25, 0.25, 0.25, 0.25]), CFG)
    assert info["invested"] == pytest.approx(1.0)
