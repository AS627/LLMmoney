from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from llmmoney.ingest.options_recorder import Chain, _interp_dte, compute_features


def _side(kind, expiry, dte, strikes, ivs, bid=1.0, volume=10, oi=100):
    return pd.DataFrame({"type": kind, "expiry": expiry, "dte": dte, "strike": strikes, "iv": ivs,
                         "bid": bid, "volume": volume, "open_interest": oi})


def _chain():
    strikes = [80, 90, 100, 110, 120]
    frames = []
    # 20d expiry: flat 0.20 calls, puts with skew; 40d expiry: flat 0.30 calls
    frames.append(_side("C", "e20", 20, strikes, [0.20] * 5))
    frames.append(_side("P", "e20", 20, strikes, [0.30, 0.25, 0.20, 0.20, 0.20], volume=20, oi=300))
    frames.append(_side("C", "e40", 40, strikes, [0.30] * 5))
    frames.append(_side("P", "e40", 40, strikes, [0.40, 0.35, 0.30, 0.30, 0.30], volume=20, oi=300))
    # junk: zero bid and absurd IV must be ignored
    frames.append(_side("C", "e20", 20, [100], [9.0], bid=0.0))
    return Chain("TEST", 100.0, datetime(2026, 11, 11, 21, tzinfo=timezone.utc), pd.concat(frames, ignore_index=True))


def test_iv30_interpolates_total_variance():
    f = compute_features(_chain())
    expected = np.sqrt((0.5 * 0.20**2 * 20 + 0.5 * 0.30**2 * 40) / 30)
    assert f["iv30_atm"] == pytest.approx(expected)


def test_skew_and_ratios():
    f = compute_features(_chain())
    # 20d skew: put@90 (0.25) - call@110 (0.20) = 0.05; 40d: 0.35 - 0.30 = 0.05
    assert f["skew30"] == pytest.approx(0.05)
    assert f["pc_vol_ratio"] == pytest.approx(2.0, rel=0.1)  # junk call adds 10 volume
    assert f["pc_oi_ratio"] == pytest.approx(3000 / 1100)  # 10 put rows x 300 vs 10 call rows x 100 + junk 100
    assert f["n_contracts"] == 20  # junk row dropped
    assert f["date"].isoformat() == "2026-11-11"


def test_interp_falls_back_to_nearest_when_not_bracketed():
    per = pd.DataFrame({"expiry": ["a"], "dte": [45], "atm": [0.25], "skew": [0.0]})
    assert _interp_dte(per, "atm", 30) == pytest.approx(0.25)
