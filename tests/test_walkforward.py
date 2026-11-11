from datetime import date, timedelta

import numpy as np
import pandas as pd

from llmmoney.backtest.walkforward import SPECS, run
from llmmoney.models.logistic import Winsorizer, make_pipeline
from llmmoney.storage.db import connect


def test_winsorizer_uses_train_bounds_only():
    w = Winsorizer(0.0, 1.0).fit(np.array([[0.0], [1.0]]))
    assert w.transform(np.array([[5.0], [-5.0]])).ravel().tolist() == [1.0, 0.0]


def test_pipeline_drops_all_missing_feature():
    X = pd.DataFrame({"a": np.r_[np.zeros(50), np.ones(50)], "iv": np.nan})
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    m = make_pipeline().fit(X, y)
    assert m.predict_proba(X.assign(iv=0.3)).shape == (100, 2)


def test_purge_trains_only_on_outcomes_known_before_test_year(monkeypatch):
    """Rows whose outcome lands in the test year must not be in that year's training set."""
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for t in ("A", "B", "C", "D", "E", "F"):
        for m in range(12 * 9):  # 2010-2018 monthly
            d = date(2010, 1, 1) + timedelta(days=31 * m)
            x = rng.normal()
            rows.append({"ticker": t, "date": d, "f1": x})
            labels.append({"ticker": t, "date": d, "y_barrier": float(x + rng.normal() > 0.5),
                           "barrier_days": 252, "outcome_date": d + timedelta(days=365)})
    seen = {}
    import llmmoney.backtest.walkforward as wf
    real_fit = wf.fit_with_c_search

    def spy_fit(X, y, dates):
        seen.setdefault("calls", []).append(pd.to_datetime(dates).max())
        return real_fit(X, y, dates)

    monkeypatch.setattr(wf, "fit_with_c_search", spy_fit)
    monkeypatch.setattr(wf, "MODELS_DIR", __import__("pathlib").Path("/tmp/llmmoney_test_models"))
    with connect(":memory:") as con:
        con.register("f", pd.DataFrame(rows)); con.execute("CREATE TABLE features_daily AS SELECT * FROM f")
        con.register("l", pd.DataFrame(labels)); con.execute("CREATE TABLE labels_barrier AS SELECT * FROM l")
        report = run(con, "logistic_long")
    assert not report.empty
    for year, last_train_date in zip(report["year"], seen["calls"]):
        # outcome = date + 365d < 1 Jan year  =>  entry date < 1 Jan (year - 1)
        assert last_train_date < pd.Timestamp(year - 1, 1, 1)
    assert report["auc"].mean() > 0.6  # signal in f1 is learnable
