"""Logistic classifier pipeline: winsorize -> impute (+missing flags) -> scale -> L2 logistic.

Winsorization bounds are learned on the training split only (the notebook clipped at the
1st/99th percentile of the full cross-section). Features that are entirely missing in the
training split (e.g. per-stock IV before recording started) are dropped by the imputer and
come in automatically once the model is retrained on data that has them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NON_FEATURES = {"ticker", "date", "fund_age_days"}
C_GRID = (0.01, 0.1, 1.0)


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (e.g. IV before recording)
            self.lo_ = np.nanquantile(X, self.lower, axis=0)
            self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


# Expected while a feature (e.g. per-stock IV) has no history in the training window.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")


def make_pipeline(C: float = 0.1) -> Pipeline:
    return Pipeline([
        ("winsor", Winsorizer()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        # No class re-weighting: it inflates probabilities, and DCA thresholds need calibrated P(hit).
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


def fit_with_c_search(X: pd.DataFrame, y: np.ndarray, dates: pd.Series, val_frac: float = 0.2) -> Pipeline:
    """Pick C on the last `val_frac` of training dates (time-ordered), then refit on all of it."""
    d = pd.to_datetime(dates)
    cut = d.quantile(1 - val_frac)
    tr, va = (d < cut).to_numpy(), (d >= cut).to_numpy()
    best_c, best = C_GRID[1], np.inf
    if tr.sum() > 100 and va.sum() > 50 and len(np.unique(y[va])) == len(np.unique(y)):
        for c in C_GRID:
            m = make_pipeline(c).fit(X[tr], y[tr])
            loss = log_loss(y[va], m.predict_proba(X[va]), labels=m.classes_)
            if loss < best:
                best_c, best = c, loss
    return make_pipeline(best_c).fit(X, y)
