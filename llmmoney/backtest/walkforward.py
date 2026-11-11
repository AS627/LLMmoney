"""Expanding-window walk-forward evaluation for the logistic models.

For each test year Y the model trains only on rows whose label outcome was known before
1 Jan Y (outcome_date < Y). That is a purge equal to the label horizon (252 trading days
for the barrier label, 5 for the short label), so overlapping label windows cannot leak
into the test year. Out-of-sample predictions are written to `signals`; a final model fit
on every known label is saved to models_out/ and scores the latest feature rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score

from llmmoney.models.logistic import feature_columns, fit_with_c_search


@dataclass(frozen=True)
class Spec:
    name: str
    label_table: str
    label_col: str
    score_col: str | None   # continuous outcome for IC (None -> use the label)
    min_train_years: int


SPECS = {
    "logistic_long": Spec("logistic_long", "labels_barrier", "y_barrier", None, 5),
    "logistic_short": Spec("logistic_short", "labels_fwd", "y_fwd", "fwd_excess", 5),
}
MODELS_DIR = Path("models_out")


def load(con: duckdb.DuckDBPyConnection, spec: Spec) -> pd.DataFrame:
    extra = f", l.{spec.score_col}" if spec.score_col else ""
    df = con.execute(f"""
        SELECT f.*, l.{spec.label_col} AS y, l.outcome_date{extra}
        FROM features_daily f JOIN {spec.label_table} l USING (ticker, date)
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    df["outcome_date"] = pd.to_datetime(df["outcome_date"])
    return df


def _signal(model, X: pd.DataFrame) -> np.ndarray:
    """Scalar score: P(hit) for the binary model, P(up) - P(down) for the 3-class model."""
    p = model.predict_proba(X)
    return p[:, 1] if p.shape[1] == 2 else p[:, 2] - p[:, 0]


def _probs_frame(model, X: pd.DataFrame, meta: pd.DataFrame, name: str) -> pd.DataFrame:
    p = model.predict_proba(X)
    out = meta[["ticker", "date"]].copy()
    if p.shape[1] == 2:
        out["p_down"], out["p_flat"], out["p_up"] = p[:, 0], np.nan, p[:, 1]
    else:
        out["p_down"], out["p_flat"], out["p_up"] = p[:, 0], p[:, 1], p[:, 2]
    out["model"] = name
    out["created_at"] = datetime.now()
    out["date"] = out["date"].dt.date
    return out


def _year_metrics(y: np.ndarray, p: np.ndarray, score: np.ndarray, dates: pd.Series, outcome: np.ndarray) -> dict:
    m: dict = {"n": len(y), "base_rate": float(np.mean(y == y.max())) if p.ndim == 1 else None}
    if p.ndim == 1:  # binary
        m["auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        m["logloss"] = log_loss(y, np.c_[1 - p, p], labels=[0, 1])
        top = p >= np.quantile(p, 0.9)
        m["top_decile_rate"] = float(y[top].mean())
    # per-date rank IC, averaged (cross-sectional skill)
    ics = []
    for _, idx in pd.Series(np.arange(len(y))).groupby(dates.to_numpy()):
        if len(idx) >= 5 and np.std(score[idx]) > 0 and np.std(outcome[idx]) > 0:
            ics.append(spearmanr(score[idx], outcome[idx]).statistic)
    m["ic_mean"] = float(np.mean(ics)) if ics else np.nan
    return m


def run(con: duckdb.DuckDBPyConnection, model_name: str) -> pd.DataFrame:
    spec = SPECS[model_name]
    df = load(con, spec)
    feats = feature_columns(df.drop(columns=["y", "outcome_date"] + ([spec.score_col] if spec.score_col else [])))
    known = df[df["y"].notna() & df["outcome_date"].notna()]
    first_year = known["date"].dt.year.min() + spec.min_train_years
    last_year = known["date"].dt.year.max()

    reports, oos = [], []
    for year in range(first_year, last_year + 1):
        start = pd.Timestamp(year, 1, 1)
        train = known[known["outcome_date"] < start]
        test = known[known["date"].dt.year == year]
        if len(train) < 500 or test.empty:
            continue
        y_tr = train["y"].to_numpy().astype(int)
        model = fit_with_c_search(train[feats], y_tr, train["date"])
        y_te = test["y"].to_numpy().astype(int)
        p_all = model.predict_proba(test[feats])
        p = p_all[:, 1] if p_all.shape[1] == 2 else p_all
        score = _signal(model, test[feats])
        outcome = test[spec.score_col].to_numpy() if spec.score_col else y_te.astype(float)
        if p_all.shape[1] == 3:
            met = {"n": len(y_te), "logloss": log_loss(y_te, p_all, labels=model.classes_),
                   "acc": float((model.classes_[p_all.argmax(1)] == y_te).mean())}
            met.update({k: v for k, v in _year_metrics(y_te, p_all, score, test["date"], outcome).items() if k == "ic_mean"})
        else:
            met = _year_metrics(y_te, p, score, test["date"], outcome)
        met.update({"year": year, "train_n": len(train), "C": model.named_steps["clf"].C})
        reports.append(met)
        oos.append(_probs_frame(model, test[feats], test, spec.name))

    if oos:
        sig = pd.concat(oos, ignore_index=True)
        con.execute("DELETE FROM signals WHERE model = ?", [spec.name])
        con.register("incoming", sig)
        con.execute("INSERT INTO signals (ticker, date, model, p_down, p_flat, p_up, created_at) "
                    "SELECT ticker, date, model, p_down, p_flat, p_up, created_at FROM incoming")
        con.unregister("incoming")

    # final model on every known label; score the most recent date for live use
    final = fit_with_c_search(known[feats], known["y"].to_numpy().astype(int), known["date"])
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": feats, "trained_through": known["outcome_date"].max()},
                MODELS_DIR / f"{spec.name}.joblib")
    return pd.DataFrame(reports)


def reliability(con: duckdb.DuckDBPyConnection, model_name: str, bins: int = 10) -> pd.DataFrame:
    """Calibration of out-of-sample P(hit) for the long model: predicted vs realized by decile."""
    df = con.execute("""
        SELECT s.p_up AS p, l.y_barrier AS y FROM signals s
        JOIN labels_barrier l USING (ticker, date) WHERE s.model = ? AND l.y_barrier IS NOT NULL
    """, [model_name]).df()
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    return df.groupby("bin").agg(pred=("p", "mean"), realized=("y", "mean"), n=("y", "size")).round(3)
