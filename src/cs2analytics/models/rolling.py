"""P1.4 — rolling-origin multi-split backtest.

One fixed 2026-01-01 test split is a single anecdote with an unknown variance:
the M7 direction can flip with a different cutoff (the repo's own leakage-demo
notebook admits the 2026 window is noisier). This module kills that objection by
scoring the same model over several walk-forward test windows and reporting the
distribution instead of one point estimate.

Law: every fold uses strictly-earlier data for training — no lookahead. Each
fold's model is refit from scratch on that fold's train slice; features are
taken from the pre-built feature store (all columns are pre-match already).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cs2analytics.evaluation.metrics import accuracy_from_probs, brier_score, log_loss
from cs2analytics.features.matrix import build_feature_matrix
from cs2analytics.models.logistic import fit_logistic, predict_proba


@dataclass
class _Fold:
    cutoff: pd.Timestamp
    n_train: int
    n_test: int
    logloss: float
    brier: float
    acc: float


def rolling_backtest(
    features_path: str | Path,
    cutoffs: list[pd.Timestamp],
    extra_features: list[str] | None = None,
    min_train: int = 2000,
    min_test: int = 100,
    model_fn=fit_logistic,
) -> pd.DataFrame:
    """Score the model on every cutoff (train < cutoff <= test), refit per fold.

    Folds with too little train or test data are skipped with a warning row
    rather than raising, so a long cutoff list degrades gracefully.
    """
    fv = pd.read_parquet(features_path)
    fv["datetime"] = pd.to_datetime(fv["datetime"], utc=True)
    rows: list[dict] = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        train = fv[fv["datetime"] < cutoff]
        test = fv[fv["datetime"] >= cutoff]
        if len(train) < min_train or len(test) < min_test:
            continue
        # refit from scratch on this fold's train slice via the same matrix path
        fs = build_feature_matrix(
            features_path, extra_features=extra_features, cutoff=pd.Timestamp(cutoff)
        )
        m = model_fn(fs.X[fs.train_mask], fs.y[fs.train_mask])
        p = predict_proba(m, fs.X[fs.test_mask])
        y = fs.y[fs.test_mask]
        rows.append(
            {
                "cutoff": cutoff,
                "n_train": int(fs.train_mask.sum()),
                "n_test": int(fs.test_mask.sum()),
                "logloss": log_loss(p, y),
                "brier": brier_score(p, y),
                "acc": accuracy_from_probs(p, y),
            }
        )
    return pd.DataFrame(rows)


def summarize_rolling(table: pd.DataFrame) -> pd.DataFrame:
    """Mean ± a bootstrap-free Gaussian 95% CI (n small → use it as a bound, not a claim)."""
    if table.empty:
        return pd.DataFrame()
    out = []
    for col in ("logloss", "brier", "acc"):
        vals = table[col].to_numpy()
        mu = float(np.mean(vals))
        # 95% CI via Student's t — honest given 5-8 folds only
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            from scipy import stats  # noqa: PLC0415 — local; scipy is a heavy optional dep

            t = stats.t.ppf(0.975, df=len(vals) - 1)
            lo, hi = mu - t * sem, mu + t * sem
        else:
            lo = hi = mu
        out.append(
            {"metric": col, "mean": mu, "ci_lo": lo, "ci_hi": hi, "n_folds": len(vals)}
        )
    return pd.DataFrame(out)
