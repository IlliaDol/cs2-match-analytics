"""M7 §2 — model layer: logistic + GBM pipelines. No tuning on test."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def fit_logistic(X_tr: np.ndarray, y_tr: np.ndarray) -> Pipeline:
    """StandardScaler + LogisticRegression — the interpretable baseline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    ).fit(X_tr, y_tr)


def fit_gbm(X_tr: np.ndarray, y_tr: np.ndarray) -> Pipeline:
    """HistGradientBoostingClassifier — mild regularization, no tuning on test."""
    return Pipeline(
        [
            (
                "gbm",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    max_iter=200,
                    learning_rate=0.06,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    ).fit(X_tr, y_tr)


def fit_gbm_isotonic(X_tr: np.ndarray, y_tr: np.ndarray) -> CalibratedClassifierCV:
    """GBM + isotonic calibration, cv = TimeSeriesSplit(5) on train only."""
    gbm = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=200,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    return CalibratedClassifierCV(gbm, method="isotonic", cv=TimeSeriesSplit(n_splits=5)).fit(
        X_tr, y_tr
    )


def predict_proba(model: object, X: np.ndarray) -> np.ndarray:
    """Column of P(y=1) as a 1-D array."""
    return model.predict_proba(X)[:, 1]
