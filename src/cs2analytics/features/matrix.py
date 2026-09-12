"""M7 §1 — feature matrix construction with a leakage guard.

Default features (ALL pre-match, leakage law): elo_diff, form5_diff,
rest_days_diff, is_bo1, tier one-hot. `extra_features` can append columns —
and any forbidden (post-outcome) name raises LeakageError, checked against the
FINAL feature list.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CUTOFF = pd.Timestamp("2026-01-01")

DEFAULT_FEATURES = ["elo_diff", "form5_diff", "rest_days_diff", "is_bo1", "tier"]

#: post-outcome columns that must NEVER enter a model
FORBIDDEN_FEATURES = {
    "games_played",
    "margin",
    "swept",
    "result",
    "winner",
    "t1_series_score",
    "t2_series_score",
    "total_maps",
    "bo5_sweep",
}


class LeakageError(ValueError):
    """Raised when a forbidden (post-outcome) column is requested as a feature."""


class FeatureSet:
    """X, y, feature_names, dates, split, cutoff — the M7 training contract."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        dates: pd.Series,
        split: pd.Series,
        cutoff: pd.Timestamp,
    ) -> None:
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.dates = dates
        self.split = split
        self.cutoff = cutoff

    @property
    def train_mask(self) -> np.ndarray:
        return (self.split == "train").to_numpy()

    @property
    def test_mask(self) -> np.ndarray:
        return (self.split == "test").to_numpy()


def build_feature_matrix(
    features_path: str | Path,
    extra_features: list[str] | None = None,
    cutoff: pd.Timestamp = DEFAULT_CUTOFF,
) -> FeatureSet:
    """Assemble X/y with a time split and the leakage guard.

    train = strictly before cutoff, test = on/after cutoff.
    Raises LeakageError if any requested feature (default or extra) is forbidden.
    """
    df = pd.read_parquet(features_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    names = list(DEFAULT_FEATURES) + list(extra_features or [])
    bad = [n for n in names if n in FORBIDDEN_FEATURES]
    if bad:
        raise LeakageError(
            f"forbidden post-outcome feature(s): {bad} — games_played/margin/swept/"
            "result-style columns are only known AFTER the match"
        )

    tier_onehot = pd.get_dummies(df["tier"], prefix="tier").astype(float)
    base = df[["elo_diff", "form5_diff", "rest_days_diff", "is_bo1"]].astype(float)
    X_frame = pd.concat([base, tier_onehot], axis=1)

    # extra pre-match features arrive as columns of the feature store; append them
    for n in (extra_features or []):
        if n != "tier" and n not in X_frame.columns and n in df.columns:
            X_frame = pd.concat([X_frame, df[[n]].astype(float).reset_index(drop=True)], axis=1)

    ordered: list[str] = []
    for n in names:
        if n == "tier":
            ordered.extend(c for c in X_frame.columns if c.startswith("tier_"))
        else:
            ordered.append(n)
    missing = [c for c in ordered if c not in X_frame.columns]
    if missing:
        raise KeyError(f"requested feature(s) not in the feature store: {missing}")
    X_frame = X_frame[ordered]

    tz = df["datetime"].dt.tz
    if (cutoff.tz is None) and (tz is not None):
        boundary = cutoff.tz_localize(tz)
    elif (cutoff.tz is not None) and (tz is not None) and cutoff.tz != tz:
        boundary = cutoff.tz_convert(tz)
    else:
        boundary = cutoff
    split = np.where(df["datetime"] < boundary, "train", "test")
    split = pd.Series(split, name="split")

    X = X_frame.to_numpy(dtype=float)
    if np.isnan(X).any():
        n_nan = int(np.isnan(X).any(axis=1).sum())
        raise ValueError(f"X contains NaN rows ({n_nan}) — fix the feature store first")
    y = df["result"].to_numpy(dtype=float)

    return FeatureSet(
        X=X,
        y=y,
        feature_names=list(X_frame.columns),
        dates=df["datetime"],
        split=split,
        cutoff=boundary,  # tz-matched to the data so date comparisons stay valid
    )
