"""M7 §1 — feature matrix construction with a leakage guard.

Default features (ALL pre-match, leakage law): elo_diff, form5_diff,
rest_days_diff, is_bo1, tier one-hot. `extra_features` can append columns —
and any forbidden (post-outcome) name raises LeakageError, checked against the
FINAL feature list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CUTOFF = pd.Timestamp("2026-01-01")

DEFAULT_FEATURES = ["elo_diff", "form5_diff", "rest_days_diff", "is_bo1", "tier"]

#: KNOWN-SAFE pre-match columns. Allowlist, not denylist: a feature may enter a
#: model only if it is listed here (plus the tier_* one-hots and any explicit
#: `extra_features` names). A new post-outcome column in the feature store can
#: therefore never leak silently — it is simply rejected unless listed.
ALLOWED_BASE_FEATURES = {
    "elo_diff",
    "form5_diff",
    "rest_days_diff",
    "is_bo1",
    "h2h_t1_win_share",
    "roster_stability_diff",
    "standin_diff",
    # derived pre-match interaction: is_bo1 * elo_diff (both pre-match, product
    # of two pre-match columns cannot leak)
    "bo1_elo",
}

#: stale alias kept for any external importer; new code should use the allowlist
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


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """X, y, feature_names, dates, split, cutoff — the M7 training contract."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    dates: pd.Series
    split: pd.Series
    cutoff: pd.Timestamp

    @property
    def train_mask(self) -> np.ndarray:
        return (self.split == "train").to_numpy()

    @property
    def test_mask(self) -> np.ndarray:
        return (self.split == "test").to_numpy()


def _validate_feature_names(
    names: list[str], extra_features: list[str]
) -> None:
    """Reject features that are known to describe the match outcome."""
    for name in names:
        is_tier_column = name == "tier" or name.startswith("tier_")
        if is_tier_column:
            continue
        if name in FORBIDDEN_FEATURES or name not in ALLOWED_BASE_FEATURES:
            if name not in extra_features:
                raise LeakageError(
                    f"feature {name!r} is not in the known-safe pre-match allowlist — "
                    "add it to ALLOWED_BASE_FEATURES only after verifying it is pre-match"
                )

    # Explicit additions are intentionally stricter than the defaults: they
    # must be reviewed and named in the allowlist before entering a model.
    for name in extra_features:
        if name in FORBIDDEN_FEATURES:
            raise LeakageError(f"forbidden post-outcome feature {name!r}")
        if name not in ALLOWED_BASE_FEATURES:
            raise LeakageError(f"extra feature {name!r} not allowlisted as pre-match")


def _build_feature_frame(
    data: pd.DataFrame,
    extra_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build the numeric feature frame and retain the requested order."""
    tier_onehot = pd.get_dummies(data["tier"], prefix="tier").astype(float)
    base = data[["elo_diff", "form5_diff", "rest_days_diff", "is_bo1"]].astype(float)
    frame = pd.concat([base, tier_onehot], axis=1)

    for name in extra_features:
        if name != "tier" and name not in frame.columns and name in data.columns:
            frame = pd.concat([frame, data[[name]].astype(float)], axis=1)

    requested_names = list(DEFAULT_FEATURES) + extra_features
    ordered_names: list[str] = []
    for name in requested_names:
        if name == "tier":
            ordered_names.extend(column for column in frame if column.startswith("tier_"))
        else:
            ordered_names.append(name)

    missing = [name for name in ordered_names if name not in frame.columns]
    if missing:
        raise KeyError(f"requested feature(s) not in the feature store: {missing}")
    return frame[ordered_names], ordered_names


def _match_cutoff_to_dates(
    dates: pd.Series,
    cutoff: pd.Timestamp,
) -> pd.Timestamp:
    """Return a cutoff with the same timezone awareness as the data."""
    data_timezone = dates.dt.tz
    if cutoff.tz is None and data_timezone is not None:
        return cutoff.tz_localize(data_timezone)
    if cutoff.tz is not None and data_timezone is not None and cutoff.tz != data_timezone:
        return cutoff.tz_convert(data_timezone)
    return cutoff


def _split_by_cutoff(
    dates: pd.Series,
    cutoff: pd.Timestamp,
) -> tuple[pd.Series, pd.Timestamp]:
    """Label rows as train/test using a strict chronological boundary."""
    boundary = _match_cutoff_to_dates(dates, cutoff)
    labels = np.where(dates < boundary, "train", "test")
    return pd.Series(labels, name="split"), boundary


def build_feature_matrix(
    features_path: str | Path,
    extra_features: list[str] | None = None,
    cutoff: pd.Timestamp = DEFAULT_CUTOFF,
) -> FeatureSet:
    """Assemble X/y with a time split and the leakage guard.

    train = strictly before cutoff, test = on/after cutoff.
    Raises LeakageError if any requested feature (default or extra) is forbidden.
    """
    data = pd.read_parquet(features_path)
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
    extras = list(extra_features or [])
    requested_names = list(DEFAULT_FEATURES) + extras
    _validate_feature_names(requested_names, extras)

    feature_frame, feature_names = _build_feature_frame(data, extras)
    split, boundary = _split_by_cutoff(data["datetime"], cutoff)

    X = feature_frame.to_numpy(dtype=float)
    if np.isnan(X).any():
        n_nan = int(np.isnan(X).any(axis=1).sum())
        raise ValueError(f"X contains NaN rows ({n_nan}) — fix the feature store first")
    y = data["result"].to_numpy(dtype=float)

    return FeatureSet(
        X=X,
        y=y,
        feature_names=feature_names,
        dates=data["datetime"],
        split=split,
        cutoff=boundary,  # tz-matched to the data so date comparisons stay valid
    )
