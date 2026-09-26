"""M11 §3 — drift monitoring: PSI on the Elo-gap feature.

`psi(expected, actual)` = Population Stability Index between two samples;
`drift_report` computes per-month Elo-gap PSI vs the training-window
distribution and writes outputs/m11_drift.csv.

README sentence: "PSI on the Elo-gap feature flags when the team landscape has
shifted enough that the model needs refitting."
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_FLAG = 0.25  # conventional "investigate" threshold


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples over shared bins.

    Bin edges come from `expected`'s quantiles (equal-frequency); zero-count
    bins get an epsilon guard. Identical distributions give exactly 0.0.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # degenerate expected (all equal values)
        return 0.0
    eps = 1e-6
    e_counts = np.histogram(expected, bins=edges)[0] / len(expected)
    a_counts = np.histogram(actual, bins=edges)[0] / len(actual)
    e_pct = np.clip(e_counts, eps, None)
    a_pct = np.clip(a_counts, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(
    features: pd.DataFrame,
    cutoff: pd.Timestamp,
    bins: int = 10,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Per-month PSI of every monitored column vs its pre-cutoff distribution.

    Default columns: every numeric feature present in the frame except
    datetime/meta columns — i.e. the full feature set, not just elo_diff.
    Column names go into a `feature` column; the classic single-feature report
    is the special case columns=["elo_diff"].
    """
    df = features.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    boundary = cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff
    if columns is None:
        skip = {"datetime", "month", "match_id", "result", "tier", "is_bo1"}
        columns = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    rows = []
    for month, sub in df.groupby("month"):
        for col in columns:
            train = df.loc[df["datetime"] < boundary, col].dropna().to_numpy()
            p = psi(train, sub[col].dropna().to_numpy(), bins=bins)
            rows.append(
                {
                    "month": month,
                    "feature": col,
                    "n": len(sub),
                    "psi": p,
                    "flag": p > PSI_FLAG,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    from pathlib import Path

    from cs2analytics.features.matrix import DEFAULT_CUTOFF, build_feature_matrix

    repo = Path(__file__).resolve().parents[3]
    fs = build_feature_matrix(
        repo / "outputs" / "features_v1.parquet",
        extra_features=["roster_stability_diff", "standin_diff"],
    )
    # monitor EVERY trained feature, not just elo_diff: rebuild the wide frame
    # from the matrix (X columns are exactly fs.feature_names)
    df = fs.dates.to_frame("datetime").reset_index(drop=True)
    X = pd.DataFrame(fs.X, columns=fs.feature_names).reset_index(drop=True)
    wide = pd.concat([df, X], axis=1)
    report = drift_report(wide, DEFAULT_CUTOFF)
    dest = repo / "outputs" / "m11_drift.csv"
    report.to_csv(dest, index=False)
    flagged = int(report["flag"].sum())
    per_feature = report.groupby("feature")["flag"].sum()
    logger.info(
        "wrote %s: %d months x %d features, %d flags (psi > %s)",
        dest.name,
        report["month"].nunique(),
        report["feature"].nunique(),
        flagged,
        PSI_FLAG,
    )
    for feat, n_flags in per_feature[per_feature > 0].items():
        logger.warning("drift flags on %s: %d months", feat, n_flags)


if __name__ == "__main__":
    main()
