"""M11 §3 — drift monitoring: PSI on the Elo-gap feature.

`psi(expected, actual)` = Population Stability Index between two samples;
`drift_report` computes per-month Elo-gap PSI vs the training-window
distribution and writes outputs/m11_drift.csv.

README sentence: "PSI on the Elo-gap feature flags when the team landscape has
shifted enough that the model needs refitting."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
) -> pd.DataFrame:
    """Per-month PSI of elo_diff vs the training window (pre-cutoff) distribution."""
    df = features.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    boundary = cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff
    train = df.loc[df["datetime"] < boundary, "elo_diff"].to_numpy()
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    rows = []
    for month, sub in df.groupby("month"):
        p = psi(train, sub["elo_diff"].to_numpy(), bins=bins)
        rows.append({"month": month, "n": len(sub), "psi": p, "flag": p > PSI_FLAG})
    return pd.DataFrame(rows)


def main() -> None:
    from pathlib import Path

    from cs2analytics.features.matrix import DEFAULT_CUTOFF, build_feature_matrix

    repo = Path(__file__).resolve().parents[3]
    fs = build_feature_matrix(repo / "outputs" / "features_v1.parquet")
    df = fs.dates.to_frame("datetime").assign(elo_diff=fs.X[:, fs.feature_names.index("elo_diff")])
    report = drift_report(df, DEFAULT_CUTOFF)
    dest = repo / "outputs" / "m11_drift.csv"
    report.to_csv(dest, index=False)
    flagged = int(report["flag"].sum())
    print(f"wrote {dest.name}: {len(report)} months, {flagged} flagged (psi > {PSI_FLAG})")


if __name__ == "__main__":
    main()
