"""Run the rolling-origin multi-split backtest and write outputs/m11_rolling.csv.

Reproducible companion to src/cs2analytics/models/rolling.py. From repo root:

    .venv/Scripts/python.exe scripts/build_rolling_backtest.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cs2analytics.models.rolling import rolling_backtest, summarize_rolling

REPO = Path(__file__).resolve().parents[1]

# Monthly cutoffs proved fragile at the tail (June fold n=201, logloss 0.683).
# Widen: 2025 quarterly cutoffs (expanding train) + all 2026 monthly ones.
CUTOFFS = (
    [pd.Timestamp(f"{y}-{m:02d}-01", tz="UTC") for y, m in ((2024, 7), (2025, 1), (2025, 7))]
    + [pd.Timestamp(f"2026-{m:02d}-01", tz="UTC") for m in (1, 2, 3, 4, 5, 6)]
)


def main() -> None:
    table = rolling_backtest(
        REPO / "outputs" / "features_v1.parquet",
        CUTOFFS,
        extra_features=["roster_stability_diff", "standin_diff"],
    )
    summary = summarize_rolling(table)
    dest = REPO / "outputs" / "m11_rolling.csv"
    table.to_csv(dest, index=False)
    summary.to_csv(REPO / "outputs" / "m11_rolling_summary.csv", index=False)
    print(f"[rolling] wrote {dest.relative_to(REPO)} ({len(table)} folds)")
    print(table.to_string())
    print(summary.to_string())


if __name__ == "__main__":
    main()
