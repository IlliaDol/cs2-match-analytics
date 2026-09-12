"""M13 §1 — monthly Elo table for the R forecaster -> outputs/elo_monthly.csv.

Monthly mean rating (post-match, after each month's games) per team for the
teams with the most series. Run from repo root:
    .venv/Scripts/python.exe scripts/build_elo_monthly.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cs2analytics.features.elo import run_elo_backtest

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    series = pd.read_csv(REPO / "outputs" / "series_clean.csv")
    series["datetime"] = pd.to_datetime(series["datetime"], utc=True, format="ISO8601")
    series = series.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    bt = run_elo_backtest(series, k=32.0)

    # post-match ratings per row (engine records pre; apply the update here)
    def post(ra, rb, s, k=32.0):
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        return ra + k * (s - e), rb + k * ((1.0 - s) - (1.0 - e))

    records = []
    ratings: dict[str, float] = {}
    for row in bt.itertuples(index=False):
        r1 = ratings.get(row.team1, 1500.0)
        r2 = ratings.get(row.team2, 1500.0)
        n1, n2 = post(r1, r2, row.result)
        ratings[row.team1] = n1
        ratings[row.team2] = n2
        records.append((row.datetime, row.team1, n1))
        records.append((row.datetime, row.team2, n2))

    long = pd.DataFrame(records, columns=["datetime", "team", "elo"])
    long["month"] = long["datetime"].dt.to_period("M").astype(str)

    top10 = long["team"].value_counts().head(10).index
    monthly = (
        long[long["team"].isin(top10)]
        .sort_values("datetime")
        .groupby(["team", "month"], as_index=False)
        .tail(1)[["team", "month", "datetime", "elo"]]
        .sort_values(["team", "month"])
        .reset_index(drop=True)
    )
    monthly.to_csv(REPO / "outputs" / "elo_monthly.csv", index=False)
    n_teams = monthly["team"].nunique()
    n_months = monthly["month"].nunique()
    print(f"wrote outputs/elo_monthly.csv: {len(monthly)} rows, {n_teams} teams, {n_months} months")


if __name__ == "__main__":
    main()
