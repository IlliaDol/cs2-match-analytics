"""Build outputs/cs2.duckdb — M5 §1.

Loads the M3 wrangled series table + the raw teams lookup into DuckDB and
writes a team_stats aggregation table (join target for Q4). Run from repo root:
    .venv/Scripts/python.exe db/build_db.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DB_PATH = REPO / "outputs" / "cs2.duckdb"


def main() -> None:
    series = pd.read_csv(REPO / "outputs" / "series_clean.csv")
    teams = pd.read_csv(REPO / "data" / "raw" / "teams.csv")

    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))

    con.register("series_df", series)
    con.register("teams_df", teams)
    # typed matches table: datetime arrives as ISO-8601 VARCHAR with Z suffix
    con.execute(
        """
        CREATE TABLE matches AS
        SELECT * EXCLUDE (datetime), try_cast(datetime AS TIMESTAMPTZ) AS datetime
        FROM series_df
        """
    )
    con.execute("CREATE TABLE teams AS SELECT * FROM teams_df")

    con.execute(
        """
        CREATE TABLE team_stats AS
        SELECT
            team,
            COUNT(*) AS n_series,
            SUM(win) AS n_wins,
            SUM(win) / COUNT(*) AS win_share
        FROM (
            SELECT team1 AS team, (winner = team1)::INT AS win FROM matches
            UNION ALL
            SELECT team2 AS team, (winner = team2)::INT AS win FROM matches
        )
        GROUP BY team
        """
    )

    counts = con.execute(
        """
        SELECT 'matches' AS tbl, COUNT(*) AS n FROM matches
        UNION ALL SELECT 'teams', COUNT(*) FROM teams
        UNION ALL SELECT 'team_stats', COUNT(*) FROM team_stats
        """
    ).fetchall()
    for tbl, n in counts:
        print(f"{tbl}: {n} rows")
    con.close()


if __name__ == "__main__":
    main()
