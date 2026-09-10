"""M5 tests — DuckDB query contracts (data-gated; CI skips them).

Requires the human's db/build_db.py + db/queries.sql + db/run_queries.py to have run.
pip install duckdb first.
"""

from pathlib import Path

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

REPO = Path(__file__).parent.parent
DB = REPO / "outputs" / "cs2.duckdb"


@pytest.fixture(scope="module")
def con():
    if not DB.exists():
        pytest.skip("outputs/cs2.duckdb not built yet — run db/build_db.py (M5 §1)")
    return duckdb.connect(str(DB), read_only=True)


# --- §1 build contract ---------------------------------------------------------


def test_matches_row_count(con):
    n = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    assert n == 9922


def test_teams_table_loaded(con):
    n = con.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    assert n >= 19000


def test_team_stats_sane(con):
    n = con.execute("SELECT COUNT(*) FROM team_stats").fetchone()[0]
    assert n >= 700
    rows = con.execute("SELECT n_series, n_wins, win_share FROM team_stats LIMIT 100").fetchall()
    assert all(w <= s for s, w, _ in rows)


# --- Q1: monthly upsets ---------------------------------------------------------


def test_q1_monthly_upsets(con):
    q = (REPO / "outputs" / "q1.csv").read_text(encoding="utf-8")
    df = pd.read_csv(REPO / "outputs" / "q1.csv")
    assert {"month", "n_series", "n_upsets", "upset_rate"} <= set(df.columns)
    assert len(df) >= 40
    assert df["n_series"].sum() == 9922, "Q1 fanned out rows — the CTE join exploded"
    assert df["upset_rate"].between(0, 1).all()
    del q


# --- Q2: form going into a match (leakage trap) ----------------------------------


def test_q2_form_no_future_leakage(con):
    df = pd.read_csv(REPO / "outputs" / "q2.csv")
    assert {"match_id", "team", "form_win_share"} <= set(df.columns)
    assert df["form_win_share"].between(0, 1).all()
    # trap: recompute one team's form by hand at a random late match
    sample = df.sort_values("match_id").iloc[-1]
    hist = con.execute(
        """
        SELECT COUNT(*) FILTER (WHERE winner = ?) AS wins, COUNT(*) AS n
        FROM (
            SELECT * FROM matches
            WHERE (team1 = ? OR team2 = ?) AND datetime < (
                SELECT datetime FROM matches WHERE match_id = ?
            )
            ORDER BY datetime DESC LIMIT 3
        )
        """,
        [sample["team"], sample["team"], sample["team"], sample["match_id"]],
    ).fetchone()
    expected = hist[0] / hist[1]
    assert abs(sample["form_win_share"] - expected) < 1e-9, "Q2 used future data!"


# --- Q3: streaks -----------------------------------------------------------------


def test_q3_streaks(con):
    df = pd.read_csv(REPO / "outputs" / "q3.csv")
    assert {"team", "streak_length", "streak_start", "streak_end"} <= set(df.columns)
    assert len(df) >= 20
    assert (df["streak_length"] >= 5).all()


# --- Q4: elo-gap buckets ----------------------------------------------------------


def test_q4_upset_rate_monotone(con):
    df = pd.read_csv(REPO / "outputs" / "q4.csv")
    assert len(df) == 4
    rates = df.sort_values("elo_bucket")["upset_rate"].tolist()
    assert all(a >= b for a, b in zip(rates, rates[1:], strict=False)), (
        "upset rate must (weakly) decrease as the Elo gap grows — football logic"
    )


# --- Q5: head-to-head -------------------------------------------------------------


def test_q5_head_to_head(con):
    df = pd.read_csv(REPO / "outputs" / "q5.csv")
    assert {"team_a", "team_b", "a_wins", "b_wins", "n"} <= set(df.columns)
    assert len(df) >= 50
    assert (df["a_wins"] + df["b_wins"] == df["n"]).all()
    assert (df["n"] >= 4).all()
