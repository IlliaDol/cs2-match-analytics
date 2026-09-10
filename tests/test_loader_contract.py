"""Contract tests for the matches loader.

These pin the BEHAVIOR of load_matches() against the real primary file
(cs2_all_tiers_games.csv) and the quirks documented in DATA.md.

Contract: load_matches() returns the SERIES table — exactly one row per match_id,
with winner-sorted book scores disentangled into team-specific series scores.
"""

import pandas as pd
import pytest

from cs2analytics.io import loader

PRIMARY = "data/raw/cs2_all_tiers_games.csv"

# The contract schema: exactly these columns, exactly this order (DATA.md).
EXPECTED_COLUMNS = (
    "match_id",
    "datetime",
    "tournament",
    "team1",
    "team2",
    "winner",
    "score1_match",
    "score2_match",
    "t1_series_score",
    "t2_series_score",
    "games_played",
    "bestOf",
)


@pytest.fixture(scope="module")
def real_matches():
    return loader.load_matches(PRIMARY)


# --- core contract -----------------------------------------------------------


def test_load_returns_dataframe(real_matches):
    assert isinstance(real_matches, pd.DataFrame)
    assert len(real_matches) > 1000


def test_columns_exact(real_matches):
    assert list(real_matches.columns) == list(EXPECTED_COLUMNS)


def test_datetime_is_parsed(real_matches):
    assert pd.api.types.is_datetime64_any_dtype(real_matches["datetime"])
    assert real_matches["datetime"].notna().all()
    assert real_matches["datetime"].min().year >= 2023
    assert real_matches["datetime"].max().year <= 2026


def test_scores_are_non_negative_integers(real_matches):
    for col in ("score1_match", "score2_match", "t1_series_score", "t2_series_score"):
        assert pd.api.types.is_integer_dtype(real_matches[col]), col
        assert (real_matches[col] >= 0).all(), col


def test_one_row_per_match(real_matches):
    """The loader returns the series table, deduped: match_id is unique."""
    assert real_matches["match_id"].is_unique


def test_rows_sorted_by_datetime(real_matches):
    order = real_matches["datetime"].sort_values(kind="mergesort")
    assert order.index.equals(real_matches.index)


# --- DATA.md Quirk 1: scores are winner-sorted, not team-sorted --------------


def test_winner_sorted_scores_are_disentangled(real_matches):
    """t1/t2 series scores must be team-specific: t1 >= t2 exactly when
    team1 won (equal only for genuine Bo2 draws)."""
    t1 = real_matches["t1_series_score"]
    t2 = real_matches["t2_series_score"]
    t1_won = real_matches["winner"] == real_matches["team1"]
    t2_won = real_matches["winner"] == real_matches["team2"]
    assert (t1[t1_won] >= t2[t1_won]).all()
    assert (t2[t2_won] >= t1[t2_won]).all()


def test_winner_score_values_are_valid(real_matches):
    """(loser, winner) series pairs come from a small valid set (DATA.md Quirk 1).

    Orientation note: `lo`/`hi` are min/max across the team-specific score
    columns, so the pair is (loser_score, winner_score) — e.g. (0, 2) is a
    2-0 series regardless of which team won.
    """
    lo = real_matches[["t1_series_score", "t2_series_score"]].min(axis=1).astype(int)
    hi = real_matches[["t1_series_score", "t2_series_score"]].max(axis=1).astype(int)
    valid = {(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (1, 1), (2, 2)}
    assert set(zip(lo, hi, strict=True)) <= valid


def test_winner_is_one_of_the_two_teams(real_matches):
    winners = pd.concat([real_matches["team1"], real_matches["team2"]])
    assert real_matches["winner"].isin(winners).all()


# --- DATA.md Quirk 2: Bo1 masquerade -----------------------------------------


def test_games_played_complete_and_positive(real_matches):
    assert real_matches["games_played"].notna().all()
    assert (real_matches["games_played"] > 0).all()
    assert (real_matches["games_played"] <= 7).all()


def test_games_played_consistent_with_series_scores(real_matches):
    """For decided series: winner score + loser score == maps played.
    (Bo1s correctly yield 1; Bo3 2-1 yields 3; Bo2 draws yield 2.)"""
    total_maps = real_matches["t1_series_score"] + real_matches["t2_series_score"]
    assert (total_maps == real_matches["games_played"]).all()


# --- team names ---------------------------------------------------------------


def test_team_names_stripped_and_present(real_matches):
    for col in ("team1", "team2", "winner"):
        s = real_matches[col].astype("string")
        assert (s == s.str.strip()).all(), col
        assert s.notna().all(), col


# --- normalize_team_name (pure function contract) -----------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Natus Vincere", "natus vincere"),
        ("  FaZe  ", "faze"),
        ("Na'Vi", "na'vi"),
        ("G2 Esports", "g2 esports"),
        ("", ""),
        (None, None),
    ],
)
def test_normalize_team_name(raw, expected):
    assert loader.normalize_team_name(raw) == expected


# --- error handling ------------------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load_matches(tmp_path / "definitely_not_here.csv")
