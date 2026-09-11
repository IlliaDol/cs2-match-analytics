"""Contract tests for the matches loader.

These pin the BEHAVIOR of load_matches() against the real primary file
(cs2_all_tiers_games.csv) and the quirks documented in DATA.md.

Contract: load_matches() returns the SERIES table — exactly one row per match_id,
with series scores verified as TEAM-sorted (score1 = team1's maps won).

Tests that need the real Kaggle file skip automatically when it is absent
(CI: data is git-ignored). Pure-function and error-handling tests always run.
"""

from pathlib import Path

import pandas as pd
import pytest

from cs2analytics.io import loader

PRIMARY = Path("data/raw/cs2_all_tiers_games.csv")
_HAS_REAL_DATA = PRIMARY.exists()

_needs_real_data = pytest.mark.skipif(
    not _HAS_REAL_DATA, reason="real Kaggle data not available (CI: data is git-ignored)"
)

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


@_needs_real_data
def test_load_returns_dataframe(real_matches):
    assert isinstance(real_matches, pd.DataFrame)
    assert len(real_matches) > 1000


@_needs_real_data
def test_columns_exact(real_matches):
    assert list(real_matches.columns) == list(EXPECTED_COLUMNS)


@_needs_real_data
def test_datetime_is_parsed(real_matches):
    assert pd.api.types.is_datetime64_any_dtype(real_matches["datetime"])
    assert real_matches["datetime"].notna().all()
    assert real_matches["datetime"].min().year >= 2023
    assert real_matches["datetime"].max().year <= 2026


@_needs_real_data
def test_scores_are_non_negative_integers(real_matches):
    for col in ("score1_match", "score2_match", "t1_series_score", "t2_series_score"):
        assert pd.api.types.is_integer_dtype(real_matches[col]), col
        assert (real_matches[col] >= 0).all(), col


@_needs_real_data
def test_one_row_per_match(real_matches):
    """The loader returns the series table, deduped: match_id is unique."""
    assert real_matches["match_id"].is_unique


@_needs_real_data
def test_rows_sorted_by_datetime(real_matches):
    order = real_matches["datetime"].sort_values(kind="mergesort")
    assert order.index.equals(real_matches.index)


# --- DATA.md Quirk 1 (corrected): series scores are TEAM-sorted ----------------


@_needs_real_data
def test_scores_are_team_sorted_and_winner_is_higher_score(real_matches):
    """score1_match = team1's maps won, score2_match = team2's maps won.
    Verified vs map-level round scores and two external Major finals.
    The winner must be the team with the HIGHER score (ties dropped)."""
    t1_won = real_matches["winner"] == real_matches["team1"]
    t2_won = real_matches["winner"] == real_matches["team2"]
    assert (t1_won | t2_won).all(), "every row must have a decisive winner"
    assert (
        real_matches.loc[t1_won, "t1_series_score"] > real_matches.loc[t1_won, "t2_series_score"]
    ).all()
    assert (
        real_matches.loc[t2_won, "t2_series_score"] > real_matches.loc[t2_won, "t1_series_score"]
    ).all()
    # team-sorted, not winner-sorted: both score orders occur (~55/45)
    assert (real_matches["score1_match"] > real_matches["score2_match"]).mean() > 0.40
    assert (real_matches["score2_match"] > real_matches["score1_match"]).mean() > 0.40


@_needs_real_data
def test_scores_match_map_level_evidence(real_matches):
    """Ground-truth cross-check vs map-level rows, using STRONG map evidence:
    a map is attributed only when the round-score direction and the team1_win
    flag AGREE (DATA.md Quirk 4: they contradict on ~1.5% of map rows, and on
    those rows neither source is trustworthy). Forfeited maps carry no reliable
    rounds and are invisible to this check, so agreement cannot be 100%.
    Contract: winner agreement >= 99% on matches with >= 1 strongly-attributed
    map (validated: 99.3%; residuals are forfeit-pattern series)."""
    import numpy as np

    raw = pd.read_csv(PRIMARY, low_memory=False)
    maps = raw.loc[raw["is_total"] == False].copy()  # noqa: E712
    for c in ("score1_game", "score2_game"):
        maps[c] = pd.to_numeric(maps[c], errors="coerce")
    rounds_t1 = maps["score1_game"] > maps["score2_game"]
    flag_t1 = maps["team1_win"] == 1
    strong = (
        maps["score1_game"].notna()
        & maps["score2_game"].notna()
        & (maps["score1_game"] != maps["score2_game"])
        & (rounds_t1 == flag_t1)
    )
    maps["map_winner"] = np.where(
        strong & rounds_t1, maps["team1"], np.where(strong, maps["team2"], None)
    )
    wins = (
        maps.dropna(subset=["map_winner"])
        .groupby(["match_id", "map_winner"])
        .size()
        .unstack(fill_value=0)
    )

    comparable = real_matches[real_matches["match_id"].isin(wins.index)]
    sampled = comparable.sample(min(800, len(comparable)), random_state=42)

    winner_ok = 0
    decided = 0
    for _, row in sampled.iterrows():
        w1 = int(wins.loc[row["match_id"]].get(row["team1"], 0))
        w2 = int(wins.loc[row["match_id"]].get(row["team2"], 0))
        if w1 == w2:
            continue  # strong evidence split evenly / forfeit-invisible
        decided += 1
        if (w1 > w2 and row["winner"] == row["team1"]) or (
            w2 > w1 and row["winner"] == row["team2"]
        ):
            winner_ok += 1
    assert decided > 500, f"too few strongly-decided comparisons: {decided}"
    assert winner_ok / decided >= 0.99, f"winner agreement only {winner_ok}/{decided}"


@_needs_real_data
def test_winner_is_one_of_the_two_teams(real_matches):
    winners = pd.concat([real_matches["team1"], real_matches["team2"]])
    assert real_matches["winner"].isin(winners).all()


@_needs_real_data
def test_known_major_finals_have_correct_winners(real_matches):
    """External ground truth: NAVI beat FaZe 2-1 (PGL Copenhagen 2024 final),
    Team Spirit beat FaZe 2-1 (Perfect World Shanghai 2024 final)."""
    cph = real_matches[real_matches["match_id"] == 1048414]
    assert len(cph) == 1 and cph["winner"].iloc[0] == "Natus Vincere"
    sh = real_matches[real_matches["match_id"] == 1859211]
    assert len(sh) == 1 and sh["winner"].iloc[0] == "Team Spirit"


# --- DATA.md Quirk 2: Bo1 masquerade -----------------------------------------


@_needs_real_data
def test_games_played_complete_and_positive(real_matches):
    assert real_matches["games_played"].notna().all()
    assert (real_matches["games_played"] > 0).all()
    assert (real_matches["games_played"] <= 7).all()


@_needs_real_data
def test_games_played_consistent_with_series_scores(real_matches):
    """For decided series: winner score + loser score == maps played.
    (Bo1s correctly yield 1; Bo3 2-1 yields 3; Bo2 draws yield 2.)"""
    total_maps = real_matches["t1_series_score"] + real_matches["t2_series_score"]
    assert (total_maps == real_matches["games_played"]).all()


# --- team names ---------------------------------------------------------------


@_needs_real_data
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
