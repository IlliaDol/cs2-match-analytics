"""Data loading and schema contracts.

Contract (enforced by tests/test_loader_contract.py and documented in DATA.md):
- load_matches() returns the SERIES table: exactly one row per match_id.
- Series scores are TEAM-sorted (Quirk 1): score1_match = team1's maps won,
  score2_match = team2's maps won. Verified against map-level round scores,
  (score1+score2 == games_played on 100% of decided rows) and against two
  external ground-truth results (PGL Copenhagen 2024 and Perfect World
  Shanghai 2024 Major finals).
- The series-row `team1_win` flag is UNRELIABLE (94% zeros, contradicts map
  evidence) — never used for the winner.
- bestOf is unreliable for Bo1s (Quirk 2); games_played is the truth for maps played.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns kept from the raw file, in contract order (game_id / is_total are
# intentionally dropped: the loader returns the series table only).
_RAW_KEEP = (
    "match_id",
    "datetime",
    "tournament",
    "team1",
    "team2",
    "team1_win",
    "score1_match",
    "score2_match",
    "games_played",
    "bestOf",
)

_CONTRACT_COLUMNS = (
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


def _read_raw_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw file and separate series rows from map rows."""
    raw = pd.read_csv(path, parse_dates=["datetime"])
    maps = raw.loc[raw["is_total"] == False, :].copy()  # noqa: E712
    series = raw.loc[raw["is_total"] == True, _RAW_KEEP].copy()  # noqa: E712
    return series, maps


def _clean_series_rows(series: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate and unusable series rows before deriving outcomes."""
    before = len(series)
    series = series.drop_duplicates(subset="match_id", keep="first")
    if dropped := before - len(series):
        logger.info("dropped %d duplicate series rows", dropped)

    missing_teams = series["team1"].isna() | series["team2"].isna()
    if dropped := int(missing_teams.sum()):
        logger.info("dropped %d rows with missing team labels", dropped)
        series = series.loc[~missing_teams]
    return series


def _winner_from_series_scores(
    series: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return winner, team-1 score, and team-2 score from series evidence."""
    score1 = pd.to_numeric(series["score1_match"], errors="coerce")
    score2 = pd.to_numeric(series["score2_match"], errors="coerce")
    winner = pd.Series(pd.NA, index=series.index, dtype="object")
    winner[score1 > score2] = series.loc[score1 > score2, "team1"]
    winner[score2 > score1] = series.loc[score2 > score1, "team2"]
    return winner, score1, score2


def _winner_counts_from_maps(maps: pd.DataFrame, match_ids: pd.Series) -> pd.DataFrame:
    """Count map winners for matches whose series scores need a fallback."""
    maps = maps.copy()
    score1 = pd.to_numeric(maps["score1_game"], errors="coerce")
    score2 = pd.to_numeric(maps["score2_game"], errors="coerce")
    rounds_known = score1.notna() & score2.notna() & (score1 != score2)
    maps["map_winner"] = np.where(
        rounds_known & (score1 > score2),
        maps["team1"],
        np.where(
            rounds_known,
            maps["team2"],
            np.where(maps["team1_win"] == 1, maps["team1"], maps["team2"]),
        ),
    )
    return (
        maps.loc[maps["match_id"].isin(match_ids), ["match_id", "map_winner"]]
        .dropna()
        .groupby(["match_id", "map_winner"])
        .size()
        .unstack(fill_value=0)
    )


def _recover_missing_winners(
    series: pd.DataFrame,
    maps: pd.DataFrame,
    winner: pd.Series,
    score1: pd.Series,
    score2: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Use map-level evidence to repair tied or missing series scores."""
    missing = winner.isna()
    if not missing.any():
        return winner, score1, score2

    logger.info(
        "%d series rows with tied/missing scores — deriving winner from map-level rows",
        int(missing.sum()),
    )
    counts = _winner_counts_from_maps(maps, series.loc[missing, "match_id"])

    for row_index in series.index[missing]:
        match_id = series.at[row_index, "match_id"]
        if match_id not in counts.index:
            continue

        match_counts = counts.loc[match_id]
        team1_wins = match_counts.get(series.at[row_index, "team1"], 0)
        team2_wins = match_counts.get(series.at[row_index, "team2"], 0)
        if team1_wins > team2_wins:
            winner.loc[row_index] = series.at[row_index, "team1"]
        elif team2_wins > team1_wins:
            winner.loc[row_index] = series.at[row_index, "team2"]
        else:
            continue

        # Keep the repaired scores team-oriented, matching the loader contract.
        score1.loc[row_index] = float(team1_wins)
        score2.loc[row_index] = float(team2_wins)

    return winner, score1, score2


def _drop_invalid_series(
    series: pd.DataFrame,
    winner: pd.Series,
    score1: pd.Series,
    score2: pd.Series,
) -> pd.DataFrame:
    """Attach derived values and discard rows that cannot support the contract."""
    series = series.copy()
    series["winner"] = winner
    if unresolved := int(series["winner"].isna().sum()):
        logger.info("dropping %d rows with undeterminable winner", unresolved)
        series = series.dropna(subset=["winner"])

    series["t1_series_score"] = score1
    series["t2_series_score"] = score2
    games_played = pd.to_numeric(series["games_played"], errors="coerce")
    inconsistent = ((series["t1_series_score"] + series["t2_series_score"]) != games_played).fillna(
        False
    )
    if dropped := int(inconsistent.sum()):
        logger.info("dropping %d rows where scores contradict games_played", dropped)
        series = series.loc[~inconsistent]
    return series


def _finalize_series(series: pd.DataFrame) -> pd.DataFrame:
    """Apply the public schema, dtypes, stable ordering, and index reset."""
    series = series.drop(columns=["team1_win"])
    nullable_integer_columns = (
        "score1_match",
        "score2_match",
        "t1_series_score",
        "t2_series_score",
        "bestOf",
    )
    for column in nullable_integer_columns:
        series[column] = series[column].astype("Int64")

    # A missing games_played means one map in this source (DATA.md Quirk 2).
    series["games_played"] = (
        pd.to_numeric(series["games_played"], errors="coerce").fillna(1).astype("int64")
    )
    return series[list(_CONTRACT_COLUMNS)].sort_values("datetime", kind="mergesort").reset_index(
        drop=True
    )


def load_matches(path: str | Path) -> pd.DataFrame:
    """Load a raw Kaggle matches CSV and return the normalized series table.

    Steps:
    1. read CSV (datetime is ISO; scores come as floats due to NaN holes)
    2. keep only series rows (is_total == True) -> one row per match
    3. dedupe on match_id (keep first), log the count dropped
    4. winner = team with more maps (scores are TEAM-sorted, Quirk 1); the few
       tied/missing-score rows fall back to map-level evidence or get dropped
    5. cast scores to int, sort by datetime

    Raises FileNotFoundError naturally if `path` does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"no such file: {src}")

    series, maps = _read_raw_tables(src)
    series = _clean_series_rows(series)
    winner, score1, score2 = _winner_from_series_scores(series)
    winner, score1, score2 = _recover_missing_winners(
        series, maps, winner, score1, score2
    )
    series = _drop_invalid_series(series, winner, score1, score2)
    return _finalize_series(series)


def normalize_team_name(name: str | None) -> str | None:
    """Canonicalize a team name: strip whitespace, lowercase; None stays None.

    EXPOSED UTILITY, not a pipeline step: `load_matches` prefers the id/`teams.csv`
    resolution path (see DATA.md Quirk 6) and does not call this. Callers who want
    a casing/whitespace-only canonical form may import it directly.
    """
    if name is None:
        return None
    return str(name).strip().lower()
