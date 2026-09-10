"""Data loading and schema contracts.

Contract (enforced by tests/test_loader_contract.py and documented in DATA.md):
- load_matches() returns the SERIES table: exactly one row per match_id.
- Book score columns are winner-sorted (Quirk 1); this loader disentangles them
  into team-specific scores t1_series_score / t2_series_score.
- bestOf is unreliable for Bo1s (Quirk 2); games_played is the truth for maps played.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Columns kept from the raw file, in contract order (game_id / is_total are
# intentionally dropped: the loader returns the series table only).
_RAW_KEEP = [
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
]

_CONTRACT_COLUMNS = [
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
]


def load_matches(path: str | Path) -> pd.DataFrame:
    """Load a raw Kaggle matches CSV and return the normalized series table.

    Steps:
    1. read CSV (datetime is ISO; scores come as floats due to NaN holes)
    2. keep only series rows (is_total == True) -> one row per match
    3. dedupe on match_id (keep first), log the count dropped
    4. disentangle winner-sorted scores into team-specific series scores
    5. derive `winner` team name, cast scores to int, sort by datetime

    Raises FileNotFoundError naturally if `path` does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"no such file: {src}")

    df = pd.read_csv(src, parse_dates=["datetime"])

    # 2) series table only (Quirk 3: series rows are authoritative)
    series = df.loc[df["is_total"] == True, _RAW_KEEP].copy()  # noqa: E712

    # drop duplicated series rows (same match_id), keep first; log count
    n_before = len(series)
    series = series.drop_duplicates(subset="match_id", keep="first")
    n_dropped = n_before - len(series)
    if n_dropped:
        print(f"[loader] dropped {n_dropped} duplicate series rows")

    # winner-sorted scores -> team-specific scores (Quirk 1):
    #    (score1_match, score2_match) = (winner's score, loser's score)
    # drop matches with missing team labels first (1 row: match_id 10064713 —
    # both team fields empty in the raw file; a match without teams is unusable)
    n_bad_teams = int((series["team1"].isna() | series["team2"].isna()).sum())
    series = series.dropna(subset=["team1", "team2"])
    if n_bad_teams:
        print(f"[loader] dropped {n_bad_teams} rows with missing team labels")

    t1_won = series["team1_win"] == 1
    series["t1_series_score"] = np.where(
        t1_won,
        np.maximum(series["score1_match"], series["score2_match"]),
        np.minimum(series["score1_match"], series["score2_match"]),
    )
    series["t2_series_score"] = np.where(
        t1_won,
        np.minimum(series["score1_match"], series["score2_match"]),
        np.maximum(series["score1_match"], series["score2_match"]),
    )
    series["winner"] = np.where(t1_won, series["team1"], series["team2"])

    # tidy up: contract columns, dtypes, order
    series = series.drop(columns=["team1_win"])
    int_cols = [
        "score1_match",
        "score2_match",
        "t1_series_score",
        "t2_series_score",
        "games_played",
        "bestOf",
    ]
    for col in int_cols:
        series[col] = series[col].astype("Int64")  # nullable int: no floats
    series["games_played"] = series["games_played"].fillna(1).astype("int64")

    series = series[_CONTRACT_COLUMNS]
    series = series.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    return series


def normalize_team_name(name: str | None) -> str | None:
    """Canonicalize a team name: strip whitespace, lowercase; None stays None."""
    if name is None:
        return None
    return str(name).strip().lower()
