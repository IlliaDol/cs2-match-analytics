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
    4. winner = team with more maps (scores are TEAM-sorted, Quirk 1); the few
       tied/missing-score rows fall back to map-level evidence or get dropped
    5. cast scores to int, sort by datetime

    Raises FileNotFoundError naturally if `path` does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"no such file: {src}")

    df = pd.read_csv(src, parse_dates=["datetime"])
    maps_all = df.loc[df["is_total"] == False, :].copy()  # noqa: E712

    # 2) series table only (Quirk 3: series rows are authoritative)
    series = df.loc[df["is_total"] == True, _RAW_KEEP].copy()  # noqa: E712

    # drop duplicated series rows (same match_id), keep first; log count
    n_before = len(series)
    series = series.drop_duplicates(subset="match_id", keep="first")
    n_dropped = n_before - len(series)
    if n_dropped:
        print(f"[loader] dropped {n_dropped} duplicate series rows")

    # drop matches with missing team labels first (1 row: match_id 10064713 —
    # both team fields empty in the raw file; a match without teams is unusable)
    n_bad_teams = int((series["team1"].isna() | series["team2"].isna()).sum())
    series = series.dropna(subset=["team1", "team2"])
    if n_bad_teams:
        print(f"[loader] dropped {n_bad_teams} rows with missing team labels")

    # --- winner (Quirk 1, corrected): series scores are TEAM-sorted, the
    # team with MORE maps won the series. 100% consistent with games_played.
    s1 = pd.to_numeric(series["score1_match"], errors="coerce")
    s2 = pd.to_numeric(series["score2_match"], errors="coerce")
    winner = pd.Series(pd.NA, index=series.index, dtype="object")
    winner[s1 > s2] = series.loc[s1 > s2, "team1"]
    winner[s2 > s1] = series.loc[s2 > s1, "team2"]

    # fallback for the few rows where scores are tied/missing (6 of 9,922):
    # map-level rows are trustworthy (round scores + a 98.4%-reliable flag).
    broken = winner.isna()
    if broken.any():
        print(
            f"[loader] {int(broken.sum())} series rows with tied/missing scores — "
            f"deriving winner from map-level rows"
        )
        maps_all["score1_game"] = pd.to_numeric(maps_all["score1_game"], errors="coerce")
        maps_all["score2_game"] = pd.to_numeric(maps_all["score2_game"], errors="coerce")
        rounds_ok = (
            maps_all["score1_game"].notna()
            & maps_all["score2_game"].notna()
            & (maps_all["score1_game"] != maps_all["score2_game"])
        )
        # map row: rounds are team-sorted too -> team with more rounds won the map
        maps_all["map_winner"] = np.where(
            rounds_ok & (maps_all["score1_game"] > maps_all["score2_game"]),
            maps_all["team1"],
            np.where(
                rounds_ok,
                maps_all["team2"],
                np.where(maps_all["team1_win"] == 1, maps_all["team1"], maps_all["team2"]),
            ),
        )
        map_winner_by_match = maps_all.loc[
            maps_all["match_id"].isin(series.loc[broken, "match_id"]),
            ["match_id", "map_winner"],
        ].dropna()
        wins = map_winner_by_match.groupby(["match_id", "map_winner"]).size().unstack(fill_value=0)
        for idx in series.index[broken]:
            mid = series.at[idx, "match_id"]
            if mid not in wins.index:
                continue
            row = wins.loc[mid]
            w1 = row.get(series.at[idx, "team1"], 0)
            w2 = row.get(series.at[idx, "team2"], 0)
            if w1 > w2:
                winner.loc[idx] = series.at[idx, "team1"]
                s1.loc[idx], s2.loc[idx] = float(w1), float(w2)  # heal corrupt scores
            elif w2 > w1:
                winner.loc[idx] = series.at[idx, "team2"]
                s1.loc[idx], s2.loc[idx] = float(w1), float(w2)
            # w1 == w2 (genuine Bo2 draw / forfeit) -> stays NA and gets dropped

    series["winner"] = winner
    n_unresolved = int(series["winner"].isna().sum())
    if n_unresolved:
        print(f"[loader] dropping {n_unresolved} rows with undeterminable winner")
        series = series.dropna(subset=["winner"])

    # team-specific series scores: identity columns (score1 = team1's maps)
    series["t1_series_score"] = s1
    series["t2_series_score"] = s2

    # corrupt-record guard: scores must sum to games_played (held for 100% of
    # raw decided rows; catches fallback-healed rows that violate it)
    gp = pd.to_numeric(series["games_played"], errors="coerce")
    bad_sum = (series["t1_series_score"] + series["t2_series_score"]) != gp
    bad_sum = bad_sum.fillna(False)
    if bad_sum.any():
        n_bad = int(bad_sum.sum())
        print(f"[loader] dropping {n_bad} rows where scores contradict games_played")
        series = series.loc[~bad_sum]

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
