"""Roster-shape features (M7/P0.2): roster stability + stand-ins.

Every value is computed strictly from lineups of matches BEFORE the reference
timestamp — the same leakage law as form.py. A lineup is the tuple of the five
player ids for one team on one series row, sorted so order/slot does not matter.

Features (per team side, before ref_ts):
- roster_stability: share of players in the CURRENT lineup who also played in the
  team's most-recent prior series with a known lineup.
- standin: 1 if the current lineup shares at most 3 players with that prior
  lineup (i.e. >=2 changes), else 0. NaN when either lineup is unknown.

Both degrade gracefully: unknown current or prior lineup -> NaN (the feature
store fills NaN the same way form5/rest already do).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _lineup(row: pd.Series, side: str) -> tuple[int, ...] | None:
    """Sorted tuple of that side's player ids, or None if any of the 5 are NaN."""
    ids = [row[f"{side}_player{i}_id"] for i in range(1, 6)]
    if any(pd.isna(x) for x in ids):
        return None
    return tuple(sorted(int(x) for x in ids))


def build_roster_features(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per series with team1/team2 roster_stability and standin flags.

    `matches` must be chronological and contain the 10 lineup id columns
    (team1_player1_id .. team2_player5_id) plus match_id, datetime, team1, team2.
    """
    df = matches.copy()
    df["_dt"] = pd.to_datetime(df["datetime"])

    rows: list[dict] = []
    # per-team rolling state: team -> (last_ts, last_lineup)
    last: dict[str, tuple[pd.Timestamp, tuple[int, ...] | None]] = {}

    for _, row in df.iterrows():
        ts = row["_dt"]
        rec: dict = {"match_id": row["match_id"]}
        for side in ("team1", "team2"):
            team = row[side]
            cur = _lineup(row, side)
            prev = last.get(team)
            _, prev_lu = prev if prev else (None, None)

            if cur is None or prev_lu is None:
                rec[f"{side}_stability"] = np.nan
                rec[f"{side}_standin"] = np.nan
            else:
                shared = len(set(cur) & set(prev_lu))
                rec[f"{side}_stability"] = shared / 5.0
                rec[f"{side}_standin"] = 1.0 if shared <= 3 else 0.0
            # update only if we actually saw a lineup this row
            if cur is not None:
                last[team] = (ts, cur)
        rows.append(rec)

    return pd.DataFrame(rows)
