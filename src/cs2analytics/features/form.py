"""M6 §3 — rolling pre-match features. Leakage law: look BACKWARDS only.

Every function takes an explicit reference timestamp and must never use any
match at or after it. These become the form/rest/h2h columns of the M7 feature
matrix.
"""

from __future__ import annotations

import pandas as pd


def _team_history(matches: pd.DataFrame, team: str, ref_ts: pd.Timestamp) -> pd.DataFrame:
    """All of `team`'s series strictly before ref_ts, sorted chronologically."""
    dt = pd.to_datetime(matches["datetime"])
    mask = (dt < ref_ts) & ((matches["team1"] == team) | (matches["team2"] == team))
    return matches.loc[mask].assign(_dt=dt.loc[mask]).sort_values("_dt", kind="mergesort")


def rolling_form(
    matches: pd.DataFrame,
    team: str,
    n: int = 5,
    ref_ts: pd.Timestamp | None = None,
) -> float:
    """Win share over the team's last n series strictly BEFORE ref_ts.

    Teams with no prior matches return 0.5 (uninformative prior) — matches the
    DuckDB q2 contract.
    """
    ref = pd.Timestamp(ref_ts)
    dt = pd.to_datetime(matches["datetime"])
    mask = (dt < ref) & ((matches["team1"] == team) | (matches["team2"] == team))
    hist = matches.loc[mask].assign(_dt=dt.loc[mask]).sort_values("_dt", kind="mergesort")
    last = hist.tail(n)
    if last.empty:
        return 0.5
    wins = (last["winner"] == team).astype(float)
    return float(wins.mean())


def days_rest(matches: pd.DataFrame, team: str, ref_ts: pd.Timestamp) -> int | None:
    """Whole days between the team's previous series (strictly before ref) and ref.

    None if this is the team's first observed series. Cost note: same-day back-
    to-back series return 0 — a real schedule-crunch signal.
    """
    ref = pd.Timestamp(ref_ts)
    dt = pd.to_datetime(matches["datetime"])
    mask = (dt < ref) & ((matches["team1"] == team) | (matches["team2"] == team))
    hist = dt.loc[mask]
    if hist.empty:
        return None
    return int((ref - hist.max()).total_seconds() // 86_400)


def h2h_record(
    matches: pd.DataFrame,
    team_a: str,
    team_b: str,
    before_ts: pd.Timestamp,
) -> tuple[int, int]:
    """(a_wins, b_wins) in series between the two teams strictly before before_ts."""
    ref = pd.Timestamp(before_ts)
    dt = pd.to_datetime(matches["datetime"])
    between = (dt < ref) & (
        ((matches["team1"] == team_a) & (matches["team2"] == team_b))
        | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    )
    sub = matches.loc[between]
    a_wins = int((sub["winner"] == team_a).sum())
    b_wins = int((sub["winner"] == team_b).sum())
    return a_wins, b_wins
