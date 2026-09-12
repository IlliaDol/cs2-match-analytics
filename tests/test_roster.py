"""Roster-feature contract tests (P0.2 improvement). No private data needed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cs2analytics.features.roster import build_roster_features


def _frame() -> pd.DataFrame:
    """Two series, same two teams, known 5-player lineups."""
    data: dict = {
        "match_id": ["m1", "m2"],
        "datetime": pd.to_datetime(["2026-01-01", "2026-01-08"], utc=True),
        "team1": ["A", "A"],
        "team2": ["B", "B"],
    }
    for s in (1, 2):
        base = 100 if s == 1 else 200
        for i in range(1, 6):
            data[f"team{s}_player{i}_id"] = [base + i, base + i]
    return pd.DataFrame(data)


def test_first_series_has_unknown_stability():
    df = _frame()
    out = build_roster_features(df)
    first = out.iloc[0]
    assert pd.isna(first["team1_stability"]) and pd.isna(first["team1_standin"])


def test_same_lineup_is_full_stability_no_standin():
    df = _frame()
    out = build_roster_features(df)
    second = out.iloc[1]
    assert second["team1_stability"] == 1.0
    assert second["team1_standin"] == 0.0
    assert second["team2_stability"] == 1.0


def test_two_changes_flags_standin():
    df = _frame()
    # team1 replaces players 4 and 5 in m2 -> shares 3/5 -> standin
    df.loc[1, "team1_player4_id"] = 999
    df.loc[1, "team1_player5_id"] = 998
    out = build_roster_features(df)
    second = out.iloc[1]
    assert second["team1_stability"] == 0.6
    assert second["team1_standin"] == 1.0
    # team2 unchanged
    assert second["team2_stability"] == 1.0
    assert second["team2_standin"] == 0.0


def test_missing_lineup_yields_nan_even_with_history():
    df = _frame()
    df.loc[1, "team1_player1_id"] = np.nan
    out = build_roster_features(df)
    second = out.iloc[1]
    assert pd.isna(second["team1_stability"])
    assert pd.isna(second["team1_standin"])
    # team2 still fine
    assert second["team2_stability"] == 1.0
