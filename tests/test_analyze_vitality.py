"""Contracts for the Vitality window review (scripts/analyze_vitality.py).

Pure-function tests with synthetic frames — no disk reads, no real data. Three of these are
regressions for bugs found while building the report:

* the series `winner` field is a team NAME, so orienting it by side inverted every series
  Vitality played as team2 (the report first said 42-11 instead of 53-11);
* a 0-1 Bo1 defeat is not a "clean sweep", and it must not be counted twice;
* KAST arrives on a 0-100 scale, so it has to be rescaled before percent formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_vitality as av  # noqa: E402

TEAM = av.TEAM
OTHER = "Some Other Team"


def series_row(match_id, when, team1, team2, winner, s1, s2, best_of=3, event="Test Cup", tier=1):
    """One synthetic series row with every column vital_series() touches."""
    return {
        "match_id": match_id,
        "datetime": pd.Timestamp(when, tz="UTC"),
        "tournament": event,
        "tier": tier,
        "bestOf": best_of,
        "team1": team1,
        "team2": team2,
        "winner": winner,
        "t1_series_score": s1,
        "t2_series_score": s2,
        "games_played": s1 + s2,
        "elo_t1_pre": 2000.0,
        "elo_t2_pre": 1800.0,
        "p_t1": 0.75,
    }


def to_frame(rows) -> pd.DataFrame:
    return av.vital_series(pd.DataFrame(rows))


# --------------------------------------------------------------------------- orientation


def test_vitality_as_team2_win_is_a_win():
    """Regression: XOR-ing the team-NAME `winner` field with the side flag inverted this."""
    frame = to_frame([series_row(1, "2025-09-01", OTHER, TEAM, TEAM, 0, 2)])
    assert frame.loc[0, "won"] is True or bool(frame.loc[0, "won"]) is True
    assert frame.loc[0, "opponent"] == OTHER
    assert frame.loc[0, "score_for"] == 2 and frame.loc[0, "score_against"] == 0


def test_vitality_as_team2_loss_is_a_loss():
    frame = to_frame([series_row(1, "2025-09-01", TEAM, OTHER, OTHER, 1, 2)])
    assert bool(frame.loc[0, "won"]) is False
    assert frame.loc[0, "loss_margin"] == 1          # 2 - 1: lost by a single map


def test_scores_are_oriented_to_vitality_both_ways():
    frame = to_frame(
        [
            series_row(1, "2025-09-01", TEAM, OTHER, TEAM, 2, 1),      # team1 side
            series_row(2, "2025-09-02", OTHER, TEAM, TEAM, 1, 2),      # team2 side, same result
        ]
    )
    assert frame["score_for"].tolist() == [2, 2]
    assert frame["score_against"].tolist() == [1, 1]
    assert frame["won"].tolist() == [True, True]
    assert frame["opponent"].tolist() == [OTHER, OTHER]


def test_window_filter_is_inclusive_of_both_named_days():
    frame = to_frame(
        [
            series_row(1, "2025-08-19", TEAM, OTHER, TEAM, 2, 0),      # before
            series_row(2, "2025-08-20", TEAM, OTHER, TEAM, 2, 0),      # first day
            series_row(3, "2026-05-05", TEAM, OTHER, TEAM, 2, 0),      # last day
            series_row(4, "2026-05-06", TEAM, OTHER, TEAM, 2, 0),      # after
        ]
    )
    assert frame["match_id"].tolist() == [2, 3]


def test_elo_and_expected_win_follow_the_side():
    frame = to_frame([series_row(1, "2025-09-01", OTHER, TEAM, TEAM, 0, 3)])
    assert frame.loc[0, "elo"] == 1800.0 and frame.loc[0, "opp_elo"] == 2000.0
    assert frame.loc[0, "p_win"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- closeness


def _vitality_and_maps(series_rows, map_rows):
    vs = to_frame(series_rows)
    # closeness() expects the map columns even when there are no maps (those cases test the
    # series-level counts only).
    vm = pd.DataFrame(map_rows) if map_rows else pd.DataFrame(columns=["close", "overtime", "won"])
    if not vm.empty:
        vm["won"] = vm["won"].astype(bool)
    return vs, vm


def test_sweeps_and_single_map_defeats_are_disjoint():
    """Regression: a 0-1 Bo1 defeat is not a sweep, and it was counted in both buckets."""
    rows = [
        series_row(1, "2025-09-01", TEAM, OTHER, OTHER, 0, 2, best_of=3),      # clean sweep
        series_row(2, "2025-09-02", TEAM, OTHER, OTHER, 2, 3, best_of=5),      # 2-3, one map
        series_row(3, "2025-09-03", TEAM, OTHER, OTHER, 0, 1, best_of=1),      # Bo1, one map
    ]
    vs, vm = _vitality_and_maps(rows, [])
    close = av.closeness(vm, vs)
    assert close["sweeps_against"] == 1
    assert close["one_map_losses"] == 2
    assert close["sweeps_against"] + close["one_map_losses"] == len(vs)        # no double count
    assert close["bo1"] == 1


def test_sweeps_for_needs_a_series_long_enough_to_sweep():
    vs, vm = _vitality_and_maps(
        [
            series_row(1, "2025-09-01", TEAM, OTHER, TEAM, 1, 0, best_of=1),
            series_row(2, "2025-09-02", TEAM, OTHER, TEAM, 2, 0, best_of=3),
        ],
        [],
    )
    assert av.closeness(vm, vs)["sweeps_for"] == 1


# --------------------------------------------------------------------------- helpers


def test_streaks_take_the_longest_runs():
    rows = [
        series_row(i, f"2025-09-0{i}", TEAM, OTHER, TEAM if i < 4 else OTHER, 2, 0)
        for i in range(1, 7)
    ]
    frame = to_frame(rows)
    assert av.streaks(frame) == (3, 3)


def test_map_rows_use_the_side_relative_flag():
    """On map rows `team1_win` IS side-relative, so the XOR is the correct form there."""
    columns = {"match_id": 1, "game_id": 10, "datetime": pd.Timestamp("2025-09-01", tz="UTC"),
               "tournament": "Test Cup", "team1_id": 1, "team2_id": 2, "map_name": "Dust2",
               "bestOf": 3}
    # scores and flag must agree: row A is Vitality on team2 winning 13-7, row B is its mirror
    raw = pd.DataFrame([{**columns, "team1": OTHER, "team2": TEAM, "team1_win": False,
                         "score1_game": 7.0, "score2_game": 13.0},
                        {**columns, "game_id": 11, "team1": TEAM, "team2": OTHER,
                         "team1_win": True, "score1_game": 13.0, "score2_game": 7.0}])
    for i in range(1, 6):
        for stat in ("kills", "deaths", "assists", "adr", "kast"):
            raw[f"team1_player{i}_{stat}"] = 1.0
            raw[f"team2_player{i}_{stat}"] = 1.0
    raw.loc[raw["game_id"] == 10, ["team2_player1_kast"]] = 77.44          # 0-100 scale

    av._RAW_CACHE["pair"] = (raw, raw["team1"] == TEAM)
    try:
        maps = av.load_maps()
    finally:
        av._RAW_CACHE.pop("pair", None)

    assert maps["won"].tolist() == [True, True]        # team2 lost => Vitality won, and vice versa
    assert maps["team_kast"].max() <= 1.0              # rescaled to a fraction
    assert maps["round_diff"].tolist() == [6.0, 6.0]


def test_md_table_formats_percentages_integers_and_missing():
    frame = pd.DataFrame(
        [{"name": "a", "rate": 0.5, "kills": 2768.0, "note": None},
         {"name": "b", "rate": 0.667, "kills": 3.0, "note": "x"}]
    )
    text = av.md_table(frame, {"name": "Name", "rate": "Rate", "kills": "K", "note": "Note"},
                       percent={"rate"})
    lines = text.splitlines()
    assert lines[0] == "| Name | Rate | K | Note |"
    assert "50.0%" in lines[2] and "2768" in lines[2] and "—" in lines[2]
    assert "2768.000" not in text and "0.667%" not in text
