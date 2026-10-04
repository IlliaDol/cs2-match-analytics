"""Contract tests for the causal roster-change study."""

from __future__ import annotations

import pandas as pd
import pytest

from cs2analytics.causal import (
    balance_table,
    bootstrap_did,
    build_event_windows,
    build_team_match_panel,
    did_estimate,
    event_study_table,
    nearest_control_matches,
)


def _matches() -> pd.DataFrame:
    rows: list[dict] = []
    lineups = {
        "A": ["a1", "a2", "a3", "a4", "a5"],
        "B": ["b1", "b2", "b3", "b4", "b5"],
    }
    for index in range(5):
        a = lineups["A"].copy()
        if index >= 3:
            a[-1] = "a6"
        rows.append(
            {
                "match_id": f"m{index}",
                "datetime": f"2026-01-{index + 1:02d}",
                "team1": "A",
                "team2": "B",
                "winner": "A" if index % 2 == 0 else "B",
                "t1_series_score": 2 if index % 2 == 0 else 0,
                "t2_series_score": 0 if index % 2 == 0 else 2,
                "tier": "tier1",
                "games_played": 2,
                **{f"team1_player{i + 1}_id": player for i, player in enumerate(a)},
                **{f"team2_player{i + 1}_id": player for i, player in enumerate(lineups["B"])},
            }
        )
    return pd.DataFrame(rows)


def test_roster_change_requires_known_prior_and_counts_replacements():
    panel = build_team_match_panel(_matches())
    a = panel[panel["team"] == "A"].sort_values("datetime").reset_index(drop=True)
    assert not bool(a.loc[0, "roster_changed"])
    assert not bool(a.loc[1, "roster_changed"])
    assert bool(a.loc[3, "roster_changed"])
    assert a.loc[3, "change_count"] == 1
    assert a.loc[3, "pre_stable_matches"] == 2


def test_unknown_lineup_is_not_treated_as_stable():
    matches = _matches()
    matches.loc[2, "team1_player1_id"] = None
    panel = build_team_match_panel(matches)
    a = panel[panel["team"] == "A"].sort_values("datetime").reset_index(drop=True)
    assert not bool(a.loc[2, "roster_known"])
    assert not bool(a.loc[3, "roster_changed"])


def test_event_window_is_aligned_and_rejects_contamination():
    panel = build_team_match_panel(_matches())
    events = build_event_windows(panel, pre_window=2, post_window=1, min_pre_stable=2)
    assert len(events) == 4
    assert events["relative_match"].tolist() == [-2, -1, 0, 1]
    assert events["post"].tolist() == [0, 0, 1, 1]
    assert events["event_id"].nunique() == 1
    assert events["treated"].eq(1).all()


def test_event_window_requires_enough_history():
    panel = build_team_match_panel(_matches())
    events = build_event_windows(panel, pre_window=3, post_window=2, min_pre_stable=2)
    assert events.empty


def test_did_returns_transparent_four_cell_estimate():
    panel = pd.DataFrame(
        {
            "treated": [1, 1, 0, 0],
            "post": [0, 1, 0, 1],
            "win": [0.4, 0.6, 0.5, 0.55],
        }
    )
    result = did_estimate(panel)
    assert result["treated_change"] == pytest.approx(0.2)
    assert result["control_change"] == pytest.approx(0.05)
    assert result["estimate"] == pytest.approx(0.15)


def test_bootstrap_did_is_reproducible():
    panel = pd.DataFrame(
        {
            "event_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "treated": [1, 1, 0, 0, 1, 1, 0, 0],
            "post": [0, 1, 0, 1, 0, 1, 0, 1],
            "win": [0.4, 0.6, 0.5, 0.55, 0.3, 0.7, 0.4, 0.5],
        }
    )
    first = bootstrap_did(panel, n_boot=200, seed=7)
    second = bootstrap_did(panel, n_boot=200, seed=7)
    assert first == second
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]


def test_event_study_and_balance_tables_are_reportable():
    panel = pd.DataFrame(
        {
            "relative_match": [-1, -1, 0, 0],
            "treated": [1, 0, 1, 0],
            "win": [0.5, 0.4, 0.6, 0.45],
            "rating": [1500, 1490, 1500, 1490],
        }
    )
    study = event_study_table(panel)
    assert study.loc[study["relative_match"] == 0, "difference"].iloc[0] == pytest.approx(0.15)
    balance = balance_table(panel.assign(treated=panel["treated"]), ["rating"])
    assert balance.loc[0, "smd"] > 0 or balance.loc[0, "smd"] == float("inf")


def test_nearest_control_match_is_deterministic():
    panel = build_team_match_panel(_matches())
    treated = build_event_windows(panel, pre_window=2, post_window=1, min_pre_stable=2)
    matches = nearest_control_matches(
        panel,
        treated,
        covariates=["pre_stable_matches"],
        max_date_days=10,
        pre_window=2,
        post_window=1,
    )
    assert len(matches) == 1
    assert matches.loc[0, "control_team"] == "B"
