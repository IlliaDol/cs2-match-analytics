"""M6 tests — cleaning functions + rolling form features.

The human implements src/cs2analytics/cleaning.py and src/cs2analytics/features/form.py.
Synthetic fixtures only — everything here runs in CI once the modules exist.

Until then the whole module skips (guarded import), so CI and the rest of the
suite stay green while M6 is pending.
"""

import pandas as pd
import pytest

try:
    from cs2analytics.cleaning import (
        deduplicate_series,
        flag_forfeits,
        normalize_team_names,
        winsorize_round_scores,
    )
    from cs2analytics.features.form import days_rest, h2h_record, rolling_form

    _M6_READY = True
    _M6_REASON = ""
except ModuleNotFoundError as _e:
    _M6_READY = False
    _M6_REASON = f"M6 modules not implemented yet ({_e.name}) — the human's build"

pytestmark = [
    pytest.mark.skipif(not _M6_READY, reason=_M6_REASON),
]


@pytest.fixture()
def dirty_series():
    return pd.DataFrame(
        {
            "match_id": [1, 1, 2, 4],
            "datetime": pd.to_datetime(
                ["2025-01-10", "2025-01-10", "2025-01-11", "2025-01-12"]
            ),
            "team1": ["NAVI", "NAVI", "FaZe", "G2"],
            "team2": ["FaZe", "FaZe", "NAVI", "NAVI"],
            "winner": ["NAVI", "NAVI", "FaZe", "NAVI"],
            "t1_series_score": [2, 2, 2, 0],
            "t2_series_score": [0, 0, 0, 0],
            "games_played": [2, 2, 3, 1],
        }
    )


# --- cleaning.py -----------------------------------------------------------------


def test_deduplicate_series_keeps_first_and_logs(dirty_series, capsys):
    out = deduplicate_series(dirty_series)
    assert out["match_id"].is_unique
    assert len(out) == 4
    assert "dropped" in capsys.readouterr().out.lower()


def test_deduplicate_series_returns_copy(dirty_series):
    out = deduplicate_series(dirty_series)
    out.loc[:, "winner"] = "X"
    assert (dirty_series["winner"] != "X").all()


def test_normalize_team_names_mapping():
    df = pd.DataFrame({"team1": ["NAVI", "Natus Vincere", "Unknown Five"]})
    mapping = {"natus vincere": "navi", "navi": "navi"}
    out = normalize_team_names(df, mapping)
    assert (out["team1"].iloc[0] == out["team1"].iloc[1])  # both -> navi
    assert out["team1"].iloc[2] == "Unknown Five"  # untouched


def test_flag_forfeits(dirty_series):
    out = flag_forfeits(dirty_series)
    assert "is_forfeit" in out.columns
    # the 0-0 single-game row is the forfeit
    assert out["is_forfeit"].tolist() == [False, False, False, True]


def test_winsorize_round_scores():
    df = pd.DataFrame(
        {"t1_series_score": [0, 2, 3, 99], "t2_series_score": [0, 0, 2, -5]}
    )
    out = winsorize_round_scores(df, lo=0, hi=25)
    assert out["t1_series_score"].max() == 25
    assert out["t2_series_score"].min() == 0
    assert len(out) == len(df)


# --- features/form.py --------------------------------------------------------------
# Synthetic schedule: team X plays 2025-02-01..06 daily vs varying opponents.
# X wins on D1, D2, loses D3, wins D4, wins D5. D6 is the reference match.


def _sched():
    return pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4, 5],
            "datetime": pd.to_datetime(
                ["2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04", "2025-02-05"]
            ),
            "team1": ["X", "X", "X", "X", "X"],
            "team2": ["A", "B", "C", "A", "B"],
            "winner": ["X", "X", "C", "X", "X"],
        }
    )


@pytest.fixture()
def schedule():
    return _sched()


def test_rolling_form_exact_fraction(schedule):
    # before 2025-02-05, X's last 5 = W W L W -> 3/4 = 0.75
    assert rolling_form(schedule, "X", n=5, ref_ts=pd.Timestamp("2025-02-05")) == 0.75


def test_rolling_form_n_smaller(schedule):
    # last 2 before 2025-02-04: L W -> 0.5
    assert rolling_form(schedule, "X", n=2, ref_ts=pd.Timestamp("2025-02-05")) == 0.5


def test_rolling_form_leakage_trap(schedule):
    """ref between D2 and D3 must see exactly D1, D2 -> 1.0"""
    assert rolling_form(schedule, "X", n=5, ref_ts=pd.Timestamp("2025-02-03")) == 1.0


def test_days_rest(schedule):
    assert days_rest(schedule, "X", ref_ts=pd.Timestamp("2025-02-05")) == 1


def test_days_rest_first_match_is_none(schedule):
    assert days_rest(schedule, "X", ref_ts=pd.Timestamp("2025-02-01")) in (None, -1)


def test_h2h_record(schedule):
    # vs A: X won both before 2025-02-06 -> (2, 0)
    assert h2h_record(schedule, "X", "A", before_ts=pd.Timestamp("2025-02-06")) == (2, 0)
    # vs C before 2025-02-04: match not played yet -> (0, 0)
    assert h2h_record(schedule, "X", "C", before_ts=pd.Timestamp("2025-02-04")) == (0, 0)
