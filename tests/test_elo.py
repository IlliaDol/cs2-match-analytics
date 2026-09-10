"""Module 4 (M4) — pure-math tests for the Elo engine.

These run EVERYWHERE (no data needed) and pin exact numbers from
docs/SPEC_M4_elo_engine.md. The human implements src/cs2analytics/features/elo.py,
src/cs2analytics/evaluation/metrics.py until green.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cs2analytics.evaluation.metrics import accuracy_from_probs, brier_score, log_loss
from cs2analytics.features.elo import (
    bo3_win_probability,
    expected_score,
    run_elo_backtest,
    update_rating,
)

REPO = Path(__file__).parent.parent
NB = REPO / "notebooks" / "elo_derivation.ipynb"


# --- expected_score -----------------------------------------------------------


def test_expected_score_equal_ratings_is_half():
    assert expected_score(1500.0, 1500.0) == 0.5


def test_expected_score_worked_example_1650_vs_1750():
    # 1 / (1 + 10^(100/400)) = 1 / (1 + 10**0.25) = 1 / 2.77828...
    assert expected_score(1650.0, 1750.0) == pytest.approx(0.359935, abs=1e-5)


def test_expected_score_dominant_pair():
    assert expected_score(2800.0, 1500.0) > 0.99


def test_expected_score_symmetry():
    rng = np.random.default_rng(42)
    for ra, rb in zip(rng.uniform(1000, 2500, 50), rng.uniform(1000, 2500, 50), strict=True):
        assert expected_score(ra, rb) == pytest.approx(1.0 - expected_score(rb, ra))


def test_expected_score_monotone_in_difference():
    diffs = np.linspace(-600, 600, 200)
    values = [expected_score(1500 + d, 1500) for d in diffs]
    assert all(a < b for a, b in zip(values, values[1:], strict=False))


# --- update_rating ------------------------------------------------------------


def test_update_rating_winner_1650_vs_1750():
    # E_A = 0.359935; R' = 1650 + 32 * (1 - 0.359935) = 1670.4821
    assert update_rating(1650.0, 1750.0, result=1.0, k=32.0) == pytest.approx(1670.4821, abs=1e-3)


def test_update_rating_loser_is_symmetric():
    expected_loss = 1650.0 - 32.0 * expected_score(1650.0, 1750.0)
    assert update_rating(1650.0, 1750.0, result=0.0, k=32.0) == pytest.approx(expected_loss)


def test_update_rating_draw_keeps_ratings():
    assert update_rating(1500.0, 1500.0, result=0.5, k=32.0) == pytest.approx(1500.0)


def test_update_rating_higher_k_moves_more():
    small = abs(update_rating(1600.0, 1600.0, 1.0, k=8) - 1600.0)
    big = abs(update_rating(1600.0, 1600.0, 1.0, k=64) - 1600.0)
    assert big > small


# --- bo3_win_probability --------------------------------------------------------


def test_bo3_win_probability_degenerate_cases():
    assert bo3_win_probability(0.0) == 0.0
    assert bo3_win_probability(1.0) == 1.0
    assert bo3_win_probability(0.5) == pytest.approx(0.5)


def test_bo3_win_probability_055():
    # p^2 * (3 - 2p) = 0.3025 * 1.9 = 0.57475
    assert bo3_win_probability(0.55) == pytest.approx(0.57475, abs=1e-6)


def test_bo3_win_probability_simulation_agrees():
    """Monte-Carlo check: simulate Bo3s and compare with the closed form."""
    rng = np.random.default_rng(7)
    p = 0.55
    sims = 200_000
    maps = rng.random((sims, 3)) < p
    # best of 3: first to 2 wins
    w = (maps[:, 0].astype(int) + maps[:, 1]) >= 1
    win = ((maps[:, 0] & maps[:, 1]) | ((maps[:, 0] | maps[:, 1]) & maps[:, 2])).mean()
    closed = bo3_win_probability(p)
    assert abs(win - closed) < 0.005
    del w


# --- metrics: exact values ------------------------------------------------------


def test_log_loss_exact_values():
    assert log_loss([0.75], [1]) == pytest.approx(-np.log(0.75), abs=1e-9)
    assert log_loss([0.75], [0]) == pytest.approx(-np.log(0.25), abs=1e-9)


def test_log_loss_never_explodes():
    assert np.isfinite(log_loss([1.0], [0]))  # clipped, not inf


def test_brier_score_exact_values():
    assert brier_score([0.75], [1]) == pytest.approx(0.0625)
    assert brier_score([0.5], [1]) == pytest.approx(0.25)


def test_accuracy_from_probs():
    assert accuracy_from_probs([0.6, 0.3, 0.7], [1, 0, 1]) == pytest.approx(1.0)
    assert accuracy_from_probs([0.6, 0.3, 0.7], [0, 1, 1]) == pytest.approx(1 / 3)


# --- run_elo_backtest on the tracked toy dataset ---------------------------------


def test_backtest_toy_dataset_chronological_and_complete():
    toy = REPO / "data" / "raw" / "toy_matches.csv"
    if not toy.exists():
        pytest.fail("toy dataset missing from repo (it must stay tracked)")
    toy_df = pd.read_csv(toy)
    # Build the series view of the toy data: one row per match_id, winner from
    # score comparison, datetime from the earliest map of the match.
    toy_series = (
        toy_df.sort_values("game_id")
        .groupby(["match_id", "team1", "team2"], as_index=False)
        .agg(datetime=("datetime", "first"), score1=("score1", "sum"), score2=("score2", "sum"))
    )
    toy_series["winner"] = toy_series.apply(
        lambda r: r["team1"] if r["score1"] > r["score2"] else r["team2"], axis=1
    )
    toy_series["datetime"] = pd.to_datetime(toy_series["datetime"])
    res = run_elo_backtest(toy_series, k=32.0)
    assert len(res) == len(toy_series)
    assert res["p_t1"].between(0, 1).all()
    assert not res[["elo_t1_pre", "elo_t2_pre"]].isna().any().any()
    # first match must use the base rating for both teams
    first = res.sort_values("datetime").iloc[0]
    assert first["elo_t1_pre"] == pytest.approx(1500.0)
    assert first["elo_t2_pre"] == pytest.approx(1500.0)


# --- derivation notebook must exist and contain the math -------------------------


@pytest.mark.skipif(not (REPO / "notebooks").exists(), reason="notebooks dir not created yet")
def test_elo_derivation_notebook_contains_required_elements():
    if not NB.exists():
        pytest.skip("notebooks/elo_derivation.ipynb not written yet — REQUIRED by M4 spec §1")
    raw = json.loads(NB.read_text(encoding="utf-8"))
    text = " ".join("".join(cell.get("source", [])) for cell in raw.get("cells", []))
    for needle in (
        r"\frac{1}{1+10",  # expected-score formula
        "400",  # scale constant
        "1670",  # worked example result (1650 + 32*(1-0.36))
        "173",  # Bradley-Terry scale equivalence
    ):
        assert needle in text, f"derivation notebook missing required element: {needle}"
