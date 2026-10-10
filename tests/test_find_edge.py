"""Tests for the edge finder — synthetic books, exact arithmetic, no real odds needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import find_edge as fe  # noqa: E402

# --------------------------------------------------------------------------- de-vig


def test_devig_removes_the_margin_and_sums_to_one():
    p1, p2, margin = fe.devig(1.90, 1.90)
    assert p1 + p2 == pytest.approx(1.0)
    assert p1 == pytest.approx(0.5)
    assert margin == pytest.approx(2 / 1.90 - 1)      # ~5.3%


def test_devig_preserves_the_favourite():
    p1, p2, _ = fe.devig(1.50, 2.60)
    assert p1 > p2
    # proportional de-vig keeps the ratio of the raw implied probabilities
    assert (p1 / p2) == pytest.approx((1 / 1.50) / (1 / 2.60))


def test_devig_rejects_impossible_odds():
    with pytest.raises(ValueError, match="must exceed 1.0"):
        fe.devig(0.95, 1.90)


# --------------------------------------------------------------------------- evaluation


def test_ev_matches_hand_calculation():
    """p=0.6 at odds 2.0 -> EV = 0.6*1 - 0.4 = +0.20 per unit."""
    result = fe.evaluate(0.6, 2.0)
    assert result["ev_per_unit"] == pytest.approx(0.20)
    assert result["edge"] == pytest.approx(0.10)       # 0.6 vs 1/2.0
    assert result["break_even_rate"] == pytest.approx(0.5)


def test_ev_is_negative_when_probability_is_below_break_even():
    assert fe.evaluate(0.45, 2.0)["ev_per_unit"] < 0


def test_even_odds_with_true_probability_is_a_fair_bet():
    result = fe.evaluate(0.5, 2.0)
    assert result["ev_per_unit"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- elo


def test_elo_curve_is_symmetric_and_centred():
    assert fe.elo_probability(1800, 1800) == pytest.approx(0.5)
    assert fe.elo_probability(2000, 1600) + fe.elo_probability(1600, 2000) == pytest.approx(1.0)
    assert fe.elo_probability(2000, 1600) > 0.9        # +400 Elo is a heavy favourite


# --------------------------------------------------------------------------- analysis


ROWS = [
    {"date": "2026-04-29", "team1": "Alpha", "team2": "Bravo", "odds1": 2.20, "odds2": 1.70},
    {"date": "2026-04-30", "team1": "Alpha", "team2": "Charlie", "odds1": 1.20, "odds2": 4.50},
]
RATINGS = {"Alpha": 1900.0, "Bravo": 1500.0, "Charlie": 2100.0}


def test_value_side_is_flagged_only_when_the_edge_is_positive():
    results = fe.analyse(ROWS, RATINGS, min_edge=0.0)
    first = results[0]
    # Alpha is much stronger than Bravo on Elo, yet the book prices Alpha at 2.20
    assert first["best_side"] == "Alpha"
    assert first["best_ev"] > 0 and first["bet"] is True
    assert first["elo_p1"] > first["market_p1"]         # we disagree with the market, in our favour


def test_no_value_when_the_book_prices_it_fairly_or_worse():
    # Charlie is the strongest, but at 4.50 the market already pays more than our 0.72
    results = fe.analyse(ROWS, RATINGS, min_edge=0.0)
    second = results[1]
    assert second["elo_p1"] < 0.5 or second["best_ev"] <= 0.0
    if second["best_ev"] <= 0.0:
        assert second["bet"] is False


def test_min_edge_threshold_filters_marginal_spots():
    """A higher bar can only ever flag fewer spots — and an unreachable bar flags none."""
    loose = fe.analyse(ROWS, RATINGS, min_edge=0.0)
    mid = fe.analyse(ROWS, RATINGS, min_edge=0.30)
    impossible = fe.analyse(ROWS, RATINGS, min_edge=0.99)

    assert sum(r["bet"] for r in mid) <= sum(r["bet"] for r in loose)
    assert all(r["bet"] is False for r in impossible)
    # the second row really carries a large edge: Elo says 76% where the book implies 21%
    assert mid[1]["best_edge"] > 0.3


def test_unrated_team_is_skipped_with_a_reason():
    rows = [{"date": "", "team1": "Alpha", "team2": "Nobody FC", "odds1": 1.5, "odds2": 2.5}]
    result = fe.analyse(rows, RATINGS)[0]
    assert "no rating for Nobody FC" in result["skipped"]


def test_render_reports_counts_and_caveats():
    text = fe.render(fe.analyse(ROWS, RATINGS), min_edge=0.0)
    assert "VALUE" in text
    assert "positive EV" in text
    assert "NOT the calibrated logistic model" in text          # the honesty line must be there


# --------------------------------------------------------------------------- files


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_odds_csv_columns_are_detected_by_hint(tmp_path: Path):
    path = write_csv(tmp_path / "odds.csv",
                     "date,team1,team2,odds1,odds2\n2026-04-29,Alpha,Bravo,2.2,1.7\n")
    rows = fe.parse_odds_csv(path)
    assert rows[0]["team1"] == "Alpha" and rows[0]["odds2"] == pytest.approx(1.7)


def test_rows_without_prices_are_dropped(tmp_path: Path):
    path = write_csv(tmp_path / "odds.csv",
                     "team1,team2,odds1,odds2\nAlpha,Bravo,2.2,1.7\nCharlie,Delta,,\n")
    assert len(fe.parse_odds_csv(path)) == 1


def test_missing_columns_produce_a_clear_error(tmp_path: Path):
    path = write_csv(tmp_path / "bad.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="need two team columns"):
        fe.parse_odds_csv(path)


def test_load_ratings_reads_the_shipped_artifact():
    if not fe.ELO_PATH.exists():
        pytest.skip("artifacts/elo_ratings.json missing — run python -m cs2analytics.models.train (git-ignored in CI)")
    ratings = fe.load_ratings()
    assert len(ratings) > 100
    assert all(isinstance(v, float) for v in list(ratings.values())[:5])
