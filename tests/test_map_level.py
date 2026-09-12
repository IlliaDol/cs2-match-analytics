"""Map-level signal contract tests (P1.5): the series formula + row filter."""

from __future__ import annotations

import pandas as pd

from cs2analytics.models.map_level import _trustworthy_map_rows, series_from_map_probs


def test_series_from_map_probs_bo1():
    assert series_from_map_probs(0.7, 1) == 0.7


def test_series_from_map_probs_symmetry():
    # swapping the map probability must swap the series probability
    a, b = series_from_map_probs(0.6, 3), series_from_map_probs(0.4, 3)
    assert abs((a + b) - 1.0) < 1e-9


def test_series_from_map_probs_monotone_in_bo():
    # a favourite's series chance rises as the series gets longer
    p5 = series_from_map_probs(0.7, 5)
    p3 = series_from_map_probs(0.7, 3)
    p1 = series_from_map_probs(0.7, 1)
    assert p5 > p3 > p1


def test_series_from_map_probs_clips():
    assert 0.0 < series_from_map_probs(0.9999, 3) < 1.0


def test_trustworthy_map_rows_filters_disagreements():
    df = pd.DataFrame(
        {
            "is_total": [False, False, False],
            "score1_game": [16.0, 8.0, 16.0],
            "score2_game": [8.0, 16.0, 16.0],
            "team1_win": [1, 1, 0],  # row 1 disagrees (score2 > score1 but team1_win=1)
        }
    )
    out = _trustworthy_map_rows(df)
    assert len(out) == 2  # the disagreement row is dropped
