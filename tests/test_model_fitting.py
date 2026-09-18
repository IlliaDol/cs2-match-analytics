"""Synthetic-data unit tests for the model-FITTING code (B.1 improvement).

These never touch data/raw — they fabricate ~200 rows and check that every
model-fitting function imports, fits, and returns sane shapes. They are gated
only on the package (which CI installs via `pip install -e ".[dev,ml]"`), never
on the private dataset, so a fresh clone runs REAL modeling code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_series(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = [f"team_{i:03d}" for i in range(8)]
    match_ids = [f"m{i:05d}" for i in range(n)]
    t1 = rng.choice(teams, n)
    t2 = rng.choice(teams, n)
    # avoid self-matches
    swaps = t1 == t2
    t2[swaps] = rng.choice([t for t in teams if t != t1[swaps][0]], sum(swaps))
    dates = pd.date_range("2025-01-01", periods=n, freq="7D", tz="UTC")
    elo1 = rng.normal(1500, 150, n)
    elo2 = rng.normal(1500, 150, n)
    prob = 1.0 / (1.0 + np.exp(-(elo1 - elo2) / 250.0))
    winner = np.where(rng.uniform(0, 1, n) < prob, t1, t2)
    return pd.DataFrame(
        {
            "match_id": match_ids,
            "datetime": dates,
            "team1": t1,
            "team2": t2,
            "winner": winner,
            "tier": "tier1",
            "games_played": rng.choice([1, 3], n),
            "score1_match": rng.choice([1, 2], n),
            "score2_match": rng.choice([0, 1], n),
        }
    )


def _synthetic_maps(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    teams = [f"team_{i:03d}" for i in range(8)]
    t1 = rng.choice(teams, n)
    t2 = rng.choice(teams, n)
    maps = pd.DataFrame(
        {
            "match_id": [f"m{i:05d}" for i in range(n)],
            "datetime": pd.date_range("2025-01-01", periods=n, freq="3D", tz="UTC"),
            "team1": t1,
            "team2": t2,
            "map_name": rng.choice(["Mirage", "Nuke", "Dust2"], n),
            "is_total": False,
            "team1_win": rng.integers(0, 2, n),
            "score1_game": rng.integers(0, 17, n),
            "score2_game": rng.integers(0, 17, n),
        }
    )
    return maps


def test_fit_logistic_and_predict():
    from cs2analytics.models.logistic import fit_logistic, predict_proba

    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    model = fit_logistic(X, y)
    p = predict_proba(model, X[:10])
    assert p.shape == (10,)
    assert ((p >= 0) & (p <= 1)).all()


def test_fit_gbm_and_isotonic():
    from cs2analytics.models.gbm import fit_gbm, fit_gbm_isotonic, predict_proba

    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    m = fit_gbm(X, y)
    p = predict_proba(m, X[:10])
    assert p.shape == (10,)
    m_iso = fit_gbm_isotonic(X, y)
    p_iso = predict_proba(m_iso, X[:10])
    assert p_iso.shape == (10,)


def test_elo_backtest_and_score_variant():
    from cs2analytics.features.elo import run_elo_backtest
    from cs2analytics.models.backtest import score_variant, summarize

    s = _synthetic_series()
    bt = run_elo_backtest(s, k=32.0)
    assert len(bt) == len(s)
    assert {"p_t1", "result", "datetime"} <= set(bt.columns)
    scored = score_variant(s, "elo_k16")
    summary = summarize(scored, "test")
    assert summary["n"] > 0 and 0.0 <= summary["logloss"] <= 2.0


def test_run_map_specific_elo():
    from cs2analytics.models.backtest import run_map_specific_elo

    out = run_map_specific_elo(_synthetic_maps())
    assert len(out) <= 200
    assert {"p_t1", "result", "map_name"} <= set(out.columns)


def test_build_backtest_table():
    from cs2analytics.models.backtest import build_backtest_table

    s = _synthetic_series()
    table = build_backtest_table(s)
    assert "elo_k32" in set(table["model"])
    assert "constant_0.5" in set(table["model"])


def test_prepare_bt_data_and_fit_bayes():
    pytest.importorskip("pymc")
    from cs2analytics.models.bayes import fit_bayesian_ratings, prepare_bt_data

    s = _synthetic_series(300)
    t1_idx, t2_idx, teams = prepare_bt_data(s, min_series=2)
    assert len(teams) >= 2
    result = np.ones(len(t1_idx))  # all team1 wins for a deterministic quick sample
    out = fit_bayesian_ratings(t1_idx, t2_idx, result, teams, draws=50, tune=50, chains=2)
    assert {"team", "posterior_mean", "hdi_3", "hdi_97"} <= set(out.columns)
