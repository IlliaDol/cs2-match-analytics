"""Model-vs-market contract tests (P2.8). No odds data needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cs2analytics.evaluation.market import de_vig, model_vs_market, odds_to_market_probability


def test_de_vig_normalizes():
    p1, p2 = de_vig(0.5, 0.6)  # vig of 10% total
    assert abs((p1 + p2) - 1.0) < 1e-9


def test_odds_to_market_probability():
    assert odds_to_market_probability(2.0) == 0.5
    assert 0.0 < odds_to_market_probability(1.5) < 1.0


def test_model_beats_market_returns_negative_delta():
    rng = np.random.default_rng(0)
    n = 500
    base = rng.uniform(0.25, 0.75, n)
    y = (rng.uniform(0, 1, n) < base).astype(float)  # outcome drawn from base
    model_p = base  # a perfect (oracle) model
    market_p = np.full(n, 0.5)  # the no-skill market: always 50/50
    df = pd.DataFrame({"model_p_t1": model_p, "market_p_t1": market_p, "result": y})
    out = model_vs_market(df)
    assert out["logloss_delta"] < 0, "model should beat a 50/50 market here"


def test_missing_odds_raises():
    df = pd.DataFrame({"model_p_t1": [0.5], "result": [1]})
    with pytest.raises(ValueError, match="no market odds"):
        model_vs_market(df)
