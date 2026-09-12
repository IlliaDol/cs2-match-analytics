"""Model-vs-market scaffold (P2.8).

The v1 bookmaker-odds subset is too small to compare against, so this module is
a CONTRACT, not a result: it defines the metrics and the join the comparison
will need once a usable odds dataset exists. It never fabricates odds, and the
evaluation fails loudly if the input is empty rather than faking a number.

Agreed convention (kept identical to the rest of the repo):
- All probabilities are P(team1 wins the series), oriented to the same `team1`.
- Market probability = 1 / decimal_odds, then de-vigged (normalized) so
  p_team1 + p_team2 = 1 — the Sharpe/normalization method documented here.
- Model probability = whatever the lr+roster (or later) model outputs.

Metrics returned: logloss of model vs outcome, logloss of market vs outcome,
Brier both, and the model-vs-market Brier/logloss DELTA (negative = model beats
market on that metric).
"""

from __future__ import annotations

import pandas as pd

from cs2analytics.evaluation.metrics import brier_score, log_loss


def de_vig(p1: float, p2: float) -> tuple[float, float]:
    """Remove the bookmaker margin by normalizing implied probabilities to sum 1."""
    total = p1 + p2
    if total <= 0:
        raise ValueError("implied probabilities must be positive and non-zero")
    return p1 / total, p2 / total


def model_vs_market(
    df: pd.DataFrame,
    model_p: str = "model_p_t1",
    market_p: str = "market_p_t1",
    outcome: str = "result",
) -> dict[str, float]:
    """Compare model and market on the same series rows.

    `df` must contain a `market_p_t1` column (already de-vigged by the caller or
    via `de_vig`) and a `result` column of 0/1 team1-win outcomes.
    """
    if df.empty or market_p not in df.columns:
        raise ValueError(
            "no market odds available — the v1 odds subset is too small; see README Limitations"
        )
    y = df[outcome].to_numpy(dtype=float)
    pm = df[model_p].to_numpy(dtype=float)
    pk = df[market_p].to_numpy(dtype=float)
    return {
        "n": int(df.shape[0]),
        "model_logloss": log_loss(pm, y),
        "market_logloss": log_loss(pk, y),
        "logloss_delta": log_loss(pm, y) - log_loss(pk, y),
        "model_brier": brier_score(pm, y),
        "market_brier": brier_score(pk, y),
        "brier_delta": brier_score(pm, y) - brier_score(pk, y),
    }


def odds_to_market_probability(decimal_odds: float) -> float:
    """One side of the de-vig: raw implied probability of a single decimal odd."""
    return 1.0 / max(decimal_odds, 1.001)
