"""Elo rating engine — pure functions.

M4 module. The math is derived in notebooks/elo_derivation.ipynb (REQUIRED reading):
- expected_score: the Elo logistic curve, E_A = 1 / (1 + 10^((R_B - R_A) / 400))
- update_rating:   one online step, R' = R + K (S - E)
- bo3_win_probability: closed form p^2 (3 - 2p) for best-of-3 from per-map p
- run_elo_backtest: walk-forward rating replay over a series table

Pure functions only — no I/O here except reading the dataframe passed in.
Leakage law: backtest records PRE-match ratings and predictions only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_score(ra: float, rb: float) -> float:
    r"""Elo expected score for player/team A.

    \(E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}\)

    Equal ratings give 0.5; a 400-point advantage gives 10/(1+10) = 10/11.
    """
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def update_rating(ra: float, rb: float, result: float, k: float = 32.0) -> float:
    """One Elo update for team A: R' = R + K (S - E).

    `result` is the actual score S (1 = win, 0 = loss, 0.5 = draw).
    """
    return ra + k * (result - expected_score(ra, rb))


def bo3_win_probability(p_map: float) -> float:
    r"""Probability of winning a best-of-3 given per-map win probability \(p\).

    Derived in the M4 notebook: \(P = p^2 (3 - 2p)\) — win in 2 (p^2) plus
    win in 3 (2 p^2 (1-p)); no draws exist in CS2 map play.
    """
    if not 0.0 <= p_map <= 1.0:
        raise ValueError(f"p_map must be in [0, 1], got {p_map}")
    return p_map * p_map * (3.0 - 2.0 * p_map)


def run_elo_backtest(
    df: pd.DataFrame,
    k: float = 32.0,
    base: float = 1500.0,
) -> pd.DataFrame:
    """Walk-forward Elo replay over a series table.

    Required input columns: match_id, datetime, team1, team2, winner.
    Output columns: match_id, datetime, team1, team2, winner, elo_t1_pre,
    elo_t2_pre, p_t1, result, logloss, brier.

    Leakage law: predictions use PRE-match ratings; updates happen after.
    """
    required = {"match_id", "datetime", "team1", "team2", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns for backtest: {sorted(missing)}")

    out = df.sort_values("datetime", kind="mergesort").reset_index(drop=True).copy()
    ratings: dict[str, float] = {}

    def get(name: str) -> float:
        return ratings.get(name, base)

    pre_t1 = np.empty(len(out), dtype=float)
    pre_t2 = np.empty(len(out), dtype=float)
    p_t1 = np.empty(len(out), dtype=float)
    results = np.empty(len(out), dtype=float)

    for i in out.index:
        ra, rb = get(out.at[i, "team1"]), get(out.at[i, "team2"])
        e_a = expected_score(ra, rb)
        s = 1.0 if out.at[i, "winner"] == out.at[i, "team1"] else 0.0

        pre_t1[i], pre_t2[i], p_t1[i], results[i] = ra, rb, e_a, s
        p_clipped = min(max(e_a, 1e-15), 1.0 - 1e-15)
        out.at[i, "logloss"] = float(-(s * np.log(p_clipped) + (1.0 - s) * np.log(1.0 - p_clipped)))
        out.at[i, "brier"] = (e_a - s) ** 2
        ratings[out.at[i, "team1"]] = update_rating(ra, rb, s, k)
        ratings[out.at[i, "team2"]] = update_rating(rb, ra, 1.0 - s, k)

    out["elo_t1_pre"] = pre_t1
    out["elo_t2_pre"] = pre_t2
    out["p_t1"] = p_t1
    out["result"] = results
    return out
