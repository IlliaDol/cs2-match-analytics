"""Map-level signal + hierarchical series model (P1.5).

Two layers:

1. `map_win_model` — P(team1 wins a map) as a logistic function of (elo_diff,
   a per-team×map win-rate differential, and a map-name baseline), fit on the
   98% of map rows whose rounds+flag agree (Quirk 4). All features are PRE-map.

2. `series_from_map_probs` — feed per-map win probabilities into a best-of-n
   formula so a series probability is a *function of map probabilities*, not a
   single flat number. Bo1 = p1; Bo3/Bo5 = independent-map binomial over the
   ordered map slate (an honest approximation, documented, not a claim).

Leakage law: every map's features use only maps strictly before it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _trustworthy_map_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Map rows where rounds and team1_win agree (DATA.md Quirk 4)."""
    m = raw[raw["is_total"] == False].copy()  # noqa: E712 — dataset Boolean column
    has_r = m["score1_game"].notna() & m["score2_game"].notna()
    has_f = m["team1_win"].notna()
    agree = has_r & has_f & ((m["score1_game"] > m["score2_game"]) == (m["team1_win"] == 1))
    return m.loc[agree]


def map_win_model(map_rows: pd.DataFrame, min_maps_per_team: int = 5) -> dict:
    """Fit logistic P(team1 wins map) on (elo_diff, map_winrate_diff, map_name dummmies).

    Returns {"model": fitted sklearn model, "features": [...], "map_rows": n}.
    `map_rows` must already contain elo_diff per row (join upstream from Elo replay).
    """
    df = map_rows.dropna(subset=["elo_diff", "map_name", "team1_map_wr", "team2_map_wr"])
    if len(df) < 100:
        raise ValueError("fewer than 100 trustworthy map rows after filtering")

    def _build(mdf: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(
            {
                "elo_diff": mdf["elo_diff"].to_numpy(),
                "map_wr_diff": (mdf["team1_map_wr"] - mdf["team2_map_wr"]).to_numpy(),
            }
        )
        d = pd.get_dummies(mdf["map_name"], prefix="map").astype(float)
        return pd.concat([X, d], axis=1)

    X = _build(df)
    y = (df["team1_win"] == 1).astype(int).to_numpy()
    model = LogisticRegression(max_iter=1000).fit(X, y)
    return {
        "model": model,
        "features": list(X.columns),
        "n_rows": len(df),
        "build_X": _build,
    }


def map_name_baseline(map_rows: pd.DataFrame) -> dict[str, float]:
    """P(team1 wins) per map name — the raw pool baseline, for the series model."""
    g = map_rows.groupby("map_name")["team1_win"].mean()
    return {k: float(v) for k, v in g.items()}


def series_from_map_probs(
    p1: float, bo: int, n_maps: int | None = None
) -> float:
    """P(team1 wins the series) from a single-map win probability under best-of.

    Best-of-n with n = 2*ceil(bo/2): team1 wins by taking ceil(n/2) maps.
    Bo1: just p1. Bo3: 2 or 3 maps won; Bo5: 3, 4 or 5. Independent-map
    binomial (documented approximation — vetoes and momentum are not modeled).
    """
    p1 = float(np.clip(p1, 1e-6, 1 - 1e-6))
    if bo == 1:
        return p1
    # best-of-3: first to 2 (max 3 maps); best-of-5: first to 3 (max 5 maps).
    # Independent-map binomial: P(team1 wins) = sum_{w=ceil(k)}^{2k-1} C(2k-1,w) p^w (1-p)^(2k-1-w)
    k = (bo + 1) // 2
    total = 0.0
    import math  # noqa: PLC0415

    for wins in range(k, 2 * k):
        total += math.comb(2 * k - 1, wins) * (p1**wins) * ((1 - p1) ** (2 * k - 1 - wins))
    return float(np.clip(total, 0.0, 1.0))
