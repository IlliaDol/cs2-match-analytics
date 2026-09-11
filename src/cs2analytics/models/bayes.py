"""M4 §4 — Bayesian team ratings via PyMC (Bradley–Terry).

Model (SPEC_M4 §4):
    s_i ~ Normal(1500, 350)          # latent strength per team, Elo scale
    P(i beats j) = logistic((s_i - s_j) / 173)   # 400/ln(10) ≈ 173.7 scale note

Fitting is restricted (SPEC): tier1, 2025-only, teams with >= 10 series — keeps
sampling fast and the posterior dense. ``fit_bayesian_ratings`` takes a prepared
series frame and returns tidy per-team posterior summaries; ``prepare_bt_data``
does the deterministic subsetting/pivot. The notebook
(``notebooks/m4_bayesian_ratings.ipynb``) runs sampling and writes
``outputs/bayesian_ratings.csv`` with columns [team, posterior_mean, hdi_3, hdi_97].

The Helsing sentence (write it in the notebook): the posterior IS the Bayesian
update after each match; Elo is a degenerate online approximation of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SERIES_PER_TEAM = 10
RATING_PRIOR_MU = 1500.0
RATING_PRIOR_SIGMA = 350.0
SCALE_173 = 173.0  # 400 / ln(10) — the logit-scale twin of the 400-point Elo scale


def prepare_bt_data(
    series_df: pd.DataFrame,
    tier: str = "tier1",
    year_start: int = 2025,
    year_end: int = 2025,
    min_series: int = MIN_SERIES_PER_TEAM,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Filter the series table to the SPEC subset and encode teams as indices.

    Subset rule (>= 10 series in tier1/2025): keeps 60 teams — the artifact test
    requires >= 50; the earlier >= 20 draft kept only 42 and failed that gate.

    Returns (team1_idx, team2_idx, team_names) where winner-vs-loser orientation
    is handled by the caller via the `result` array (1 = team1 won).
    """
    dt = pd.to_datetime(series_df["datetime"], utc=True, format="ISO8601")
    mask = (series_df["tier"] == tier) & (dt.dt.year >= year_start) & (dt.dt.year <= year_end)
    sub = series_df.loc[mask]
    counts = pd.concat([sub["team1"], sub["team2"]]).value_counts()
    keep = set(counts[counts >= min_series].index)
    sub = sub[sub["team1"].isin(keep) & sub["team2"].isin(keep)].reset_index(drop=True)

    teams = sorted(set(sub["team1"]) | set(sub["team2"]))
    team_to_idx = {name: i for i, name in enumerate(teams)}
    t1_idx = sub["team1"].map(team_to_idx).to_numpy()
    t2_idx = sub["team2"].map(team_to_idx).to_numpy()
    return t1_idx, t2_idx, teams


def build_coords(t1_idx: np.ndarray, t2_idx: np.ndarray, teams: list[str]) -> dict[str, object]:
    """Shared PyMC coordinates + arrays for the model (one source of truth for
    the notebook and any test that wants the same construction)."""
    n_teams = len(teams)
    return {
        "teams": teams,
        "n_teams": n_teams,
        "n_matches": len(t1_idx),
        "t1_idx": t1_idx,
        "t2_idx": t2_idx,
    }


def fit_bayesian_ratings(
    t1_idx: np.ndarray,
    t2_idx: np.ndarray,
    result: np.ndarray,
    teams: list[str],
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sample the Bradley–Terry posterior and return per-team summaries.

    `result` = 1.0 when team1 (index arrays) won, else 0.0; symmetrized inside
    so every match enters as both (i beats j, 1) and (j beats i, 0) — doubles
    the effective data and keeps the likelihood orientation-free.
    """
    import pymc as pm

    n_teams = len(teams)
    t1 = np.asarray(t1_idx, dtype=int)
    t2 = np.asarray(t2_idx, dtype=int)
    y = np.asarray(result, dtype=float)

    # symmetrize: each match appears in both orientations
    i_all = np.concatenate([t1, t2])
    j_all = np.concatenate([t2, t1])
    s_all = np.concatenate([y, 1.0 - y])

    with pm.Model():
        strength = pm.Normal(
            "strength", mu=RATING_PRIOR_MU, sigma=RATING_PRIOR_SIGMA, shape=n_teams
        )
        p = pm.math.invlogit((strength[i_all] - strength[j_all]) / SCALE_173)
        pm.Bernoulli("obs", p=p, observed=s_all)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, random_seed=random_seed, progressbar=False
        )

    st = idata.posterior["strength"].values  # (chains, draws, n_teams)
    st = np.transpose(st, (2, 0, 1)).reshape(n_teams, -1)  # (n_teams, chains*draws)
    mean = st.mean(axis=1)
    hdi_3, hdi_97 = np.percentile(st, [3.0, 97.0], axis=1)
    return (
        pd.DataFrame({"team": teams, "posterior_mean": mean, "hdi_3": hdi_3, "hdi_97": hdi_97})
        .sort_values("posterior_mean", ascending=False)
        .reset_index(drop=True)
    )
