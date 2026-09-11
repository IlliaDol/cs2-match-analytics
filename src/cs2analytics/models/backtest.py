"""M4 §3 — backtest report harness: Elo variants, summaries, the results table.

This module does NOT re-implement the Elo engine (see features/elo.py). It wraps
``run_elo_backtest`` with the things the report needs:

- ``TIME_SPLIT``            — the one split law of the repo (train < 2025-08-01 <= test)
- ``score_variant``         — score a named variant (fixed-k Elo or map-specific Elo)
- ``run_map_specific_elo``  — Elo replay at map granularity, aggregated to series level
- ``summarize``             — logloss / brier / acc / n on one split, via evaluation.metrics
- ``build_backtest_table``  — the exact table the M4 report notebook exports

Leakage law: every variant records PRE-match (or PRE-map) ratings and predictions only.
Ratings keep evolving across the split boundary — that is correct walk-forward behavior,
since ratings at any moment are a function of the past only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cs2analytics.evaluation.metrics import accuracy_from_probs, brier_score, log_loss
from cs2analytics.features.elo import expected_score, run_elo_backtest, update_rating

TIME_SPLIT = pd.Timestamp("2025-08-01")

#: fixed-k Elo variants: model label -> K
K_VARIANTS: dict[str, float] = {
    "elo_k8": 8.0,
    "elo_k16": 16.0,
    "elo_k32": 32.0,
    "elo_k64": 64.0,
}

_TABLE_COLUMNS = ["model", "split", "logloss", "brier", "acc", "n"]


def _validate_pairs(series_df: pd.DataFrame) -> None:
    """Series rows must name the winner among the two teams, one row per match."""
    winner_is_t1 = series_df["winner"] == series_df["team1"]
    winner_is_t2 = series_df["winner"] == series_df["team2"]
    if not (winner_is_t1 | winner_is_t2).all():
        bad = int((~(winner_is_t1 | winner_is_t2)).sum())
        raise ValueError(f"{bad} series rows have a winner outside {{team1, team2}}")
    if series_df["match_id"].duplicated().any():
        raise ValueError("series table must have exactly one row per match_id")


def _split_mask(df: pd.DataFrame, split: str) -> np.ndarray:
    """train = strictly before TIME_SPLIT, test = on/after (tz-safe)."""
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    dt = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    boundary = TIME_SPLIT.tz_localize("UTC")
    if split == "train":
        return (dt < boundary).to_numpy()
    return (dt >= boundary).to_numpy()


def summarize(pred_df: pd.DataFrame, split: str) -> dict[str, float | int]:
    """Metrics of a scored frame (needs datetime, p_t1, result) on one split.

    Computed via ``evaluation.metrics`` from the recorded probabilities — never
    from the convenience per-row loss columns, so a broken column cannot hide.
    """
    mask = _split_mask(pred_df, split)
    sub = pred_df.loc[mask]
    p, y = sub["p_t1"], sub["result"]
    return {
        "logloss": log_loss(p, y),
        "brier": brier_score(p, y),
        "acc": accuracy_from_probs(p, y),
        "n": int(len(sub)),
    }


def run_map_specific_elo(
    games: pd.DataFrame,
    k: float = 32.0,
    base: float = 1500.0,
) -> pd.DataFrame:
    """Elo replay where each (team, map) pair owns its own rating.

    Input: map-level rows of the games table (``is_total=False``), columns
    match_id, datetime, team1, team2, map_name, team1_win, score1_game, score2_game.

    Winner of each map = the team with more rounds on that map (round scores are
    TEAM-sorted — DATA.md quirk 4 correction, 2026-09-11); the per-map flag is
    used only for forfeit rows without usable rounds.
    Output: one row per played map with pre-map prediction columns, so
    ``summarize`` works on it directly (n counts maps, not series).
    """
    required = {
        "match_id",
        "datetime",
        "team1",
        "team2",
        "map_name",
        "team1_win",
        "score1_game",
        "score2_game",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games table missing columns: {sorted(missing)}")

    maps = games.loc[~games["is_total"].astype(bool)].copy()
    maps = maps.loc[maps["map_name"].notna()]
    maps["score1_game"] = pd.to_numeric(maps["score1_game"], errors="coerce")
    maps["score2_game"] = pd.to_numeric(maps["score2_game"], errors="coerce")
    maps["datetime"] = pd.to_datetime(maps["datetime"], utc=True, format="ISO8601")
    maps = maps.sort_values(["datetime", "match_id"], kind="mergesort").reset_index(drop=True)

    # DATA.md Quirk 4 (corrected): rounds are TEAM-sorted (score1_game = team1's
    # rounds, verified 6/6 in both Major finals) and agree with the flag on
    # ~98.5% of rows. Winner source: round-score direction when usable, the
    # flag only on forfeit rows (0-0 rounds, no usable direction).
    t1_won_flag = maps["team1_win"].astype(int) == 1
    s1 = maps["score1_game"].to_numpy(dtype=float)
    s2 = maps["score2_game"].to_numpy(dtype=float)
    rounds_known = ~(np.isnan(s1) | np.isnan(s2))
    rounds_say_t1 = s1 > s2
    t1_won = np.where(rounds_known, rounds_say_t1, t1_won_flag.to_numpy())

    ratings: dict[tuple[str, str], float] = {}
    n = len(maps)
    pre_t1 = np.empty(n)
    pre_t2 = np.empty(n)
    p_t1 = np.empty(n)
    result = np.empty(n)

    for i in range(n):
        team1, team2 = maps.at[i, "team1"], maps.at[i, "team2"]
        map_name = maps.at[i, "map_name"]
        key1, key2 = (team1, map_name), (team2, map_name)
        ra = ratings.get(key1, base)
        rb = ratings.get(key2, base)
        e = expected_score(ra, rb)
        s = 1.0 if t1_won[i] else 0.0

        pre_t1[i], pre_t2[i], p_t1[i], result[i] = ra, rb, e, s
        ratings[key1] = update_rating(ra, rb, s, k)
        ratings[key2] = update_rating(rb, ra, 1.0 - s, k)

    out = maps[["match_id", "datetime", "team1", "team2", "map_name"]].copy()
    out["elo_t1_pre"] = pre_t1
    out["elo_t2_pre"] = pre_t2
    out["p_t1"] = p_t1
    out["result"] = result
    return out


def score_variant(
    df: pd.DataFrame,
    variant: str,
    k: float = 32.0,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score one named variant; returns a frame ``summarize`` understands.

    - ``elo_k8/k16/k32/k64``: series-level ``run_elo_backtest`` with that K.
    - ``elo_mapspecific_k32``: map-level replay via ``run_map_specific_elo``
      (needs ``games``). One row per played map, not per series.
    """
    if variant in K_VARIANTS:
        return run_elo_backtest(df, k=K_VARIANTS[variant])
    if variant == "elo_mapspecific_k32":
        if games is None:
            raise ValueError(f"{variant} needs the map-level games table (games=...)")
        return run_map_specific_elo(games, k=k)
    raise ValueError(
        f"unknown variant {variant!r}; known: {sorted(K_VARIANTS) + ['elo_mapspecific_k32']}"
    )


def constant_half_baseline(pred_df: pd.DataFrame, split: str) -> dict[str, float | int]:
    """The always-predict-0.5 row — computed, never hardcoded."""
    mask = _split_mask(pred_df, split)
    sub = pred_df.loc[mask]
    n = len(sub)
    p = np.full(n, 0.5)
    y = sub["result"].to_numpy(dtype=float)
    return {
        "logloss": log_loss(p, y),
        "brier": brier_score(p, y),
        "acc": accuracy_from_probs(p, y),
        "n": n,
    }


def build_backtest_table(
    series_df: pd.DataFrame,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The M4 report table: model × split with logloss / brier / acc / n.

    Rows: elo_k32 (train + test), elo_k8/k16/k64 + elo_mapspecific_k32 +
    constant_0.5 (test). Elo must beat constant_0.5 on test logloss.
    """
    _validate_pairs(series_df)

    rows: list[dict[str, float | int | str]] = []
    k32 = run_elo_backtest(series_df, k=32.0)
    rows.append({"model": "elo_k32", "split": "train", **summarize(k32, "train")})
    rows.append({"model": "elo_k32", "split": "test", **summarize(k32, "test")})

    for variant in ("elo_k8", "elo_k16", "elo_k64"):
        scored = score_variant(series_df, variant)
        rows.append({"model": variant, "split": "test", **summarize(scored, "test")})

    if games is not None:
        maps_scored = score_variant(series_df, "elo_mapspecific_k32", games=games)
        rows.append(
            {"model": "elo_mapspecific_k32", "split": "test", **summarize(maps_scored, "test")}
        )

    rows.append({"model": "constant_0.5", "split": "test", **constant_half_baseline(k32, "test")})

    return pd.DataFrame(rows)[_TABLE_COLUMNS].sort_values(["model", "split"]).reset_index(drop=True)
