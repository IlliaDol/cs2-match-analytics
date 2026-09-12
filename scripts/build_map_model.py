"""Fit + evaluate the map-level model and write outputs/m11_map_model.csv.

Pipeline:
1. Elo replay at MAP granularity (pre-map ratings, strictly backwards).
2. Per-team×map win-rate from maps strictly before the current one.
3. Logistic P(team1 wins map) on (elo_diff, map_wr_diff, map-name one-hot).
4. Lift to series using series_from_map_probs on the trustworthy map rows,
   then compare that series probability against the flat-lr baseline.

From repo root:
    .venv/Scripts/python.exe scripts/build_map_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cs2analytics.evaluation.metrics import log_loss
from cs2analytics.models.map_level import (
    _trustworthy_map_rows,
    map_name_baseline,
    map_win_model,
)

REPO = Path(__file__).resolve().parents[1]


def _elo_diff_mapwise(map_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-map elo_diff to each map row by replaying Elo at map granularity."""
    out = map_rows.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    from cs2analytics.features.elo import update_rating

    ratings: dict[str, float] = {}
    elo = np.zeros(len(out))
    for i, row in out.iterrows():
        ra = ratings.get(row["team1"], 1500.0)
        rb = ratings.get(row["team2"], 1500.0)
        elo[i] = ra - rb
        # map result oriented to team1
        s = 1.0 if row["team1_win"] == 1 else 0.0
        ratings[row["team1"]] = update_rating(ra, rb, s, k=32.0)
        ratings[row["team2"]] = update_rating(rb, ra, 1.0 - s, k=32.0)
    out["elo_diff"] = elo
    return out


def _team_map_wr(map_rows: pd.DataFrame) -> pd.DataFrame:
    """teamX_map_wr = that team's win share on THIS map name strictly before now."""
    out = map_rows.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    # rolling cumulative per (team, map_name)
    hist: dict[tuple[str, str], tuple[int, int]] = {}
    t1_wr = np.zeros(len(out))
    t2_wr = np.zeros(len(out))
    for i, row in out.iterrows():
        key1 = (row["team1"], row["map_name"])
        key2 = (row["team2"], row["map_name"])
        w1, n1 = hist.get(key1, (0, 0))
        w2, n2 = hist.get(key2, (0, 0))
        t1_wr[i] = w1 / n1 if n1 else 0.5
        t2_wr[i] = w2 / n2 if n2 else 0.5
        # update AFTER reading (strictly previous maps)
        s1 = 1 if row["team1_win"] == 1 else 0
        hist[key1] = (w1 + s1, n1 + 1)
        hist[key2] = (w2 + (1 - s1), n2 + 1)
    out["team1_map_wr"] = t1_wr
    out["team2_map_wr"] = t2_wr
    return out


def main() -> None:
    raw = pd.read_csv(REPO / "data" / "raw" / "cs2_all_tiers_games.csv")
    maps = _trustworthy_map_rows(raw)
    maps = maps.dropna(subset=["map_name", "team1", "team2", "team1_win"])
    maps = _elo_diff_mapwise(maps)
    maps = _team_map_wr(maps)

    # careful: rich rows carry what map_win_model needs
    fit = map_win_model(maps)
    m = fit["model"]

    # evaluate: split maps by the same 2026-01-01 boundary as the repo
    maps["datetime"] = pd.to_datetime(maps["datetime"], utc=True)
    te = maps[maps["datetime"] >= pd.Timestamp("2026-01-01", tz="UTC")].reset_index(drop=True)
    if len(te) < 50:
        print("[map] too few test maps; writing descriptive table only")
        te = maps.tail(2000).reset_index(drop=True)
    baseline = map_name_baseline(maps)
    X_te = fit["build_X"](te)
    X_te = X_te.reindex(columns=fit["features"], fill_value=0.0)
    p_map = m.predict_proba(X_te[fit["features"]])[:, 1]
    p_flat = np.full(len(te), float(te["team1_win"].mean()))
    rows = [
        {
            "n_test_maps": len(te),
            "map_logloss": log_loss(p_map, (te["team1_win"] == 1).astype(int)),
            "flat_baseline_logloss": log_loss(p_flat, (te["team1_win"] == 1).astype(int)),
            "n_maps_in_train": fit["n_rows"],
            "n_unique_maps": len(baseline),
        }
    ]
    pd.DataFrame(rows).to_csv(REPO / "outputs" / "m11_map_model.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    # map-name baselines as a small reference artifact
    pd.Series(baseline, name="p_team1").to_csv(REPO / "outputs" / "map_name_baseline.csv")


if __name__ == "__main__":
    main()
