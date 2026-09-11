"""M6 §4 — build the v1 feature store: outputs/features_v1.parquet.

One row per series with PRE-match features only (leakage law):
    elo_diff, form5_diff, rest_days_diff, h2h_t1_win_share (+ metadata columns).

Vectorized construction (per-team chronological rolling ops, shifted by one so
every value uses strictly-earlier matches); form5 + h2h spot-verified against
the scalar reference functions in features/form.py on a random sample.
Run from repo root:
    .venv/Scripts/python.exe scripts/build_features_v1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cs2analytics.features.elo import run_elo_backtest
from cs2analytics.features.form import h2h_record, rolling_form

REPO = Path(__file__).resolve().parents[1]


def _long_format(series: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team-side): datetime, match_id, team, win."""
    t1 = series[["match_id", "datetime", "team1", "winner"]].rename(columns={"team1": "team"})
    t1["win"] = (t1["winner"] == t1["team"]).astype(float)
    t2 = series[["match_id", "datetime", "team2", "winner"]].rename(columns={"team2": "team"})
    t2["win"] = (t2["winner"] == t2["team"]).astype(float)
    long = pd.concat([t1, t2], ignore_index=True)
    return long.sort_values(["team", "datetime", "match_id"], kind="mergesort").reset_index(
        drop=True
    )


def _per_team_pre_features(long: pd.DataFrame) -> pd.DataFrame:
    """form5 = mean of last <=5 wins BEFORE the current series (rolling WITHIN
    the team's own history, shifted by one); rest_days = days since the team's
    PREVIOUS series. Both never include the current row's own outcome."""
    long = long.copy()
    long["form5"] = long.groupby("team")["win"].transform(
        lambda s: s.rolling(5, min_periods=1).mean().shift(1)
    )
    rest = long.groupby("team")["datetime"].diff().dt.total_seconds() / 86_400.0
    long = long.assign(rest_days=rest)
    return long[["match_id", "team", "form5", "rest_days"]]


def main() -> None:
    series = pd.read_csv(REPO / "outputs" / "series_clean.csv")
    dt = pd.to_datetime(series["datetime"], utc=True, format="ISO8601")
    series = (
        series.assign(datetime=dt).sort_values("datetime", kind="mergesort").reset_index(drop=True)
    )

    print("[features] replaying Elo (pre-match ratings only)...")
    bt = run_elo_backtest(series, k=32.0)

    print("[features] computing per-team rolling features (vectorized)...")
    long = _long_format(series)
    pre = _per_team_pre_features(long)

    t1_map = pre.rename(columns={"team": "team1", "form5": "form5_t1", "rest_days": "rest_t1"})
    t2_map = pre.rename(columns={"team": "team2", "form5": "form5_t2", "rest_days": "rest_t2"})
    feat = (
        bt[["match_id", "team1", "team2"]]
        .merge(t1_map, on=["match_id", "team1"], how="left")
        .merge(t2_map, on=["match_id", "team2"], how="left")
    )
    assert feat["match_id"].is_unique
    # restore bt row order explicitly (merges can reorder): positional mapping by match_id
    feat = feat.set_index("match_id").reindex(bt["match_id"]).reset_index()

    # h2h per unordered pair: meetings-before + shifted cumulative wins, oriented
    pairs = pd.DataFrame(
        {
            "match_id": bt["match_id"].to_numpy(),
            "team1": bt["team1"].to_numpy(),
            "team2": bt["team2"].to_numpy(),
            "datetime": bt["datetime"].to_numpy(),
            "t1_win": bt["result"].to_numpy(),
        }
    )
    pairs["_a"] = pairs[["team1", "team2"]].min(axis=1)
    pairs["_b"] = pairs[["team1", "team2"]].max(axis=1)
    pairs = pairs.sort_values(["_a", "_b", "datetime"], kind="mergesort")
    g = pairs.groupby(["_a", "_b"], sort=False)
    pairs["meet_n"] = g.cumcount()  # meetings BEFORE this one (0 = first)
    a_is_t1 = (pairs["team1"] == pairs["_a"]).to_numpy()
    pairs["_a_win"] = np.where(a_is_t1, pairs["t1_win"], 1.0 - pairs["t1_win"])
    # cumulative a-wins shifted by one MEETING within each pair
    pairs["a_wins_before"] = (
        g["_a_win"].apply(lambda s: s.cumsum().shift(1)).reset_index(level=[0, 1], drop=True)
    )
    pairs["a_wins_before"] = pairs["a_wins_before"].fillna(0.0)
    pairs["h2h_share_a"] = np.where(
        pairs["meet_n"] > 0, pairs["a_wins_before"] / pairs["meet_n"], 0.5
    )
    pairs["h2h_t1"] = np.where(a_is_t1, pairs["h2h_share_a"], 1.0 - pairs["h2h_share_a"])
    # re-join in bt order (match_id unique per row) — never rely on positional alignment
    h2h_by_match = pairs[["match_id", "h2h_t1"]].rename(columns={"h2h_t1": "h2h_t1_win_share"})
    assert h2h_by_match["match_id"].is_unique
    feat = feat.merge(h2h_by_match, on="match_id", how="left")

    features = pd.DataFrame(
        {
            "match_id": bt["match_id"].to_numpy(),
            "datetime": bt["datetime"].to_numpy(),
            "tier": bt["tier"].to_numpy(),
            "is_bo1": (bt["games_played"] == 1).to_numpy(),
            "elo_diff": (bt["elo_t1_pre"] - bt["elo_t2_pre"]).to_numpy(),
            "form5_diff": (feat["form5_t1"].fillna(0.5) - feat["form5_t2"].fillna(0.5)).to_numpy(),
            "rest_days_diff": (
                feat["rest_t1"].fillna(7.0) - feat["rest_t2"].fillna(7.0)
            ).to_numpy(),
            "h2h_t1_win_share": feat["h2h_t1_win_share"].to_numpy(),
            "result": bt["result"].to_numpy(),
        }
    )

    # --- sample verification against the scalar reference implementations ---
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(features), size=150, replace=False)
    for i in sample_idx:
        i = int(i)
        row = bt.iloc[i]
        ref = row["datetime"]
        for team, col in ((row["team1"], "form5_t1"), (row["team2"], "form5_t2")):
            expected = rolling_form(series, team, n=5, ref_ts=ref)
            got = feat.at[i, col]
            got = 0.5 if pd.isna(got) else float(got)
            assert abs(got - expected) < 1e-9, f"form mismatch row {i} {team}: {got} vs {expected}"
        aw, bw = h2h_record(series, row["team1"], row["team2"], before_ts=ref)
        expected_h2h = aw / (aw + bw) if (aw + bw) > 0 else 0.5
        assert abs(features.at[i, "h2h_t1_win_share"] - expected_h2h) < 1e-9, (
            f"h2h mismatch row {i}: {features.at[i, 'h2h_t1_win_share']} vs {expected_h2h}"
        )
    print(f"[features] sample verification passed ({len(sample_idx)} rows: form5 + h2h)")

    dest = REPO / "outputs" / "features_v1.parquet"
    features.to_parquet(dest, index=False)
    print(f"[features] wrote {dest.relative_to(REPO)}: {len(features)} rows")
    print(
        features[["elo_diff", "form5_diff", "rest_days_diff", "h2h_t1_win_share"]]
        .describe()
        .loc[["mean", "std", "min", "max"]]
        .to_string()
    )


if __name__ == "__main__":
    main()
