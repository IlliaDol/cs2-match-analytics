"""Rebuild data/interim/matches_series.csv — the tier-labeled series table.

Pipeline position: data/raw/*.csv --load_matches--> this file --> r/01_wrangle.R
--> outputs/series_clean.csv --> every later module (M4 backtest, M5 SQL, M6, M7).

Tier labels come from the file each match was loaded from (tier1/tier2/tier3);
matches appearing in several tier files are deduped (keep first).
Run from repo root:  .venv/Scripts/python.exe scripts/build_interim.py
"""

from pathlib import Path

import pandas as pd

from cs2analytics.io.loader import load_matches

REPO = Path(__file__).resolve().parents[1]
TIERS = {
    "tier1": "cs2_tier1_games.csv",
    "tier2": "cs2_tier2_games.csv",
    "tier3": "cs2_tier3_games.csv",
}


def main() -> None:
    frames = []
    for tier, fname in TIERS.items():
        df = load_matches(REPO / "data" / "raw" / fname)
        df["tier"] = tier
        frames.append(df)
        print(f"[build_interim] {tier}: {len(df)} series rows")

    out = pd.concat(frames, ignore_index=True)
    n_before = len(out)
    out = out.drop_duplicates(subset="match_id", keep="first")
    n_dupes = n_before - len(out)
    out = out.sort_values("datetime", kind="mergesort").reset_index(drop=True)

    dest = REPO / "data" / "interim" / "matches_series.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    print(f"[build_interim] cross-tier duplicates dropped: {n_dupes}")
    print(f"[build_interim] wrote {len(out)} rows -> {dest.relative_to(REPO)}")
    share = (out["winner"] == out["team1"]).mean()
    print(
        f"[build_interim] sanity — winner==team1 share: {share:.3f} (must be ~0.5-0.6, NOT ~0.06)"
    )


if __name__ == "__main__":
    main()
