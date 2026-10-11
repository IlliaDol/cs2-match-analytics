"""Does dated roster data actually improve the model? A direct A/B, no new production code.

The two strongest features in the model — `roster_stability_diff` and `standin_diff` —
compare the current line-up only against the team's *previous* match. They cannot tell 3 days
from 3 months, and they cannot see that a player joined yesterday. Liquipedia's transfer
records (scripts/fetch_liquipedia_transfers.py, 59k dated moves) can.

This script answers one question and nothing else: **on the same time-split (train <
2026-01-01, test >= 2026-01-01), do date-derived roster features beat the current feature
set?** It trains two logistic regressions on identical rows and prints acc / logloss / ECE
side by side, so the answer is a number rather than an opinion.

New features, all strictly pre-match (each uses only transfers dated on or before the match):
  tenure_min_diff  — days since the newest member of each line-up joined, team1 - team2
  tenure_avg_diff  — mean tenure across the line-up, team1 - team2
  recent_join_diff — how many of the five joined within 30 days, team1 - team2

Coverage is partial by construction: only ~54% of team names in the match data appear in
Liquipedia's transfer pages (the missing ones are mostly low-tier/amateur sides). Unknown
line-ups get 0 — meaning "no information", which is how the existing features degrade too —
and the script prints the real coverage so the number can be read honestly.

    .venv/Scripts/python.exe scripts/roster_dates_experiment.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "cs2_all_tiers_games.csv"
TRANSFERS = REPO / "data" / "interim" / "roster_transfers.csv"
FEATURES = REPO / "outputs" / "features_v1.parquet"
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
SLOTS = [f"team{s}_player{i}" for s in (1, 2) for i in range(1, 6)]
NAME_RE = re.compile(r"^(?P<nick>.*?)\s*\((?P<real>.*?)\),\s*(?P<cc>[A-Z]{2})$")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def player_key(label: str) -> tuple[str, str] | None:
    """'Donk (Danil Kryshkovets), RU' -> ('donk', 'ru'); None when unparseable."""
    found = NAME_RE.match(str(label).strip())
    if not found:
        return None
    return found.group("nick").strip().casefold(), found.group("cc").lower()


def team_key(name: str) -> str:
    return str(name).strip().casefold()


def load_joins(path: Path = TRANSFERS) -> dict[tuple[str, str], list[pd.Timestamp]]:
    """(team, player) -> sorted dates on which that player joined that team."""
    joins: dict[tuple[str, str], list[pd.Timestamp]] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            team, player, flag, date = (row.get("to_team", ""), row.get("player", ""),
                                        row.get("flag", ""), row.get("date", ""))
            if not (team and player and date):
                continue
            stamp = pd.to_datetime(date, errors="coerce", utc=True)
            if pd.isna(stamp):
                continue
            joins.setdefault((team_key(team), (player.strip().casefold(), flag.strip().lower())),
                             []).append(stamp)
    for dates in joins.values():
        dates.sort()
    return joins


def series_lineups() -> pd.DataFrame:
    """One row per series: datetime, both teams, both line-ups.

    Prefers a map-level row (a real per-map line-up) and falls back to the series summary
    row: roughly half the series in the raw file have only the summary, and dropping them
    silently cut the sample in half on the first run (5,173 of 9,923 series).
    """
    frame = pd.read_csv(RAW, low_memory=False,
                        usecols=["match_id", "datetime", "team1", "team2", "team1_win",
                                 "is_total", *SLOTS])
    frame = frame[frame["match_id"].notna()].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    # map rows (is_total False) sort first, so groupby().first() keeps a real map row
    frame = frame.sort_values(["match_id", "is_total", "datetime"])
    out = frame.groupby("match_id", as_index=False).first()
    out["result"] = out["team1_win"].astype(bool).astype(float)
    return out


def date_features(matches: pd.DataFrame,
                  joins: dict[tuple[str, str], list[pd.Timestamp]]) -> pd.DataFrame:
    """Per-series date-derived roster features (team1 - team2)."""
    rows: list[dict] = []
    for tup in matches.itertuples(index=False):
        when = tup.datetime
        rec: dict = {"match_id": tup.match_id}
        per_side: dict[str, tuple[float, float, float] | None] = {}
        for side in ("team1", "team2"):
            team = team_key(getattr(tup, side))
            tenures: list[int] = []
            for i in range(1, 6):
                key = player_key(getattr(tup, f"{side}_player{i}"))
                if key is None:
                    continue
                history = joins.get((team, key))
                if not history:
                    continue
                before = [d for d in history if d <= when]
                if before:
                    tenures.append((when - max(before)).days)
            per_side[side] = ((min(tenures), float(np.mean(tenures)),
                               float(sum(t <= 30 for t in tenures))) if tenures else None)
        for name, index in (("tenure_min_diff", 0), ("tenure_avg_diff", 1),
                            ("recent_join_diff", 2)):
            a, b = per_side["team1"], per_side["team2"]
            rec[name] = np.nan if (a is None or b is None) else a[index] - b[index]
            rec[f"{name}_known"] = float(a is not None and b is not None)
        rec["lineup_known"] = float(sum(
            1 for side in ("team1", "team2") for i in range(1, 6)
            if player_key(getattr(tup, f"{side}_player{i}")) is not None)) / 10.0
        rows.append(rec)
    return pd.DataFrame(rows)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.sum():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def fit_report(X: np.ndarray, y: np.ndarray, split: pd.Series, label: str) -> dict:
    tr, te = (split == "train").to_numpy(), (split == "test").to_numpy()
    model = LogisticRegression(max_iter=2000, random_state=0)
    model.fit(X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]
    acc = float(((p >= 0.5).astype(float) == y[te]).mean())
    loss = float(-np.mean(y[te] * np.log(np.clip(p, 1e-12, 1)) +
                          (1 - y[te]) * np.log(np.clip(1 - p, 1e-12, 1))))
    out = {"label": label, "n_features": X.shape[1], "acc": acc, "logloss": loss,
           "ece": ece(y[te], p), "n_train": int(tr.sum()), "n_test": int(te.sum())}
    print(f"  {label:<28} feats={out['n_features']:>2}  acc={acc:.4f}  "
          f"logloss={loss:.4f}  ece={out['ece']:.4f}")
    return out


def main() -> int:
    for path in (RAW, TRANSFERS, FEATURES):
        if not path.exists():
            print(f"missing {path} — run the fetchers/adapters first")
            return 1

    joins = load_joins()
    matches = series_lineups()
    dated = date_features(matches, joins)
    print(f"series: {len(matches):,} | (team, player) join records: {len(joins):,}")
    covered = float((dated["tenure_min_diff"].notna()).mean())
    print(f"series with BOTH line-ups dated: {covered:.1%} "
          f"(the rest get 0 = 'no information')")

    store = pd.read_parquet(FEATURES)
    store["datetime"] = pd.to_datetime(store["datetime"], utc=True)
    merged = store.merge(dated, on="match_id", how="inner")
    print(f"feature store rows: {len(store):,} | merged with dated rosters: {len(merged):,}")

    tier = pd.get_dummies(merged["tier"], prefix="tier").astype(float)
    baseline_cols = ["elo_diff", "form5_diff", "rest_days_diff", "is_bo1",
                     "h2h_t1_win_share", "roster_stability_diff", "standin_diff"]
    base = pd.concat([merged[baseline_cols].astype(float), tier], axis=1)
    new_cols = ["tenure_min_diff", "tenure_avg_diff", "recent_join_diff"]
    enhanced = pd.concat([base, merged[new_cols].fillna(0.0).astype(float)], axis=1)

    if base.isna().any().any():
        print("baseline has NaN — the store itself is not NaN-free, stopping")
        return 1

    y = merged["result"].to_numpy(dtype=float)
    print(f"\ntime split: train < {CUTOFF.date()}, test >= it")
    print(f"\n  {'what':<28} {'features':>8}  {'acc':>7}  {'logloss':>8}  {'ece':>7}")
    before = fit_report(base.to_numpy(dtype=float), y, merged["split_probe"]
                        if "split_probe" in merged else
                        pd.Series(np.where(merged["datetime"] < CUTOFF, "train", "test")),
                        "current 9 features")
    after = fit_report(enhanced.to_numpy(dtype=float), y, merged["split_probe"]
                       if "split_probe" in merged else
                       pd.Series(np.where(merged["datetime"] < CUTOFF, "train", "test")),
                       "+ dated roster (3)")
    print(f"\n  delta acc {after['acc'] - before['acc']:+.4f} | "
          f"logloss {after['logloss'] - before['logloss']:+.4f} (negative is better) | "
          f"ece {after['ece'] - before['ece']:+.4f}")
    print("\n  Reference: the shipped model measures acc 0.6306 / logloss 0.6342 / ece 0.0159")
    return 0


if __name__ == "__main__":
    sys.exit(main())
