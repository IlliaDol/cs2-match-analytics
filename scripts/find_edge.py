"""Bet or pass: compare our probabilities against a real bookmaker price.

The CS2 model has been calibrated and validated (README: logloss 0.6342 vs 0.6931 baseline,
ECE 0.0159 on 2,943 held-out series). The one question it could never answer was *"is that
worth betting on?"* — because the v1 odds subset is too small to chart, and because nobody had
wired prices into the pipeline. This script does the arithmetic; it does not pretend to have
data it lacks.

What it needs: a CSV of **decimal odds** for a two-way market, one row per series:

    date,team1,team2,odds1,odds2,bookmaker
    2026-04-29,Natus Vincere,G2 Esports,1.72,2.10,pinnacle

What it does, per row:

1. **de-vig** the pair: implied probabilities are 1/odds, which sum to more than 1 by the
   bookmaker's margin. Normalising them removes the margin and gives the market's honest
   probability (this is the standard "proportional de-vig"; it is stated, not hidden).
2. take **our** probability from the shipped Elo ratings (`artifacts/elo_ratings.json`,
   the M4 engine — the calibrated logistic model needs the full feature pipeline, so Elo is
   what a standalone script can honestly use);
3. compute **EV** per 1 unit staked: `p * (odds - 1) - (1 - p)`,
4. compute the **break-even hit rate** `1/odds` — the number the model must beat to be worth it,
5. flag positive-EV sides and print the edge, so "close" is visibly distinguished from "value".

Honest limits, printed with every run: no real odds file ships with this repo (the v1 subset
is too small — see README Limitations), so the committed example is clearly marked SYNTHETIC.
Elo ignores form/rest/roster and is *not* the calibrated model; it is the honest offline proxy.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ELO_PATH = REPO / "artifacts" / "elo_ratings.json"
OUT_DIR = REPO / "outputs"

TEAM_HINTS = ("team1", "team2", "home", "away", "team_a", "team_b")
ODDS_HINTS = ("odds1", "odds2", "odds_a", "odds_b", "home_odds", "away_odds")


# --------------------------------------------------------------------------- inputs


def load_ratings(path: Path = ELO_PATH) -> dict[str, float]:
    """Team -> Elo rating from the shipped artifact (tolerant of the obvious shapes)."""
    import json

    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob, dict) and "ratings" in blob:
        blob = blob["ratings"]
    if isinstance(blob, dict):
        return {str(k): float(v) for k, v in blob.items()}
    if isinstance(blob, list):                      # [{"team": ..., "elo": ...}, ...]
        out = {}
        for row in blob:
            name = row.get("team") or row.get("name")
            rating = row.get("elo") or row.get("rating")
            if name is not None and rating is not None:
                out[str(name)] = float(rating)
        return out
    raise ValueError(f"unrecognised ratings file: {path}")


def parse_odds_csv(path: Path) -> list[dict]:
    """Read the odds file, detecting the team and odds columns by name hints."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        lowered = {str(c).casefold(): c for c in reader.fieldnames}
        teams = [lowered[c] for c in lowered if any(h in c for h in TEAM_HINTS)]
        odds = [lowered[c] for c in lowered if any(h in c for h in ODDS_HINTS)]
        if len(teams) < 2 or len(odds) < 2:
            raise ValueError(
                f"{path}: need two team columns and two odds columns, found {teams} / {odds}"
            )
        date_col = next((lowered[c] for c in lowered if "date" in c), None)
        rows = []
        for raw in reader:
            try:
                o1, o2 = float(raw[odds[0]]), float(raw[odds[1]])
            except (TypeError, ValueError):
                continue                            # a row without prices is not a market
            rows.append({
                "date": (raw.get(date_col) or "").strip() if date_col else "",
                "team1": str(raw[teams[0]]).strip(),
                "team2": str(raw[teams[1]]).strip(),
                "odds1": o1,
                "odds2": o2,
            })
    if not rows:
        raise ValueError(f"{path}: no rows with usable decimal odds")
    return rows


def elo_probability(rating_a: float, rating_b: float) -> float:
    """P(A beats B) from the M4 Elo curve — the same formula the repo's engine uses."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def devig(odds1: float, odds2: float) -> tuple[float, float, float]:
    """Proportional de-vig: return (p1, p2, margin).

    Decimal odds below 1.0 are impossible in a real book, so they are rejected rather than
    silently producing a negative probability.
    """
    if odds1 <= 1.0 or odds2 <= 1.0:
        raise ValueError(f"decimal odds must exceed 1.0 (got {odds1}, {odds2})")
    raw1, raw2 = 1.0 / odds1, 1.0 / odds2
    total = raw1 + raw2
    return raw1 / total, raw2 / total, total - 1.0


def evaluate(p: float, odds: float) -> dict:
    """EV per unit staked and the break-even hit rate for one side."""
    return {
        "probability": p,
        "odds": odds,
        "implied": 1.0 / odds,
        "edge": p - 1.0 / odds,
        "ev_per_unit": p * (odds - 1.0) - (1.0 - p),
        "break_even_rate": 1.0 / odds,
    }


# --------------------------------------------------------------------------- main


def analyse(rows: list[dict], ratings: dict[str, float], min_edge: float = 0.0) -> list[dict]:
    by_key = {str(k).casefold(): v for k, v in ratings.items()}
    out = []
    for row in rows:
        r1 = by_key.get(row["team1"].casefold())
        r2 = by_key.get(row["team2"].casefold())
        missing = [t for t, r in ((row["team1"], r1), (row["team2"], r2)) if r is None]
        try:
            market1, market2, margin = devig(row["odds1"], row["odds2"])
        except ValueError as err:
            out.append({**row, "skipped": str(err)})
            continue
        if missing:
            out.append({**row, "skipped": f"no rating for {', '.join(missing)}"})
            continue

        p1 = elo_probability(r1, r2)
        side1, side2 = evaluate(p1, row["odds1"]), evaluate(1.0 - p1, row["odds2"])
        best = max((side1, side2), key=lambda s: s["ev_per_unit"])
        out.append({
            **row,
            "margin": margin,
            "market_p1": market1,
            "market_p2": market2,
            "elo_p1": p1,
            "best_side": row["team1"] if best is side1 else row["team2"],
            "best_ev": best["ev_per_unit"],
            "best_edge": best["edge"],
            "best_odds": best["odds"],
            "best_break_even": best["break_even_rate"],
            "bet": best["edge"] > min_edge,
        })
    return out


def render(results: list[dict], min_edge: float) -> str:
    lines = [
        "Bet or pass — model probability vs bookmaker price (decimal odds)",
        f"flagging sides with edge > {min_edge:.1%}; EV is per 1.0 staked",
        "",
    ]
    flagged = [r for r in results if r.get("bet")]
    skipped = [r for r in results if "skipped" in r]
    for row in results:
        if "skipped" in row:
            continue
        mark = "VALUE" if row["bet"] else "  -- "
        lines.append(
            f"{mark} {row['date'] or '        '} {row['team1'][:18]:<18} vs "
            f"{row['team2'][:18]:<18} "
            f"| market p1 {row['market_p1']:.1%} (margin {row['margin']:.1%}) "
            f"| elo p1 {row['elo_p1']:.1%} "
            f"| best {row['best_side'][:18]:<18} @ {row['best_odds']:.2f} "
            f"edge {row['best_edge']:+.1%} EV {row['best_ev']:+.3f}"
        )
    lines += [
        "",
        f"{len(flagged)} of {len(results) - len(skipped)} priced series have positive EV "
        f"above {min_edge:.1%}",
    ]
    if skipped:
        lines.append(f"skipped {len(skipped)}: " + "; ".join(
            f"{r['team1']} vs {r['team2']} ({r['skipped']})" for r in skipped[:4]
        ))
    lines += [
        "",
        "Caveats that matter: this uses the Elo engine, NOT the calibrated logistic model "
        "(that needs the full feature pipeline).",
        "And no real odds file ships here — the v1 odds subset is too small (README "
        "Limitations), so any committed example is synthetic.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("odds_csv", type=Path, help="decimal odds, one row per series")
    parser.add_argument("--min-edge", type=float, default=0.0,
                        help="require at least this probability edge to call it a value bet")
    parser.add_argument("--out", type=Path, default=None, help="write the table as CSV")
    args = parser.parse_args(argv)

    ratings = load_ratings()
    rows = parse_odds_csv(args.odds_csv)
    results = analyse(rows, ratings, min_edge=args.min_edge)
    print(render(results, args.min_edge))

    if args.out:
        import pandas as pd

        frame = pd.DataFrame(results)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        shown = args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out
        print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
