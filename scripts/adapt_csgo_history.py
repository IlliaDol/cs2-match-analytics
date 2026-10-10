"""Add the 2015-2020 CS:GO era to the dataset (the years our data was missing).

Source: Kaggle `mateusdmachado/csgo-professional-matches`, already downloaded into
`data/raw/new/`. Two files matter:

  players.csv  383,317 rows, one per player per match, with **per-map** columns
               (m1_*, m2_*, m3_*) holding kills, assists, deaths, ADR, KAST, kddiff and
               the HLTV rating, plus the match date and the event name.
  results.csv   45,773 rows, one per map: the two teams, both scores, who won the map,
               the map name and both teams' world ranks.

This is the only one of our three sources that covers 2015-2020, and it is the richest:
it has real dates (no guessing) *and* assists, which the 2021-2023 file lacks.

The one trap is the join. `results.csv` is keyed by `_map`, while a player's per-map
stats live in `m1_*` / `m2_*` / `m3_*`, which correspond to `map_1` / `map_2` / `map_3` on
the players row — *not* necessarily to the order of the result rows. So the map index is
resolved by name, and a test pins that down.

    .venv/Scripts/python.exe scripts/adapt_csgo_history.py         # -> data/interim/
    .venv/Scripts/python.exe scripts/adapt_csgo_history.py --dry   # stats only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "raw" / "new"
OUT = REPO / "data" / "interim" / "csgo_history_games.csv"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# reuse the country-name -> code table (this source stores full country names)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt_csgo_dataset import CODE_BY_COUNTRY  # noqa: E402

STATS = ("kills", "deaths", "assists", "adr", "kast")
PER_MAP_COLUMNS = ["date", "player_name", "team", "opponent", "country", "match_id",
                   "event_name", "best_of", "map_1", "map_2", "map_3"]
for _i in (1, 2, 3):
    PER_MAP_COLUMNS += [f"m{_i}_{stat}" for stat in ("kills", "assists", "deaths", "adr", "kast")]


def player_label(name: str, country: str) -> str:
    """'s1mple, UA' — this source carries no real names, so the label is nickname + code."""
    code = CODE_BY_COUNTRY.get((country or "").strip().casefold(), "")
    return f"{str(name).strip()}, {code}" if code else str(name).strip()


def map_index(maps: pd.Series, wanted: str) -> int | None:
    """Which m*-block belongs to `wanted` ('map_2' -> 2). Read by name, never by position."""
    for position in (1, 2, 3):
        if str(maps.get(f"map_{position}", "")).strip() == wanted:
            return position
    return None


def build(players: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    players = players.copy()
    players["label"] = [player_label(n, c) for n, c in zip(players["player_name"],
                                                           players["country"], strict=False)]
    rows: list[dict] = []

    by_match = {mid: grp for mid, grp in players.groupby("match_id", sort=False)}
    for result in results.itertuples(index=False):
        match = by_match.get(result.match_id)
        if match is None:
            continue
        position = map_index(match.iloc[0], str(result.map_name))
        if position is None:
            continue                                   # that map is not in the player rows
        map_name = str(result.map_name)
        team1, team2 = str(result.team_1), str(result.team_2)
        sides: dict[str, list] = {team1: [], team2: []}
        for player in match.itertuples(index=False):
            side = str(player.team)
            if side not in sides:
                continue
            sides[side].append(player)
        if min(len(sides[team1]), len(sides[team2])) < 5:
            continue                                   # need two complete line-ups

        row = {
            "match_id": result.match_id, "game_id": f"{result.match_id}-{map_name}",
            "tournament": str(getattr(match.iloc[0], "event_name", "")),
            "team1_id": team1, "team2_id": team2, "team1": team1, "team2": team2,
            "score1_game": int(result.result_1), "score2_game": int(result.result_2),
            "rounds": int(result.result_1) + int(result.result_2),
            "map_id": map_name, "map_name": map_name,
            "datetime": str(getattr(match.iloc[0], "date", "")),
            "team1_win": int(result.map_winner) == 1,
            "is_total": False, "game": "csgo",
            "bestOf": int(getattr(match.iloc[0], "best_of", 0) or 0),
        }
        for side, team in (("team1", team1), ("team2", team2)):
            for slot, player in enumerate(sides[team][:5], start=1):
                prefix = f"{side}_player{slot}"
                row[prefix] = player.label
                for stat in STATS:
                    row[f"{prefix}_{stat}"] = getattr(player, f"m{position}_{stat}", np.nan)
                row[f"{prefix}_kddiff"] = (
                    getattr(player, f"m{position}_kills", np.nan)
                    - getattr(player, f"m{position}_deaths", np.nan)
                )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    wins = frame.groupby(["match_id", "team1", "team2"])["team1_win"].agg(["sum", "size"])
    frame["games_played"] = frame["match_id"].map(wins["size"])
    frame["score1_match"] = frame["match_id"].map(wins["sum"])
    frame["score2_match"] = frame["games_played"] - frame["score1_match"]
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--dry", action="store_true", help="print the stats, write nothing")
    args = parser.parse_args(argv)

    source = Path(args.src)
    players_path = source / "players.csv"
    if not players_path.exists():
        print(f"no {players_path} — run: .venv/Scripts/python.exe -m kaggle datasets download "
              "-d mateusdmachado/csgo-professional-matches -p data/raw/new --unzip")
        return 1
    players = pd.read_csv(players_path, usecols=PER_MAP_COLUMNS, low_memory=False)
    results = pd.read_csv(source / "results.csv", low_memory=False)
    # `_map` is not a valid namedtuple field, so itertuples would mangle it — rename on read
    results = results.rename(columns={"_map": "map_name"})
    frame = build(players, results)
    if frame.empty:
        print("nothing built — check that players.csv and results.csv are the expected shape")
        return 1

    print(f"maps built        : {len(frame):,}")
    print(f"window            : {frame['datetime'].min()} -> {frame['datetime'].max()}")
    print(f"events            : {frame['tournament'].nunique():,}")
    print(f"matches           : {frame['match_id'].nunique():,}")
    print(f"maps with full 10 : {frame['team1_player5'].notna().sum():,}")
    coverage = {s: f"{frame[f'team1_player1_{s}'].isna().mean():.0%}"
                for s in ("kills", "deaths", "assists", "adr", "kast")}
    print(f"null share (P1)   : {coverage}")
    print(f"KAST sanity       : min {frame['team1_player1_kast'].min():.1f} "
          f"max {frame['team1_player1_kast'].max():.1f} (percent, like the CS2 file)")

    if args.dry:
        print("\n--dry: nothing written")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
