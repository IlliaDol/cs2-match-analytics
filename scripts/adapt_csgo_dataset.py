"""Turn the Kaggle CS:GO dataset into our map-row schema, so the two eras can be compared.

Source: `fernandopy/csgo-data-set` (HLTV-scraped, CS:GO era). Four files matter:
  matchInfos.csv   -> one row per player per map: kills, deaths, ADR, KAST, rating
  matchOverview.csv-> one row per map: teams, final score, winner
  players.csv      -> PlayerID -> nickname, real name, country
  matchIds.csv     -> MatchID -> a slug naming the event ("<id>/<a>-vs-<b>-<event>")
Together they reconstruct exactly what the CS2 file gives us: one row per map with ten
players and their kills / deaths / ADR / KAST.

Three honest differences, all handled explicitly rather than papered over:

* **No dates.** Nothing in the dataset carries a timestamp. The event slug is the only
  hint, so the date is derived from it where it has a season or year ("fall-2023" ->
  2023-10-15) and left empty otherwise. It is an *event* date, not the match date.
* **No assists.** Filled with NaN (unknown), never 0 — a zero would read as a real stat.
* **KAST is a fraction** (0.52) where our CS2 data stores a percentage (52.0).

The game label follows the transition (Illia's call, 2026-09-15): a slug naming a season
from autumn 2023 onward or a year >= 2024 is labelled `cs2`, everything else `csgo`. The
slug is a heuristic, so the label is best-effort and a few boundary matches will be wrong.

    .venv/Scripts/python.exe scripts/adapt_csgo_dataset.py         # -> data/interim/
    .venv/Scripts/python.exe scripts/adapt_csgo_dataset.py --dry   # stats only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "raw" / "new" / "data"
OUT = REPO / "data" / "interim" / "csgo_adapter_games.csv"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

STATS = ("kills", "deaths", "assists", "adr", "kast")
SEASON_DAY = {"winter": "01-15", "spring": "04-15", "summer": "07-15",
              "fall": "10-15", "autumn": "10-15"}
YEAR_RE = re.compile(r"(?<!\d)(20[0-2]\d)(?!\d)")
SEASON_RE = re.compile(r"\b(winter|spring|summer|fall|autumn)\b", re.I)

# country names -> the ISO codes our CS2 rows use, so a CS:GO player prints like a CS2 one
CODE_BY_COUNTRY = {
    "argentina": "AR", "australia": "AU", "austria": "AT", "belgium": "BE", "brazil": "BR",
    "bulgaria": "BG", "canada": "CA", "china": "CN", "czech republic": "CZ", "czechia": "CZ",
    "denmark": "DK", "egypt": "EG", "estonia": "EE", "finland": "FI", "france": "FR",
    "germany": "DE", "hungary": "HU", "india": "IN", "indonesia": "ID", "israel": "IL",
    "italy": "IT", "japan": "JP", "kazakhstan": "KZ", "latvia": "LV", "lithuania": "LT",
    "malaysia": "MY", "mongolia": "MN", "netherlands": "NL", "new zealand": "NZ",
    "norway": "NO", "poland": "PL", "portugal": "PT", "romania": "RO", "russia": "RU",
    "serbia": "RS", "singapore": "SG", "slovakia": "SK", "south africa": "ZA",
    "south korea": "KR", "spain": "ES", "sweden": "SE", "switzerland": "CH", "taiwan": "TW",
    "thailand": "TH", "turkey": "TR", "ukraine": "UA", "united kingdom": "GB",
    "united states": "US", "uruguay": "UY", "uzbekistan": "UZ", "vietnam": "VN",
    "belarus": "BY", "bosnia and herzegovina": "BA", "chile": "CL", "colombia": "CO",
    "croatia": "HR", "greece": "GR", "hong kong": "HK", "ireland": "IE", "jordan": "JO",
    "lebanon": "LB", "mexico": "MX", "peru": "PE", "philippines": "PH", "tunisia": "TN",
}


def slug_year(slug: str) -> int | None:
    """The year named in an event slug, if any ('...-fall-2023-...' -> 2023)."""
    found = YEAR_RE.search(slug or "")
    return int(found.group(1)) if found else None


def slug_game(slug: str) -> str:
    """cs2 for the CS2 era, csgo otherwise — a heuristic read of the event slug."""
    year = slug_year(slug)
    if year and year >= 2024:
        return "cs2"
    season = SEASON_RE.search(slug or "")
    if year == 2023 and season and season.group(1).casefold() in ("fall", "autumn", "winter"):
        return "cs2"                 # CS2 shipped 2023-09-27; autumn/winter 2023 is CS2
    return "csgo"


def slug_date(slug: str) -> str:
    """An approximate ISO date from the event slug, or '' when it names no year.

    Mid-season is used on purpose: the slug tells us the season, never the day, and a
    plausible midpoint is more useful than pretending to a precision we do not have.
    """
    year = slug_year(slug)
    if not year:
        return ""
    season = SEASON_RE.search(slug or "")
    day = SEASON_DAY.get(season.group(1).casefold(), "07-01") if season else "07-01"
    return f"{year}-{day}"


def player_label(nick: str, real: str, country: str) -> str:
    """Same shape as the CS2 rows: 'Donk (Danil Kryshkovets), RU'.

    The source packs the nickname into the name field ('Oleksandr 's1mple' Kostyliev'),
    which would make the same person a different player in each era — strip it so the two
    sources read alike.
    """
    nick = (nick or "").strip()
    real = re.sub(r"\s*'[^']*'\s*", " ", (real or "")).strip()
    real = re.sub(r"\s{2,}", " ", real)
    code = CODE_BY_COUNTRY.get((country or "").strip().casefold(), "")
    name = f"{nick} ({real})" if real else nick
    return f"{name}, {code}" if code else name


def load_source(src: Path = SRC) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(src / "matchOverview.csv"), pd.read_csv(src / "matchInfos.csv"),
            pd.read_csv(src / "players.csv"), pd.read_csv(src / "matchIds.csv"))


def build(overview: pd.DataFrame, infos: pd.DataFrame, players: pd.DataFrame,
          match_ids: pd.DataFrame) -> pd.DataFrame:
    """One row per map, in our schema, with a `game` column."""
    nick = players.set_index(players["ID"].astype("Int64"))["NickName"].to_dict()
    real = players.set_index(players["ID"].astype("Int64"))["Name"].to_dict()
    country = players.set_index(players["ID"].astype("Int64"))["Country"].to_dict()
    slug = {int(r.ID): str(r.Tittle) for r in match_ids.itertuples()}

    infos = infos.copy()
    infos["label"] = [player_label(nick.get(p), real.get(p), country.get(p))
                      for p in infos["PlayerID"].astype("Int64")]
    infos["kast_pct"] = infos["Total_KAST"].astype(float) * 100.0     # fraction -> percent

    rows: list[dict] = []
    for (match_id, map_name), group in overview.groupby(["MatchID", "MapName"], sort=False):
        stats = infos[(infos["MatchID"] == match_id) & (infos["MapName"] == map_name)]
        if len(stats) < 10:                                          # need two full line-ups
            continue
        team1_id, team2_id = int(group["Team1"].iloc[0]), int(group["Team2"].iloc[0])
        event = slug.get(int(match_id), "")
        row = {
            "match_id": match_id, "game_id": f"{match_id}-{map_name}",
            "tournament": event.split("/", 1)[-1] if "/" in event else event,
            "team1_id": team1_id, "team2_id": team2_id,
            "score1_game": int(group["Team1_Final_Score"].iloc[0]),
            "score2_game": int(group["Team2_Final_Score"].iloc[0]),
            "map_id": map_name, "map_name": map_name,
            "datetime": slug_date(event), "team1_win": bool(group["Team1_Win"].iloc[0]),
            "is_total": False, "game": slug_game(event),
        }
        for side, team_id in (("team1", team1_id), ("team2", team2_id)):
            side_stats = stats[stats["TeamID"] == team_id].sort_values(
                "Total_Kills", ascending=False)
            for slot in range(1, 6):
                if slot > len(side_stats):
                    break
                player = side_stats.iloc[slot - 1]
                prefix = f"{side}_player{slot}"
                row[f"{prefix}_id"] = int(player["PlayerID"])
                row[prefix] = player["label"]
                for stat, value in (("kills", player["Total_Kills"]),
                                    ("deaths", player["Total_Deaths"]),
                                    ("assists", np.nan),          # not in this dataset
                                    ("adr", player["Total_ADR"]),
                                    ("kast", player["kast_pct"])):
                    row[f"{prefix}_{stat}"] = value
                row[f"{prefix}_kddiff"] = player["Total_Kills"] - player["Total_Deaths"]
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # match-level context, derived from the maps we kept
    wins = frame.groupby(["match_id", "team1_id", "team2_id"])["team1_win"].agg(["sum", "size"])
    frame["games_played"] = frame["match_id"].map(wins["size"])
    frame["bestOf"] = (frame["games_played"] // 2) * 2 + 1
    frame["score1_match"] = frame["match_id"].map(wins["sum"])
    frame["score2_match"] = frame["games_played"] - frame["score1_match"]
    frame["team1"] = frame["team1_id"]
    frame["team2"] = frame["team2_id"]
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--dry", action="store_true", help="print the stats, write nothing")
    args = parser.parse_args(argv)

    overview, infos, players, match_ids = load_source(Path(args.src))
    frame = build(overview, infos, players, match_ids)
    if frame.empty:
        print("nothing built — is data/raw/new/data/ present? run fetch beforehand")
        return 1

    print(f"maps built        : {len(frame)}")
    print("players per map   : 10 (rows with fewer were skipped)")
    print(f"game split        : {frame['game'].value_counts().to_dict()}")
    dated = frame["datetime"].astype(str).str.len() > 0
    print(f"dated rows        : {dated.sum()} ({dated.mean():.0%}) —"
          " the rest name no year in the slug")
    print(f"years present     : {sorted(frame.loc[dated, 'datetime'].str[:4].unique())}")
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
