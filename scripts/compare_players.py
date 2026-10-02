"""Player-vs-player comparison (HLTV-style, but from our own rows).

Every map row in `data/raw/cs2_all_tiers_games.csv` carries all ten players:
name plus kills / deaths / assists / ADR / KAST. That is enough to build the duel
people actually look up on HLTV — but with things HLTV's static pages do not give
you per map:

* **KPR** (kills per round) from the map's real round count, not from an estimate;
* **head-to-head**: only the maps where the two players were on the server
  together, including whose team won those maps;
* the same split by **map** and by **event**, so you can see where a record is
  built (a strong overall K/D can hide a bad map).

What this deliberately does NOT do: invent an "HLTV Rating 2.x" number. That
formula is not public and a lookalike would be a made-up figure. We show the
ingredients — K/D, ADR, KAST, KPR — and label every rate with its sample size.

Run from the repo root:

    .venv/Scripts/python.exe scripts/compare_players.py --list Zyw
    .venv/Scripts/python.exe scripts/compare_players.py --a="ZywOo (Mathieu Herbaut), FR" \\
        --b="Donk (Danil Kryshkovets), RU" --from 2025-08-20 --to 2026-05-05
    .venv/Scripts/python.exe scripts/compare_players.py --a=A --b=B --md docs/PLAYER-COMPARISON.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "cs2_all_tiers_games.csv"
OUT_DIR = REPO / "outputs"

# Windows consoles default to cp1252, which cannot encode the non-ASCII characters that
# occur in real player names (e.g. "č") — the report would crash mid-print. Force UTF-8 on
# our own streams; a stream that cannot be reconfigured (pytest capture) is left alone.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SIDES = ("team1", "team2")
SLOTS = [f"{side}_player{i}" for side in SIDES for i in range(1, 6)]
STATS = ("kills", "deaths", "assists", "adr", "kast")


# --------------------------------------------------------------------------- load


CSGO_ADAPTER = REPO / "data" / "interim" / "csgo_adapter_games.csv"
CSGO_HISTORY = REPO / "data" / "interim" / "csgo_history_games.csv"


def _read_map_file(path: Path, game: str) -> pd.DataFrame:
    """One source file, normalised and tagged with its game."""
    frame = pd.read_csv(path, low_memory=False)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    frame = frame[~frame["is_total"].astype(bool) & frame["map_name"].notna()].copy()
    # a map's real round count: both sides' final scores
    frame["rounds"] = pd.to_numeric(frame["score1_game"], errors="coerce") + pd.to_numeric(
        frame["score2_game"], errors="coerce"
    )
    if "game" not in frame.columns:            # the CS2 file predates that column
        frame["game"] = game
    return frame


def load_map_rows(path: Path = RAW, with_csgo: bool = True) -> pd.DataFrame:
    """Map-level rows from every available source, tagged by game.

    The CS:GO era comes from the two adapters (scripts/adapt_csgo_dataset.py for
    2021-2023 and scripts/adapt_csgo_history.py for 2015-2020), each writing into
    data/interim/ (git-ignored). A file contributes only when it exists, so a checkout
    that never ran an adapter behaves exactly as it did before.
    """
    frames = [_read_map_file(path, "cs2")]
    if with_csgo:
        for extra in (CSGO_ADAPTER, CSGO_HISTORY):
            if extra.exists():
                frames.append(_read_map_file(extra, "csgo"))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def filter_games(rows: pd.DataFrame, games: list[str] | None) -> pd.DataFrame:
    """Narrow to the requested games ('cs2' / 'csgo'); None or empty keeps everything."""
    if not games:
        return rows
    wanted = {game.casefold() for game in games}
    return rows[rows["game"].astype(str).str.casefold().isin(wanted)]


def player_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (map, player) with the side already resolved.

    Orientation is the trap this function exists to avoid: ``team1_win`` is
    side-relative, so a player's "won" flag is the flag itself on team1 and its
    negation on team2 — never a comparison against a team *name*.
    """
    if "rounds" not in rows.columns:          # derive it if the caller did not
        rows = rows.assign(
            rounds=pd.to_numeric(rows["score1_game"], errors="coerce")
            + pd.to_numeric(rows["score2_game"], errors="coerce")
        )
    records = []
    for side in SIDES:
        other = "team2" if side == "team1" else "team1"
        won_side = np.where(
            side == "team1",
            rows["team1_win"].astype(bool),
            ~rows["team1_win"].astype(bool),
        )
        base = {
            "match_id": rows["match_id"].to_numpy(),
            "game_id": rows["game_id"].to_numpy(),
            "datetime": rows["datetime"].to_numpy(),
            "event": rows["tournament"].to_numpy(),
            "map_name": rows["map_name"].to_numpy(),
            "side": side,
            "team": rows[side].to_numpy(),
            "opponent": rows[other].to_numpy(),
            "rounds": rows["rounds"].to_numpy(),
            "won": won_side,
            # kept so a player's label can prefer the curated CS2 spelling (see label_for)
            "game": (rows["game"].to_numpy() if "game" in rows.columns else "cs2"),
        }
        for i in range(1, 6):
            slot = {
                **base,
                "player": rows[f"{side}_player{i}"].to_numpy(),
                "slot": i,
            }
            for stat in STATS:
                slot[stat] = pd.to_numeric(
                    rows[f"{side}_player{i}_{stat}"], errors="coerce"
                ).to_numpy()
            records.append(pd.DataFrame(slot))
    long = pd.concat(records, ignore_index=True).dropna(subset=["player"])
    long = long[long["player"].astype(str).str.strip() != ""]
    # KAST arrives on a 0-100 scale in this source; keep it as a fraction
    long["kast"] = np.where(long["kast"] > 1.5, long["kast"] / 100.0, long["kast"])
    # Era-independent identity. Each source writes the same person differently — "Oleksandr"
    # vs "Aleksandr" Kostyliev, "Donk" vs "donk" — so grouping on the raw string would
    # silently split one career into two players. Nickname + country is what identifies a
    # player across sources; the raw string stays for display.
    long["player_key"] = [key_of(p) for p in long["player"].astype(str)]
    return long.reset_index(drop=True)


def key_of(label: str) -> str:
    """Identity of a player across sources: nickname (case-folded) + country code.

    Handles both label shapes the sources use — 'Nick (Real), CC' and the short
    'Nick, CC' (the 2015-2020 file carries no real names) — so the same person keys the
    same everywhere and their eras add up instead of splitting into two players.
    """
    text = str(label).strip()
    head, _, tail = text.split(" (")[0].partition(",")
    nick = head.strip().casefold()
    if ")," in text:
        code = text.rsplit("),", 1)[1].strip().casefold()
    else:
        code = tail.strip().casefold()
    return f"{nick}|{code}"


def _load_corrected(path: Path) -> dict[str, str]:
    """key -> your own spelling, from the sheet's `corrected_latin` column."""
    out: dict[str, str] = {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return out
    for row in rows:
        canonical = (row.get("dataset_string") or "").strip()
        fixed = (row.get("corrected_latin") or "").strip()
        if canonical and fixed:
            out[key_of(canonical)] = fixed
    return out


_CORRECTED: dict[str, str] | None = None


def corrected_names(path: Path | None = None) -> dict[str, str]:
    """Cached read — the sheet is tiny and does not change during a run."""
    global _CORRECTED
    if _CORRECTED is None:
        _CORRECTED = _load_corrected(path or NAMES_CSV)
    return _CORRECTED


def label_for(long: pd.DataFrame, key: str, corrected: dict[str, str] | None = None) -> str:
    """The name to display for an identity.

    Order: a `corrected_latin` value from data/player_names.csv wins outright — that is your
    own spelling and the whole point of the sheet — then any spelling that carries the real
    name, then the curated CS2 form. Falls back to the key when nothing is known.
    """
    fixed = (corrected if corrected is not None else corrected_names()).get(key)
    if fixed:
        return fixed
    rows = long[long["player_key"] == key]
    if rows.empty:
        return key
    rich = rows[rows["player"].astype(str).str.contains(r"\(")]
    if not rich.empty:
        rows = rich              # 's1mple, UA' loses to 's1mple (…), UA' when both exist
    if "game" in rows.columns:
        cs2 = rows[rows["game"].astype(str).str.casefold() == "cs2"]
        if not cs2.empty:
            rows = cs2
    return str(rows["player"].astype(str).mode().iloc[0])


# --------------------------------------------------------------------------- pick

NAMES_CSV = REPO / "data" / "player_names.csv"

# A name typed the way a player is *written* carries decoration the data does not have:
# guillemets around the nickname ("Данил «donk» Крышковец"), low/high quotes, or curly
# apostrophes. Strip that so the native form still matches.
_QUOTED = re.compile(r"[«„\"“”'‘’]([^«»„“”\"'‘’]*)[»“”'’]")
_STRIP = str.maketrans("", "", "«»„“”\"'‘’")


def name_key(value: str) -> str:
    """Comparison key: decoration removed, whitespace collapsed, case-folded."""
    text = _QUOTED.sub(" ", value.replace("_", " "))
    return " ".join(text.translate(_STRIP).split()).casefold()


def load_name_aliases(path: Path = NAMES_CSV) -> dict[str, set[str]]:
    """Curated native / corrected names, keyed by the dataset's own player string.

    Reads data/player_names.csv (the review sheet Illia fills in). Only rows with a
    filled column contribute, so an untouched sheet changes nothing. The native name
    must come from the player's own language — a Ukrainian name is *not* transcribed
    from its Russian form.
    """
    aliases: dict[str, set[str]] = {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return aliases                       # no sheet yet: not an error
    for row in rows:
        canonical = (row.get("dataset_string") or "").strip()
        if not canonical:
            continue
        extra = {(row.get(col) or "").strip() for col in ("native_name", "corrected_latin")}
        extra.discard("")
        if extra:
            aliases[canonical] = extra
    return aliases


def alias_hits(key: str, aliases: dict[str, set[str]]) -> tuple[set[str], set[str]]:
    """Split a typed name's curated matches into (exact, loose).

    exact  — the typed name *is* the curated one.
    loose  — it is contained in it, or every word of it appears in it, which is what lets
             "Дмитрий Соколов" find "Дмитрий Эдуардович Соколов" when the page records the
             patronymic.
    """
    exact: set[str] = set()
    loose: set[str] = set()
    parts = set(key.split())
    for canonical, alternatives in aliases.items():
        for alt in alternatives:
            alt_key = name_key(alt)
            if key == alt_key:
                exact.add(canonical)
            elif key in alt_key or (parts and all(part in alt_key.split() for part in parts)):
                loose.add(canonical)
    return exact, loose


def resolve(name: str, long: pd.DataFrame, aliases: dict[str, set[str]] | None = None) -> str:
    """Resolve a player name.

    Order: exact match → case-insensitive exact match → nickname on its own → a
    fragment that is unique in the data → a curated native/corrected name from
    data/player_names.csv. Anything matching several players is an error, not a
    guess: quietly comparing the wrong player would produce a confidently wrong
    report.
    """
    names = list(pd.Index(long["player"].unique()))
    name = name.strip()                       # a copy-pasted name often carries stray spaces
    if name in names:
        return name
    loose = [n for n in names if str(n).casefold() == name.casefold()]
    if len(loose) == 1:
        return loose[0]
    # The nickname on its own — "NiKo" for "NiKo (Nikola Kovač), BA". This is the part
    # people actually type, and without it a substring search finds the twelve players
    # whose *real* name merely contains "niko". A nickname shared by two players still
    # falls through to the error below rather than picking one — but two *spellings* of the
    # same person (the CS2 and CS:GO sources disagree on "Aleksandr"/"Oleksandr", and on
    # capitalisation) count as one player, which is why this dedupes by identity.
    nick = [n for n in names if str(n).split(" (")[0].casefold() == name.casefold()]
    nick_keys = {key_of(n) for n in nick}
    if len(nick_keys) == 1:
        return sorted(nick)[0]
    if len(nick_keys) > 1:
        # Two different people really do share this nickname (only four do in this
        # dataset). Name exactly those, not the dozen substring hits, and ask for the
        # disambiguating part instead of guessing one of them. The wording keeps the
        # "matches N players" phrasing the existing contract test asserts on.
        raise SystemExit(
            f"'{name}' matches {len(nick_keys)} players who share that nickname — add the real "
            "name or country:\n  " + "\n  ".join(sorted(nick))
        )
    partial = sorted(n for n in names if name.casefold() in str(n).casefold())
    if len({key_of(n) for n in partial}) == 1:
        return partial[0]
    if len(partial) > 1:
        raise SystemExit(
            f"'{name}' matches {len(partial)} players — be more specific:\n  "
            + "\n  ".join(partial[:15])
        )
    # Curated native / corrected names (data/player_names.csv). Cyrillic or a corrected
    # Latin spelling can never match the dataset's own strings, so it is tried last — and
    # only rows that have been filled in contribute, so an empty sheet changes nothing.
    if aliases:
        key = name_key(name)
        exact, near = alias_hits(key, aliases)
        if len(exact) == 1:
            return next(iter(exact))
        if len(exact) > 1:
            raise SystemExit(
                f"'{name}' is listed for {len(exact)} players — be more specific:\n  "
                + "\n  ".join(sorted(exact))
            )
        if len(near) == 1:
            return next(iter(near))
        if len(near) > 1:
            raise SystemExit(
                f"'{name}' matches {len(near)} players — be more specific:\n  "
                + "\n  ".join(sorted(near))
            )
    raise SystemExit(f"no player matches '{name}'. Run --list to see the most-played names.")


# --------------------------------------------------------------------------- stats


def aggregates(frame: pd.DataFrame) -> dict:
    """The ingredient metrics, each with the sample size it is based on."""
    maps = len(frame)
    kills = float(frame["kills"].sum())
    deaths = float(frame["deaths"].sum())
    rounds = float(frame.loc[frame["rounds"].notna(), "rounds"].sum())
    scored = int(frame["rounds"].notna().sum())
    return {
        "maps": maps,
        "kills": kills,
        "deaths": deaths,
        "assists": float(frame["assists"].sum()),
        "kd": (kills / deaths) if deaths else float("nan"),
        "adr": float(frame["adr"].mean()) if maps else float("nan"),
        "kast": float(frame["kast"].mean()) if maps else float("nan"),
        "kpr": (kills / rounds) if rounds else float("nan"),
        "rounds": rounds,
        "scored_maps": scored,
        "window_start": frame["datetime"].min(),
        "window_end": frame["datetime"].max(),
        "events": int(frame["event"].nunique()),
        "map_win_rate": float(frame["won"].mean()) if maps else float("nan"),
    }


def per_map(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in frame.groupby("map_name"):
        agg = aggregates(group)
        rows.append(
            {
                "map": name,
                "maps": agg["maps"],
                "kd": agg["kd"],
                "adr": agg["adr"],
                "kast": agg["kast"],
                "kpr": agg["kpr"],
                "win_rate": agg["map_win_rate"],
            }
        )
    return pd.DataFrame(rows).sort_values("maps", ascending=False).reset_index(drop=True)


def per_event(frame: pd.DataFrame, min_maps: int = 5) -> pd.DataFrame:
    rows = []
    for name, group in frame.groupby("event"):
        if len(group) < min_maps:
            continue
        agg = aggregates(group)
        rows.append(
            {
                "event": name,
                "maps": agg["maps"],
                "kd": agg["kd"],
                "adr": agg["adr"],
                "kast": agg["kast"],
                "start": frame["datetime"].min(),
                "end": frame["datetime"].max(),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["event", "maps", "kd", "adr", "kast", "start", "end"])
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def head_to_head(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Maps where both players were on the server, and who won them."""
    joined = a.merge(
        b,
        on=["match_id", "game_id"],
        suffixes=("_a", "_b"),
        how="inner",
    )
    if joined.empty:
        return {"maps": 0, "shared_maps": pd.DataFrame()}
    joined = joined.assign(
        a_won=joined["won_a"].astype(bool),
        b_won=joined["won_b"].astype(bool),
    )
    same_team = (joined["team_a"] == joined["team_b"]).mean()
    shared = (
        pd.DataFrame(
            {
                "date": joined["datetime_a"].dt.date.astype(str).to_numpy(),
                "event": joined["event_a"].to_numpy(),
                "map": joined["map_name_a"].to_numpy(),
                "result": np.where(joined["a_won"].to_numpy(), "A", "B"),
                "a_adr": joined["adr_a"].round(1).to_numpy(),
                "b_adr": joined["adr_b"].round(1).to_numpy(),
                "a_kd": (joined["kills_a"] / joined["deaths_a"].replace(0, np.nan))
                .round(2)
                .to_numpy(),
                "b_kd": (joined["kills_b"] / joined["deaths_b"].replace(0, np.nan))
                .round(2)
                .to_numpy(),
                "same_team": (joined["team_a"] == joined["team_b"]).to_numpy(),
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    return {
        "maps": len(joined),
        "a_map_wins": int(joined["a_won"].sum()),
        "b_map_wins": int(joined["b_won"].sum()),
        "a_adr": float(joined["adr_a"].mean()),
        "b_adr": float(joined["adr_b"].mean()),
        "a_kd": float(joined["kills_a"].sum() / max(joined["deaths_a"].sum(), 1)),
        "b_kd": float(joined["kills_b"].sum() / max(joined["deaths_b"].sum(), 1)),
        "same_team_share": float(same_team),
        "shared_maps": shared,
    }


# --------------------------------------------------------------------------- render


def _f(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.1%}"


def md_table(frame: pd.DataFrame, headers: dict[str, str], percent: set[str] | None = None) -> str:
    percent = percent or set()
    cols = list(frame.columns)
    out = ["| " + " | ".join(headers.get(c, c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.itertuples(index=False):
        cells = []
        for column, value in zip(cols, row, strict=True):
            if column in percent:
                cells.append(_pct(float(value)))
            elif isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
                cells.append(str(int(value)) if float(value).is_integer() else _f(float(value), 2))
            elif pd.isna(value):
                cells.append("—")
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def comparison_rows(a_name: str, b_name: str, agg_a: dict, agg_b: dict) -> pd.DataFrame:
    def pair(label: str, key: str, fmt=lambda v: _f(v, 2)) -> dict:
        return {"metric": label, "a": fmt(agg_a[key]), "b": fmt(agg_b[key])}

    return pd.DataFrame(
        [
            pair("Maps played", "maps", lambda v: str(int(v))),
            pair("Kills", "kills", lambda v: str(int(v))),
            pair("Deaths", "deaths", lambda v: str(int(v))),
            pair("Assists", "assists", lambda v: str(int(v))),
            pair("K/D", "kd", lambda v: _f(v, 2)),
            pair("ADR", "adr", lambda v: _f(v, 1)),
            pair("KAST", "kast", lambda v: _pct(v)),
            pair("Kills per round", "kpr", lambda v: _f(v, 3)),
            pair("Map win rate", "map_win_rate", lambda v: _pct(v)),
            pair("Events", "events", lambda v: str(int(v))),
        ]
    ).rename(columns={"a": a_name, "b": b_name})


def render_md(
    a_name: str,
    b_name: str,
    agg_a: dict,
    agg_b: dict,
    maps_a: pd.DataFrame,
    maps_b: pd.DataFrame,
    events_a: pd.DataFrame,
    events_b: pd.DataFrame,
    h2h: dict,
    long: pd.DataFrame,
) -> str:
    window = f"{agg_a['window_start'].date()} → {agg_a['window_end'].date()}"
    coverage = (
        f"{agg_a['scored_maps']}/{agg_a['maps']} of A's maps carry round scores; "
        f"KPR is computed on those {agg_a['rounds']:.0f} rounds."
    )
    out = [
        f"# {a_name} vs {b_name}",
        "",
        f"Window **{window}** · source: `data/raw/cs2_all_tiers_games.csv` (map rows only) · "
        f"{len(long):,} player-map rows scanned.",
        "",
        "## Headline",
        "",
        md_table(comparison_rows(a_name, b_name, agg_a, agg_b), {"metric": "Metric"}),
        "",
        f"_{coverage}_",
        "",
    ]
    if h2h["maps"]:
        same_team = h2h["same_team_share"]
        note = (
            "**Note:** they shared a team on "
            f"{same_team:.0%} of these maps, so 'who won' is not always a duel."
            if same_team > 0.05
            else "They were always on opposite sides in these maps."
        )
        out += [
            "## Head to head (maps where both were on the server)",
            "",
            f"**{h2h['maps']} maps** · {a_name.split(' ')[0]}'s team won {h2h['a_map_wins']}, "
            f"{b_name.split(' ')[0]}'s team won {h2h['b_map_wins']}. "
            f"ADR {_f(h2h['a_adr'], 1)} vs {_f(h2h['b_adr'], 1)}, "
            f"K/D {_f(h2h['a_kd'], 2)} vs {_f(h2h['b_kd'], 2)} in exactly those games.",
            "",
            note,
            "",
            md_table(
                h2h["shared_maps"].tail(15),
                {
                    "date": "Date",
                    "event": "Event",
                    "map": "Map",
                    "result": "Winner",
                    "a_adr": "ADR (A)",
                    "b_adr": "ADR (B)",
                    "a_kd": "K/D (A)",
                    "b_kd": "K/D (B)",
                    "same_team": "Same team",
                },
            ),
            "",
            "_(last 15 shared maps shown; the full list is in the CSV)_",
            "",
        ]
    else:
        out += [
            "## Head to head",
            "",
            "They never appeared on the same map inside this window.",
            "",
        ]

    for label, table in (
        (f"{a_name} — by map", maps_a),
        (f"{b_name} — by map", maps_b),
    ):
        out += [
            f"## {label}",
            "",
            md_table(
                table.head(12),
                {
                    "map": "Map",
                    "maps": "Maps",
                    "kd": "K/D",
                    "adr": "ADR",
                    "kast": "KAST",
                    "kpr": "KPR",
                    "win_rate": "Win",
                },
            ),
            "",
        ]
    for label, table in ((f"{a_name} — by event", events_a), (f"{b_name} — by event", events_b)):
        if table.empty:
            continue
        view = table.assign(
            start=table["start"].dt.date.astype(str), end=table["end"].dt.date.astype(str)
        )
        out += [
            f"## {label}",
            "",
            md_table(
                view[["event", "maps", "kd", "adr", "kast", "start", "end"]],
                {
                    "event": "Event",
                    "maps": "Maps",
                    "kd": "K/D",
                    "adr": "ADR",
                    "kast": "KAST",
                    "start": "First",
                    "end": "Last",
                },
            ),
            "",
        ]
    out += [
        "## Method and caveats",
        "",
        "- Rows are map-level only (`is_total` series summaries are dropped).",
        "- **KPR** = kills ÷ the map's actual rounds (`score1_game + score2_game`), so it is a",
        "  rate, not a per-map average; maps without scores are excluded and counted above.",
        "- **KAST** is stored 0–100 in the source and shown as a fraction.",
        "- ADR/KAST are per-map averages; K/D and KPR are pooled totals, which is the standard",
        "  way to avoid small-sample inflation.",
        "- No synthetic 'rating' is invented: the HLTV Rating formula is not public, so this",
        "  reports the ingredients and lets the numbers speak.",
        "- The source has gaps (see `docs/VITALITY-ANALYSIS.md`): every rate names its own n.",
        "",
        "---",
        "",
        "_Generated by `scripts/compare_players.py`._",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- main


def load_window(start: str | None = None, end: str | None = None, maps: list[str] | None = None,
                events: list[str] | None = None, tiers: list[int] | None = None,
                games: list[str] | None = None) -> pd.DataFrame:
    """Map rows, narrowed by whatever the caller asked for.

    Every filter is optional and independent: dates, map names, event substrings, tiers and
    the game era (cs2 / csgo). Nothing is pre-computed — the comparison is built from
    whatever survives here, which is what makes an ad-hoc question ("Mirage only, tier 1,
    CS:GO, since March") work without a prepared report.

    Careful with dates: CS:GO rows carry only an event-derived date (see
    scripts/adapt_csgo_dataset.py) and 57% of them have none, so a --from/--to window
    silently drops those maps.
    """
    rows = filter_games(load_map_rows(), games)
    if start:
        rows = rows[rows["datetime"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        rows = rows[
            rows["datetime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
        ]
    if maps:
        wanted = {m.casefold() for m in maps}
        rows = rows[rows["map_name"].astype(str).str.casefold().isin(wanted)]
    if events:
        mask = pd.Series(False, index=rows.index)
        for needle in events:
            mask |= rows["tournament"].astype(str).str.contains(needle, case=False, regex=False)
        rows = rows[mask]
    if tiers:
        tier_of = tier_map()
        keep = rows["match_id"].astype(str).map(tier_of)
        rows = rows[keep.isin({str(t) for t in tiers})]
    return rows


def tier_map() -> dict[str, str]:
    """match_id -> tier, from the per-tier files the dataset already ships.

    Read lazily and cached: the tier files are the only place a match's tier is
    recorded, and they are big enough that re-reading them per filter would be rude.
    """
    if "_tier_cache" not in tier_map.__dict__:
        mapping: dict[str, str] = {}
        for tier in (1, 2, 3):
            path = RAW.parent / f"cs2_tier{tier}_games.csv"
            if not path.exists():
                continue
            ids = pd.read_csv(path, usecols=["match_id"], low_memory=False)["match_id"]
            for match_id in ids.dropna().astype(str).unique():
                mapping.setdefault(match_id, str(tier))   # first file wins
        tier_map._tier_cache = mapping          # type: ignore[attr-defined]
    return tier_map._tier_cache                 # type: ignore[attr-defined]


def _ask(prompt: str, default: str = "") -> str:
    """One question, with a default shown; blank input keeps the default."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer or default


def _choose_player(label: str, long: pd.DataFrame) -> str | None:
    """Let the user search, see the matches, and pick one by number or name."""
    names = long["player"].value_counts()
    print(f"\n{label}")
    while True:
        query = _ask("  type part of a name (empty = top 15)")
        hits = list(names.head(15).index) if not query else [
            n for n in names.index if query.casefold() in str(n).casefold()
        ]
        if not hits:
            print(f"  nothing matches {query!r} — try again")
            continue
        for i, name in enumerate(hits[:15], start=1):
            print(f"   {i:2d}. {names[name]:5d} maps  {name}")
        answer = _ask("  pick a number, or type the exact name (empty = cancel)")
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(hits[:15]):
            return hits[int(answer) - 1]
        return answer                      # resolve() validates it for real


def interactive_pick(args: argparse.Namespace) -> argparse.Namespace | None:
    """Ask for both players and every optional filter, then hand back the arguments.

    This is the point of the tool: the user asks an ad-hoc question and it is computed
    on the spot — there is no committed report to be stuck with.
    """
    rows = load_map_rows()
    long = player_frame(rows)
    counts = long["player"].value_counts()

    print("Interactive player comparison — nothing is pre-computed.")
    print(f"  data: {len(rows):,} map rows · {long['game_id'].nunique():,} maps · "
          f"{len(counts):,} players · {long['datetime'].min().date()} → "
          f"{long['datetime'].max().date()}")

    a = _choose_player("Player A", long)
    if not a:
        print("cancelled.")
        return None
    b = _choose_player("Player B", long)
    if not b:
        print("cancelled.")
        return None
    args.a, args.b = a, b

    print("\nOptional filters — press Enter to skip any of them.")
    print(f"  maps available: {', '.join(sorted(rows['map_name'].dropna().unique()))}")
    picked_maps = _ask("  maps (comma-separated, e.g. Mirage,Dust2)")
    if picked_maps:
        args.maps = [m.strip() for m in picked_maps.split(",") if m.strip()]

    events = list(rows["tournament"].dropna().unique())
    print(f"  {len(events)} events available, e.g. {', '.join(events[:4])}")
    picked_events = _ask("  event text (comma-separated, substring match)")
    if picked_events:
        args.events = [e.strip() for e in picked_events.split(",") if e.strip()]

    picked_tiers = _ask("  tiers (1,2,3 — comma-separated, empty = all)")
    if picked_tiers:
        args.tiers = [int(t) for t in picked_tiers.split(",") if t.strip().isdigit()]

    args.start = _ask("  from (YYYY-MM-DD)", str(long["datetime"].min().date())) or None
    args.end = _ask("  to (YYYY-MM-DD)", str(long["datetime"].max().date())) or None

    kept = load_window(args.start, args.end, args.maps or None, args.events or None,
                       args.tiers or None, getattr(args, "games", None))
    print(f"\nfiltered rows: {kept.shape[0]:,}")
    return args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HLTV-style two-player comparison from CS2 map rows."
    )
    parser.add_argument("--a", help="player A (exact name as in the data, or a unique fragment)")
    parser.add_argument("--b", help="player B")
    parser.add_argument("--from", dest="start", help="window start (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end", help="window end (YYYY-MM-DD)")
    parser.add_argument(
        "--list",
        nargs="?",
        const="",
        metavar="FRAGMENT",
        help="list the most-played players (optionally only names matching a fragment)",
    )
    parser.add_argument("--md", metavar="PATH", help="also write the Markdown report here")
    parser.add_argument("--map", dest="maps", action="append", default=[], metavar="NAME",
                        help="only these maps (repeatable: --map Mirage --map Dust2)")
    parser.add_argument("--event", dest="events", action="append", default=[], metavar="TEXT",
                        help="only tournaments whose name contains TEXT (repeatable)")
    parser.add_argument("--tier", dest="tiers", action="append", type=int, default=[],
                        choices=[1, 2, 3], help="only this tier (repeatable): 1, 2 or 3")
    parser.add_argument("--game", dest="games", action="append", default=[],
                        choices=["csgo", "cs2"],
                        help="only this game era (repeatable): csgo, cs2 — default both")
    parser.add_argument("--pick", action="store_true",
                        help="choose both players and the filters interactively")
    parser.add_argument(
        "--min-event-maps",
        type=int,
        default=5,
        help="drop events with fewer maps than this from the per-event tables",
    )
    args = parser.parse_args(argv)

    if args.pick:
        chosen = interactive_pick(args)
        if chosen is None:
            return 0
        args = chosen

    rows = load_window(args.start, args.end, args.maps or None, args.events or None,
                       args.tiers or None, args.games or None)
    long = player_frame(rows)

    if args.list is not None:
        # counted per identity, so a player's CS:GO and CS2 maps add up instead of
        # appearing as two people who merely share a nickname
        counts = long["player_key"].value_counts()
        labels = {key: label_for(long, key) for key in counts.index}
        if args.list:
            # search the identities, the nicknames AND the curated native names,
            # so "Костил" finds s1mple once his Ukrainian name is filled in
            needle = name_key(args.list)
            curated = {c for c, alts in load_name_aliases().items()
                       if any(needle in name_key(a) for a in alts)}
            counts = counts[[key for key in counts.index
                             if args.list.casefold() in labels[key].casefold()
                             or name_key(labels[key].split(" (")[0]) == needle
                             or labels[key] in curated]]
        print(f"{len(counts)} player(s){' matching ' + args.list if args.list else ''}:")
        for key, count in counts.head(25).items():
            print(f"  {count:5d} maps  {labels[key]}")
        return 0

    if not args.a or not args.b:
        parser.error("--a and --b are required (or use --list / --pick)")

    aliases = load_name_aliases()
    a_name, b_name = resolve(args.a, long, aliases), resolve(args.b, long, aliases)
    # resolve() returns the source's spelling; the identity is what spans both eras, so the
    # filter and the same-player guard use the key, and the label is the curated spelling.
    a_key, b_key = key_of(a_name), key_of(b_name)
    if a_key == b_key:
        parser.error("--a and --b resolve to the same player")
    a_name, b_name = label_for(long, a_key), label_for(long, b_key)

    frame_a = long[long["player_key"] == a_key]
    frame_b = long[long["player_key"] == b_key]
    agg_a, agg_b = aggregates(frame_a), aggregates(frame_b)
    h2h = head_to_head(frame_a, frame_b)
    maps_a, maps_b = per_map(frame_a), per_map(frame_b)
    events_a = per_event(frame_a, args.min_event_maps)
    events_b = per_event(frame_b, args.min_event_maps)

    print(
        f"A: {a_name}   maps={agg_a['maps']}  K/D={_f(agg_a['kd'])}  ADR={_f(agg_a['adr'], 1)}"
        f"  KAST={_pct(agg_a['kast'])}  KPR={_f(agg_a['kpr'], 3)}"
        f"  win={_pct(agg_a['map_win_rate'])}"
    )
    print(
        f"B: {b_name}   maps={agg_b['maps']}  K/D={_f(agg_b['kd'])}  ADR={_f(agg_b['adr'], 1)}"
        f"  KAST={_pct(agg_b['kast'])}  KPR={_f(agg_b['kpr'], 3)}"
        f"  win={_pct(agg_b['map_win_rate'])}"
    )
    print(
        f"head-to-head maps: {h2h['maps']}"
        + (f"  (A's team won {h2h['a_map_wins']}, B's {h2h['b_map_wins']})" if h2h["maps"] else "")
    )
    if "game" in rows.columns:
        eras = rows["game"].value_counts().to_dict()
        print("eras in this view: "
              + ", ".join(f"{k} {v:,} maps" for k, v in sorted(eras.items()))
              + ("   (use --game csgo|cs2 to restrict)" if len(eras) > 1 else ""))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = lambda n: "".join(c if c.isalnum() else "_" for c in n.split(" (")[0]).strip("_")  # noqa: E731
    head_csv = OUT_DIR / f"player_comparison_{slug(a_name)}_vs_{slug(b_name)}.csv"
    with head_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "metric", a_name, b_name])
        for row in comparison_rows(a_name, b_name, agg_a, agg_b).itertuples(index=False):
            writer.writerow(["headline", row[0], row[1], row[2]])
        writer.writerow(["head_to_head", "shared maps", h2h["maps"], h2h["maps"]])
        writer.writerow(["head_to_head", "A team map wins", h2h.get("a_map_wins", 0), ""])
        writer.writerow(["head_to_head", "B team map wins", h2h.get("b_map_wins", 0), ""])
        if h2h["maps"]:
            for row in h2h["shared_maps"].itertuples(index=False):
                writer.writerow(
                    ["shared_map", f"{row.date} {row.event} {row.map}", row.a_adr, row.b_adr]
                )
    per_map_path = OUT_DIR / f"player_comparison_{slug(a_name)}_vs_{slug(b_name)}_by_map.csv"
    maps_a.assign(player=a_name).to_csv(per_map_path, index=False)

    if args.md:
        target = Path(args.md)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render_md(a_name, b_name, agg_a, agg_b, maps_a, maps_b, events_a, events_b, h2h, long),
            encoding="utf-8",
        )
        print(f"wrote {target.relative_to(REPO) if target.is_relative_to(REPO) else target}")
    print(f"wrote {head_csv.name} and {per_map_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
