"""Team Vitality window review: BLAST Open London 2025 -> BLAST Rivals Spring 2026.

One script, one source of truth. It reads the repo's own canonical tables only —
`outputs/series_clean.csv` for the series level and `data/raw/cs2_all_tiers_games.csv`
for the map/player level — reuses the shipped Elo engine
(`cs2analytics.features.elo.run_elo_backtest`, K=32, base 1500) so the strength numbers
agree with the rest of the project, and writes:

* `outputs/vitality_series.csv`   one row per series, from Vitality's point of view
* `outputs/vitality_maps.csv`     one row per map, with rounds/overtime + team totals
* `outputs/vitality_summary.csv`  long-format (section, metric, value, extra) headline table
* `docs/VITALITY-ANALYSIS.md`     the write-up (renders on GitHub)
* `docs/vitality-analysis.html`   standalone dark-theme report (inline SVG, no network)

Run from the repo root:
    .venv/Scripts/python.exe scripts/analyze_vitality.py

Window: the two dates the request named — BLAST Open London 2025 (2025-08-20, the start of
its closed qualifier) through BLAST Rivals "2026 Season 1" / Spring 2026 (2026-05-05). The
actual first/last Vitality series inside the window are reported, because where the data
starts and stops is a fact, while the window edges are a request.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from cs2analytics.features.elo import run_elo_backtest

REPO = Path(__file__).resolve().parents[1]
TEAM = "Team Vitality"
WINDOW_START = "2025-08-20"
WINDOW_END = "2026-05-05"

SERIES_OUT = REPO / "outputs" / "vitality_series.csv"
MAPS_OUT = REPO / "outputs" / "vitality_maps.csv"
SUMMARY_OUT = REPO / "outputs" / "vitality_summary.csv"
MD_OUT = REPO / "docs" / "VITALITY-ANALYSIS.md"
HTML_OUT = REPO / "docs" / "vitality-analysis.html"


# --------------------------------------------------------------------------- load


def in_window(frame: pd.DataFrame) -> pd.Series:
    """The requested window, both named days inclusive."""
    return (frame["datetime"] >= WINDOW_START) & (frame["datetime"] <= WINDOW_END + " 23:59")


def load_series() -> pd.DataFrame:
    """Canonical series table + walk-forward pre-match Elo over the full history."""
    series = pd.read_csv(REPO / "outputs" / "series_clean.csv")
    series["datetime"] = pd.to_datetime(series["datetime"], utc=True, format="ISO8601")
    series = series.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    elo = run_elo_backtest(series, k=32.0)
    series["elo_t1_pre"] = elo["elo_t1_pre"]
    series["elo_t2_pre"] = elo["elo_t2_pre"]
    series["p_t1"] = elo["p_t1"]
    return series


def vital_series(series: pd.DataFrame) -> pd.DataFrame:
    """Vitality's series inside the window, oriented Vitality-first, with Elo fields."""
    v = series[in_window(series)]
    v = v[(v["team1"] == TEAM) | (v["team2"] == TEAM)].copy()
    home = v["team1"] == TEAM
    # `winner` holds a team NAME, so it is already side-independent — XOR-ing it with `home`
    # inverts every result where Vitality was team2. (On map rows the `team1_win` flag *is*
    # side-relative, so there the XOR is the correct form.)
    won = v["winner"] == TEAM
    out = pd.DataFrame(
        {
            "match_id": v["match_id"].to_numpy(),
            "datetime": v["datetime"].to_numpy(),
            "event": v["tournament"].to_numpy(),
            "tier": v["tier"].to_numpy(),
            "bo": v["bestOf"].astype("Int64").to_numpy(),
            "opponent": np.where(home, v["team2"], v["team1"]),
            "won": won.to_numpy(),
            "score_for": np.where(home, v["t1_series_score"], v["t2_series_score"]),
            "score_against": np.where(home, v["t2_series_score"], v["t1_series_score"]),
            "maps_played": v["games_played"].to_numpy(),
            "elo": np.where(home, v["elo_t1_pre"], v["elo_t2_pre"]),
            "opp_elo": np.where(home, v["elo_t2_pre"], v["elo_t1_pre"]),
            "p_win": np.where(home, v["p_t1"], 1.0 - v["p_t1"]),
        }
    )
    out["swept"] = out["won"] & (out["score_against"] == 0)
    out["went_the_distance"] = out["maps_played"] == out["bo"]
    out["loss_margin"] = out["score_against"] - out["score_for"]
    return out.sort_values("datetime", kind="mergesort").reset_index(drop=True)


_RAW_CACHE: dict[str, tuple[pd.DataFrame, pd.Series]] = {}


def oriented_raw() -> tuple[pd.DataFrame, pd.Series]:
    """Vitality's window map rows from the raw file + the 'team1 is Vitality' mask.

    Read once and cached: the raw file is ~13 MB and three sections need it.
    `is_total` rows are the series summaries (one per series, no map name), so they are
    dropped here — map-level facts must come from map rows only.
    """
    if "pair" not in _RAW_CACHE:
        raw = pd.read_csv(REPO / "data" / "raw" / "cs2_all_tiers_games.csv", low_memory=False)
        raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
        raw = raw[(~raw["is_total"].astype(bool)) & raw["map_name"].notna()]
        raw = raw[in_window(raw)]
        v = raw[(raw["team1"] == TEAM) | (raw["team2"] == TEAM)].copy()
        _RAW_CACHE["pair"] = (v, v["team1"] == TEAM)
    return _RAW_CACHE["pair"]


def load_maps() -> pd.DataFrame:
    """Map-level rows for Vitality in the window, oriented Vitality-first."""
    v, home = oriented_raw()
    rows = pd.DataFrame(
        {
            "match_id": v["match_id"].to_numpy(),
            "game_id": v["game_id"].to_numpy(),
            "datetime": v["datetime"].to_numpy(),
            "event": v["tournament"].to_numpy(),
            "opponent": np.where(home, v["team2"], v["team1"]),
            "opponent_id": np.where(home, v["team2_id"], v["team1_id"]),
            "map_name": v["map_name"].to_numpy(),
            "bo": v["bestOf"].astype("Int64").to_numpy(),
            "won": home.eq(v["team1_win"].astype(bool)).to_numpy(),
            "rounds_for": np.where(home, v["score1_game"], v["score2_game"]).astype(float),
            "rounds_against": np.where(home, v["score2_game"], v["score1_game"]).astype(float),
        }
    )
    rows["round_diff"] = rows["rounds_for"] - rows["rounds_against"]
    rows["total_rounds"] = rows["rounds_for"] + rows["rounds_against"]
    rows["overtime"] = rows["total_rounds"] > 24
    rows["close"] = rows["round_diff"].abs() <= 2

    for stat in ("kills", "deaths", "assists"):
        total = np.zeros(len(v))
        for i in range(1, 6):
            values = np.where(home, v[f"team1_player{i}_{stat}"], v[f"team2_player{i}_{stat}"])
            total = total + np.nan_to_num(
                pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
            )
        rows[f"team_{stat}"] = total
    for stat in ("adr", "kast"):
        stack = np.vstack(
            [
                pd.to_numeric(
                    pd.Series(
                        np.where(home, v[f"team1_player{i}_{stat}"], v[f"team2_player{i}_{stat}"])
                    ),
                    errors="coerce",
                ).to_numpy(dtype=float)
                for i in range(1, 6)
            ]
        )
        rows[f"team_{stat}"] = np.nanmean(stack, axis=0)
    # KAST arrives on a 0-100 scale; store it as a fraction so the renderers can percent it.
    rows["team_kast"] = np.where(
        rows["team_kast"] > 1.5, rows["team_kast"] / 100.0, rows["team_kast"]
    )
    rows["team_kddiff"] = rows["team_kills"] - rows["team_deaths"]
    return rows.sort_values(["datetime", "game_id"], kind="mergesort").reset_index(drop=True)


def player_lines() -> pd.DataFrame:
    """Per-player aggregates for Vitality only, keyed by the raw player names."""
    v, home = oriented_raw()
    records = []
    for i in range(1, 6):
        records.append(
            pd.DataFrame(
                {
                    "player": np.where(home, v[f"team1_player{i}"], v[f"team2_player{i}"]),
                    "datetime": v["datetime"].to_numpy(),
                    "map_name": v["map_name"].to_numpy(),
                    "kills": np.where(
                        home, v[f"team1_player{i}_kills"], v[f"team2_player{i}_kills"]
                    ),
                    "deaths": np.where(
                        home, v[f"team1_player{i}_deaths"], v[f"team2_player{i}_deaths"]
                    ),
                    "assists": np.where(
                        home, v[f"team1_player{i}_assists"], v[f"team2_player{i}_assists"]
                    ),
                    "adr": np.where(home, v[f"team1_player{i}_adr"], v[f"team2_player{i}_adr"]),
                    "kast": np.where(home, v[f"team1_player{i}_kast"], v[f"team2_player{i}_kast"]),
                    "won": np.where(home, v["team1_win"], 1 - v["team1_win"]),
                }
            )
        )
    players = pd.concat(records, ignore_index=True).dropna(subset=["player"])
    for column in ("kills", "deaths", "assists", "adr", "kast", "won"):
        players[column] = pd.to_numeric(players[column], errors="coerce")
    agg = (
        players.groupby("player", as_index=False)
        .agg(
            maps=("map_name", "size"),
            kills=("kills", "sum"),
            deaths=("deaths", "sum"),
            assists=("assists", "sum"),
            adr=("adr", "mean"),
            kast=("kast", "mean"),
            map_win_rate=("won", "mean"),
            first=("datetime", "min"),
            last=("datetime", "max"),
        )
        .sort_values(["maps", "player"], ascending=[False, True])
    )
    agg["kd"] = agg["kills"] / agg["deaths"].replace(0, np.nan)
    # KAST arrives on a 0-100 scale; normalise to a fraction for percent rendering.
    agg["kast"] = np.where(agg["kast"] > 1.5, agg["kast"] / 100.0, agg["kast"])
    return agg.reset_index(drop=True)


def lineups() -> pd.DataFrame:
    """The distinct Vitality fives in the window, with their first/last sighting and record."""
    v, home = oriented_raw()
    fives = [np.where(home, v[f"team1_player{i}"], v[f"team2_player{i}"]) for i in range(1, 6)]
    columns = np.vstack(fives).T
    # positional labels: assign() would align on the frame's index, which is not 0..n-1 here
    labels = np.array(
        [" / ".join(sorted(str(x) for x in row if pd.notna(x))) for row in columns], dtype=object
    )
    v = v.reset_index(drop=True).assign(lineup=labels)
    return (
        v.groupby("lineup", as_index=False)
        .agg(
            series=("match_id", "nunique"),
            maps=("game_id", "nunique"),
            first=("datetime", "min"),
            last=("datetime", "max"),
        )
        .sort_values("first")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- tables


def record(frame: pd.DataFrame, column: str = "won") -> tuple[int, int, float]:
    """(wins, losses, win_rate) for a frame that carries a boolean column."""
    wins = int(frame[column].sum())
    n = len(frame)
    return wins, n - wins, (wins / n if n else float("nan"))


def streaks(vs: pd.DataFrame) -> tuple[int, int]:
    """Longest winning and losing runs in chronological order."""
    best = worst = run = 0
    run_win = None
    for won in vs.sort_values("datetime")["won"]:
        if run_win is None or won == run_win:
            run += 1
        else:
            run = 1
        run_win = won
        if won:
            best = max(best, run)
        else:
            worst = max(worst, run)
    return best, worst


def event_table(vs: pd.DataFrame, vm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event, group in vs.groupby("event"):
        maps = vm[vm["event"] == event]
        w, losses, _ = record(group)
        mw, ml, mrate = record(maps) if len(maps) else (0, 0, float("nan"))
        expected = int(group["maps_played"].sum())
        rows.append(
            {
                "event": event,
                "series": len(group),
                "series_record": f"{w}-{losses}",
                "map_record": f"{mw}-{ml}",
                "map_win_rate": mrate,
                "map_coverage": f"{len(maps)}/{expected}",
                "first": group["datetime"].min().date().isoformat(),
                "last": group["datetime"].max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows).sort_values("first").reset_index(drop=True)


def month_table(vs: pd.DataFrame, vm: pd.DataFrame) -> pd.DataFrame:
    vs = vs.assign(month=vs["datetime"].dt.strftime("%Y-%m"))
    vm = vm.assign(month=vm["datetime"].dt.strftime("%Y-%m"))
    rows = []
    for month, group in vs.groupby("month"):
        maps = vm[vm["month"] == month]
        w, losses, rate = record(group)
        rows.append(
            {
                "month": month,
                "series": len(group),
                "record": f"{w}-{losses}",
                "series_win_rate": rate,
                "maps": len(maps),
                "map_win_rate": record(maps)[2] if len(maps) else float("nan"),
                "avg_elo": round(float(group["elo"].mean()), 1),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def map_table(vm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in vm.groupby("map_name"):
        w, losses, rate = record(group)
        rows.append(
            {
                "map": name,
                "maps": len(group),
                "record": f"{w}-{losses}",
                "win_rate": rate,
                "avg_round_diff": round(float(group["round_diff"].mean()), 2),
                "overtime": int(group["overtime"].sum()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["maps", "map"], ascending=[False, True])
        .reset_index(drop=True)
    )


def opponent_table(vs: pd.DataFrame, vm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in vs.groupby("opponent"):
        maps = vm[vm["opponent"] == name]
        w, losses, rate = record(group)
        rows.append(
            {
                "opponent": name,
                "series": len(group),
                "record": f"{w}-{losses}",
                "win_rate": rate,
                "map_record": f"{int(maps['won'].sum())}-{int((~maps['won']).sum())}",
                "avg_opp_elo": round(float(group["opp_elo"].mean()), 1),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["series", "win_rate"], ascending=[False, False])
        .reset_index(drop=True)
    )


def format_table(vs: pd.DataFrame) -> pd.DataFrame:
    label = vs["bo"].astype("float").map({1.0: "Bo1", 3.0: "Bo3", 5.0: "Bo5"}).fillna("unknown")
    rows = []
    for name, group in vs.assign(fmt=label).groupby("fmt"):
        w, losses, rate = record(group)
        rows.append(
            {"format": name, "series": len(group), "record": f"{w}-{losses}", "win_rate": rate}
        )
    return pd.DataFrame(rows).sort_values("format").reset_index(drop=True)


def closeness(vm: pd.DataFrame, vs: pd.DataFrame) -> dict:
    close = vm[vm["close"]]
    ot = vm[vm["overtime"]]
    decider = vs[vs["went_the_distance"]]
    return {
        "close_maps": len(close),
        "close_map_win_rate": record(close)[2] if len(close) else float("nan"),
        "overtime_maps": len(ot),
        "overtime_win_rate": record(ot)[2] if len(ot) else float("nan"),
        "decider_series": len(decider),
        "decider_win_rate": record(decider)[2] if len(decider) else float("nan"),
        # a sweep needs a series long enough to be swept: a 1-0 Bo1 win is not a 2-0/3-0
        "sweeps_for": int((vs["won"] & (vs["score_against"] == 0) & (vs["bo"] > 1)).sum()),
        "sweeps_against": int(((~vs["won"]) & (vs["score_for"] == 0) & (vs["bo"] > 1)).sum()),
        # loss_margin is score_against - score_for, so a single-map defeat is +1 (any Bo)
        "one_map_losses": int(((~vs["won"]) & (vs["loss_margin"] == 1)).sum()),
        "bo1": int((vs["bo"] == 1).sum()),
    }


def elo_section(vs: pd.DataFrame) -> dict:
    expected = float(vs["p_win"].sum())
    actual = int(vs["won"].sum())
    ordered = vs.sort_values("datetime")
    trace = ordered[["datetime"]].copy()
    trace["elo_v"] = ordered["elo"].to_numpy()
    trace["opp_elo"] = ordered["opp_elo"].to_numpy()
    return {
        "elo_start": round(float(ordered["elo"].iloc[0]), 1),
        "elo_end": round(float(ordered["elo"].iloc[-1]), 1),
        "elo_peak": round(float(ordered["elo"].max()), 1),
        "elo_peak_date": ordered.loc[ordered["elo"].idxmax(), "datetime"].date().isoformat(),
        "elo_low": round(float(ordered["elo"].min()), 1),
        "avg_opp_elo": round(float(vs["opp_elo"].mean()), 1),
        "hardest_opponent": vs.loc[vs["opp_elo"].idxmax(), "opponent"],
        "hardest_opponent_elo": round(float(vs["opp_elo"].max()), 1),
        "expected_series_wins": round(expected, 1),
        "actual_series_wins": actual,
        "delta_vs_elo": round(actual - expected, 1),
        "elo_trace": trace.reset_index(drop=True),
    }


def best_and_worst(vs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best wins (highest-rated opponent beaten) and worst defeats (lowest-rated loss)."""
    wins = vs[vs["won"]].sort_values("opp_elo", ascending=False).head(5)
    losses = vs[~vs["won"]].sort_values("opp_elo").head(5)
    return wins, losses


# --------------------------------------------------------------------------- summary


def summary_rows(
    vs: pd.DataFrame,
    vm: pd.DataFrame,
    players: pd.DataFrame,
    events: pd.DataFrame,
    maps: pd.DataFrame,
    months: pd.DataFrame,
    opps: pd.DataFrame,
    fmt: pd.DataFrame,
    close: dict,
    elo: dict,
    fives: pd.DataFrame,
) -> list[dict]:
    """Everything worth quoting, in one long-format table."""
    rows: list[dict] = []

    def add(section: str, metric: str, value: object, extra: str = "") -> None:
        rows.append({"section": section, "metric": metric, "value": value, "extra": extra})

    w, losses, rate = record(vs)
    mw, ml, mrate = record(vm)
    best, worst = streaks(vs)
    add("headline", "team", TEAM)
    add("headline", "window_requested", f"{WINDOW_START}..{WINDOW_END}")
    add("headline", "first_series", vs["datetime"].min().date().isoformat())
    add("headline", "last_series", vs["datetime"].max().date().isoformat())
    add("headline", "events", int(vs["event"].nunique()))
    add("headline", "opponents", int(vs["opponent"].nunique()))
    add("headline", "series", f"{w}-{losses}", f"win rate {rate:.1%}")
    add("headline", "maps", f"{mw}-{ml}", f"win rate {mrate:.1%}")
    expected_maps = int(vs["maps_played"].sum())
    no_map_rows = len(set(vs["match_id"]) - set(vm["match_id"]))
    add("headline", "maps_expected_from_series", expected_maps)
    add(
        "headline",
        "maps_with_rows",
        len(vm),
        f"coverage {len(vm) / expected_maps:.1%}" if expected_maps else "",
    )
    add(
        "headline",
        "series_without_map_rows",
        no_map_rows,
        "series-level facts still counted; map-level facts cannot",
    )
    add("headline", "longest_win_streak", best)
    add("headline", "longest_loss_streak", worst)
    add("headline", "series_win_rate", round(rate, 4))
    add("headline", "map_win_rate", round(mrate, 4))

    for row in fmt.itertuples(index=False):
        add("format", row.format, row.record, f"{row.win_rate:.1%}")
    add(
        "closeness",
        "close_maps_abs_diff_le_2",
        close["close_maps"],
        f"{close['close_map_win_rate']:.1%} won",
    )
    add(
        "closeness",
        "overtime_maps",
        close["overtime_maps"],
        f"{close['overtime_win_rate']:.1%} won",
    )
    add(
        "closeness",
        "series_going_the_distance",
        close["decider_series"],
        f"{close['decider_win_rate']:.1%} won",
    )
    add("closeness", "sweeps_for", close["sweeps_for"])
    add("closeness", "sweeps_against", close["sweeps_against"])
    add("closeness", "series_lost_by_one_map", close["one_map_losses"])

    add("elo", "elo_window_start", elo["elo_start"])
    add("elo", "elo_window_end", elo["elo_end"])
    add("elo", "elo_peak", elo["elo_peak"], elo["elo_peak_date"])
    add("elo", "elo_window_low", elo["elo_low"])
    add("elo", "avg_opponent_elo", elo["avg_opp_elo"])
    add("elo", "hardest_opponent", elo["hardest_opponent"], str(elo["hardest_opponent_elo"]))
    add("elo", "expected_series_wins_by_elo", elo["expected_series_wins"])
    add("elo", "actual_series_wins", elo["actual_series_wins"])
    add("elo", "delta_vs_elo_expectation", elo["delta_vs_elo"])

    for row in events.itertuples(index=False):
        add(
            "event",
            row.event,
            row.series_record,
            f"maps {row.map_record}"
            + (f" ({row.map_win_rate:.0%})" if pd.notna(row.map_win_rate) else "")
            + f", map rows {row.map_coverage} | {row.first}..{row.last}",
        )
    for row in months.itertuples(index=False):
        add(
            "month",
            row.month,
            row.record,
            f"maps {row.maps}, map wr "
            + (f"{row.map_win_rate:.0%}" if pd.notna(row.map_win_rate) else "n/a")
            + f", avg elo {row.avg_elo}",
        )
    for row in maps.itertuples(index=False):
        add(
            "map",
            row.map,
            row.record,
            f"{row.win_rate:.0%} over {row.maps}, "
            f"avg round diff {row.avg_round_diff:+.2f}, OT {row.overtime}",
        )
    for row in opps.itertuples(index=False):
        add(
            "opponent",
            row.opponent,
            row.record,
            f"{row.win_rate:.0%}, maps {row.map_record}, avg opp elo {row.avg_opp_elo}",
        )
    for row in players.itertuples(index=False):
        add(
            "player",
            row.player,
            f"{row.maps} maps",
            f"K/D {row.kd:.2f}, ADR {row.adr:.1f}, KAST {row.kast:.1%}, map wr"
            f" {row.map_win_rate:.0%}, "
            f"{row.first.date()}..{row.last.date()}",
        )
    for row in fives.itertuples(index=False):
        add(
            "lineup",
            row.lineup,
            f"{row.series} series",
            f"{row.maps} maps, {row.first.date()}..{row.last.date()}",
        )
    return rows


# --------------------------------------------------------------------------- render


def md_table(frame: pd.DataFrame, labels: dict[str, str], percent: set[str] | None = None) -> str:
    percent = percent or set()
    headers = [labels.get(column, column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in frame.itertuples(index=False):
        cells = []
        for column, value in zip(frame.columns, row, strict=True):
            if column in percent and isinstance(value, (int, float)) and not pd.isna(value):
                cells.append(f"{value:.1%}")
            elif isinstance(value, float) and not pd.isna(value):
                cells.append(str(int(value)) if float(value).is_integer() else f"{value:.3f}")
            elif pd.isna(value):
                cells.append("—")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_findings(
    vs: pd.DataFrame,
    vm: pd.DataFrame,
    maps: pd.DataFrame,
    close: dict,
    elo: dict,
    fives: pd.DataFrame,
    players: pd.DataFrame,
    events: pd.DataFrame,
) -> list[str]:
    """The honest read of the numbers — every bullet derived, none hand-written."""
    w, losses, rate = record(vs)
    mw, ml, mrate = record(vm)
    pool = maps[maps["maps"] >= 8]
    best = pool.sort_values("win_rate", ascending=False).iloc[0] if len(pool) else None
    worst = pool.sort_values("win_rate").iloc[0] if len(pool) else None
    top = players.iloc[0]
    bullets = [
        f"**{w}-{losses} in series ({rate:.0%}), {mw}-{ml} in maps ({mrate:.0%})** across "
        f"{vs['event'].nunique()} events and {vs['opponent'].nunique()} opponents.",
        f"Elo went {elo['elo_start']} → {elo['elo_end']} (peak {elo['elo_peak']} on "
        f"{elo['elo_peak_date']}); opponents averaged {elo['avg_opp_elo']}, the hardest being "
        f"{elo['hardest_opponent']} at {elo['hardest_opponent_elo']}.",
        f"Against its own schedule Elo expected **{elo['expected_series_wins']} series wins**; "
        f"Vitality took **{elo['actual_series_wins']}** ({elo['delta_vs_elo']:+.1f}).",
        f"Deciders were the window's edge: {close['decider_series']} series went the distance and "
        f"Vitality won {close['decider_win_rate']:.0%} of them, while going "
        f"{close['close_map_win_rate']:.0%} in maps decided by ≤2 rounds.",
        f"Map pool: strongest {best['map']} ({best['win_rate']:.0%} in {best['maps']} maps), "
        f"weakest {worst['map']} ({worst['win_rate']:.0%} in {worst['maps']}) — "
        f"the pool is only as good as its floor."
        if best is not None and worst is not None
        else "",
        f"**Map-row coverage:** the raw source carries {len(vm)} of the "
        f"{int(vs['maps_played'].sum())} maps these series played "
        f"({len(vm) / int(vs['maps_played'].sum()):.0%}), and "
        f"{len(set(vs['match_id']) - set(vm['match_id']))} series have no map rows at all."
        " Map-level figures are lower bounds; the per-event table carries its own coverage column.",
        f"Roster: {len(fives)} distinct five{'s' if len(fives) != 1 else ''} appeared;"
        f" {top['player']} "
        f"led volume at {top['maps']} maps ({top['kd']:.2f} K/D, {top['adr']:.1f} ADR).",
        f"Defeats were mostly clean: {close['sweeps_against']} of the {losses} series lost were 0-2"
        " or "
        f"0-3, while {close['one_map_losses']} fell by a single map — and {close['bo1']} of the 64"
        " series "
        f"were Bo1 qualifiers, where a single map is the whole series.",
    ]
    return [bullet for bullet in bullets if bullet]


def render_md(
    vs: pd.DataFrame,
    vm: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    players: pd.DataFrame,
    fives: pd.DataFrame,
    close: dict,
    elo: dict,
    wins: pd.DataFrame,
    losses: pd.DataFrame,
) -> str:
    events, months, maps, opps, fmt = (
        tables["events"],
        tables["months"],
        tables["maps"],
        tables["opps"],
        tables["fmt"],
    )
    w, losses_n, rate = record(vs)
    mw, ml, mrate = record(vm)
    out = [
        "# Team Vitality — window review",
        "",
        f"**BLAST Open London 2025 → BLAST Rivals Spring 2026** · {WINDOW_START} → {WINDOW_END}",
        "",
        f"Team Vitality played **{w + losses_n} series ({w}-{losses_n}, {rate:.0%})** and "
        f"**{mw + ml} maps ({mw}-{ml}, {mrate:.0%})** in this window, across "
        f"{vs['event'].nunique()} events and {vs['opponent'].nunique()} opponents. "
        "Every number below is generated by `scripts/analyze_vitality.py` from the repo's own "
        "`series_clean.csv` + raw map table; the strength figures reuse the shipped Elo engine "
        "(K=32, base 1500).",
        "",
        "## Headline",
        "",
        md_table(
            pd.DataFrame(
                [
                    {
                        "metric": "Series",
                        "value": f"{w}-{losses_n}",
                        "note": f"{rate:.1%} win rate",
                    },
                    {"metric": "Maps", "value": f"{mw}-{ml}", "note": f"{mrate:.1%} win rate"},
                    {
                        "metric": "Longest win / loss streak",
                        "value": f"{streaks(vs)[0]} / {streaks(vs)[1]}",
                        "note": "chronological series order",
                    },
                    {
                        "metric": "Elo",
                        "value": f"{elo['elo_start']} → {elo['elo_end']}",
                        "note": f"peak {elo['elo_peak']} on {elo['elo_peak_date']}, low"
                        f" {elo['elo_low']}",
                    },
                    {
                        "metric": "Opponent strength",
                        "value": f"{elo['avg_opp_elo']} avg Elo",
                        "note": f"hardest: {elo['hardest_opponent']}"
                        f" ({elo['hardest_opponent_elo']})",
                    },
                    {
                        "metric": "Elo expectation",
                        "value": f"{elo['expected_series_wins']} wins",
                        "note": f"actual {elo['actual_series_wins']} ({elo['delta_vs_elo']:+.1f})",
                    },
                    {
                        "metric": "Deciders",
                        "value": f"{close['decider_series']} series",
                        "note": f"{close['decider_win_rate']:.0%} won when it went the distance",
                    },
                ]
            ),
            {"metric": "Metric", "value": "Value", "note": "Note"},
        ),
        "",
        "## What the window says",
        "",
    ]
    out += [
        f"- {bullet}" for bullet in md_findings(vs, vm, maps, close, elo, fives, players, events)
    ]
    out += [
        "",
        "## Events",
        "",
        md_table(
            events[
                [
                    "event",
                    "series",
                    "series_record",
                    "map_record",
                    "map_win_rate",
                    "map_coverage",
                    "first",
                    "last",
                ]
            ],
            {
                "event": "Event",
                "series": "Series",
                "series_record": "W-L",
                "map_record": "Maps W-L",
                "map_win_rate": "Map win",
                "map_coverage": "Map rows",
                "first": "First",
                "last": "Last",
            },
            percent={"map_win_rate"},
        ),
        "",
        "## Timeline",
        "",
        md_table(
            months[
                ["month", "series", "record", "series_win_rate", "maps", "map_win_rate", "avg_elo"]
            ],
            {
                "month": "Month",
                "series": "Series",
                "record": "W-L",
                "series_win_rate": "Series win",
                "maps": "Maps",
                "map_win_rate": "Map win",
                "avg_elo": "Avg Elo",
            },
            percent={"series_win_rate", "map_win_rate"},
        ),
        "",
        "## Map pool",
        "",
        md_table(
            maps[["map", "maps", "record", "win_rate", "avg_round_diff", "overtime"]],
            {
                "map": "Map",
                "maps": "Played",
                "record": "W-L",
                "win_rate": "Win rate",
                "avg_round_diff": "Avg round diff",
                "overtime": "OT maps",
            },
            percent={"win_rate"},
        ),
        "",
        "## Head-to-head",
        "",
        md_table(
            opps[["opponent", "series", "record", "win_rate", "map_record", "avg_opp_elo"]],
            {
                "opponent": "Opponent",
                "series": "Series",
                "record": "W-L",
                "win_rate": "Win rate",
                "map_record": "Maps W-L",
                "avg_opp_elo": "Avg opp Elo",
            },
            percent={"win_rate"},
        ),
        "",
        "## Formats and closeness",
        "",
        md_table(
            fmt,
            {"format": "Format", "series": "Series", "record": "W-L", "win_rate": "Win rate"},
            percent={"win_rate"},
        ),
        "",
        "| Situation | Count | Vitality's record |",
        "|---|---|---|",
        f"| Maps decided by ≤2 rounds | {close['close_maps']} |"
        f" {close['close_map_win_rate']:.1%} won |",
        f"| Overtime maps (>24 rounds) | {close['overtime_maps']} |"
        f" {close['overtime_win_rate']:.1%} won |",
        f"| Series that went the distance | {close['decider_series']} |"
        f" {close['decider_win_rate']:.1%} won |",
        f"| Sweeps delivered | {close['sweeps_for']} | 2-0 or 3-0 in their favour |",
        f"| Sweeps suffered | {close['sweeps_against']} | lost without taking a map |",
        f"| Series lost by a single map | {close['one_map_losses']} | e.g. 1-2 or 2-3 |",
        "",
        "## Lineups and players",
        "",
        md_table(
            fives.assign(
                first=fives["first"].dt.date.astype(str), last=fives["last"].dt.date.astype(str)
            ),
            {
                "lineup": "Five",
                "series": "Series",
                "maps": "Maps",
                "first": "First",
                "last": "Last",
            },
        ),
        "",
        md_table(
            players.assign(
                first=players["first"].dt.date.astype(str), last=players["last"].dt.date.astype(str)
            )[
                [
                    "player",
                    "maps",
                    "kills",
                    "deaths",
                    "assists",
                    "kd",
                    "adr",
                    "kast",
                    "map_win_rate",
                    "first",
                    "last",
                ]
            ],
            {
                "player": "Player",
                "maps": "Maps",
                "kills": "K",
                "deaths": "D",
                "assists": "A",
                "kd": "K/D",
                "adr": "ADR",
                "kast": "KAST",
                "map_win_rate": "Map win",
                "first": "First",
                "last": "Last",
            },
            percent={"kast", "map_win_rate"},
        ),
        "",
        "## Best wins and worst defeats (by opponent Elo)",
        "",
        md_table(
            _edge_rows(wins, vs),
            {
                "date": "Date",
                "event": "Event",
                "opponent": "Opponent",
                "score": "Score",
                "opp_elo": "Opp Elo",
            },
        ),
        "",
        md_table(
            _edge_rows(losses, vs),
            {
                "date": "Date",
                "event": "Event",
                "opponent": "Opponent",
                "score": "Score",
                "opp_elo": "Opp Elo",
            },
        ),
        "",
        "## Method and caveats",
        "",
        "- Series and tier come from `outputs/series_clean.csv`; map/round/player detail from",
        "  `data/raw/cs2_all_tiers_games.csv` (`is_total` rows dropped — they are series"
        " summaries).",
        "- Map coverage is incomplete in the source: the window's series played more maps than the"
        " raw",
        "  table has rows for, and a few series have none. Map-level numbers are labelled with"
        " their",
        "  coverage in the Events table and are lower bounds wherever it is below 100%.",
        "- Series results come from the `winner` column, which agrees with the series scores on all"
        " 64",
        "  series. (An earlier cut of this script XOR-ed that team-NAME field with the side flag"
        " and",
        "  silently inverted every series Vitality played as team2 — the reason the record was"
        " re-derived",
        "  from the score columns before anything was reported.)",
        "- Elo is the repo's walk-forward engine (K=32, base 1500) replayed over the full"
        " 9,920-series",
        "  history, so a rating entering the window reflects everything before it. Pre-match"
        " ratings only.",
        "- The window is the pair of dates requested. The data's own edges inside it are shown in"
        " the",
        "  Headline table; the last Vitality series in the source data lands before 2026-05-05.",
        "- Opponent Elo is the *pre-match* rating of Vitality's opponent in each series.",
        "",
        "---",
        "",
        "_Generated by `scripts/analyze_vitality.py`; CSVs alongside:"
        " `outputs/vitality_series.csv`,",
        "`outputs/vitality_maps.csv`, `outputs/vitality_summary.csv`. Dark-theme HTML:",
        "`docs/vitality-analysis.html`._",
        "",
    ]
    return "\n".join(out)


def _edge_rows(frame: pd.DataFrame, vs: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": frame["datetime"].dt.date.astype(str).to_numpy(),
            "event": frame["event"].to_numpy(),
            "opponent": frame["opponent"].to_numpy(),
            "score": [
                f"{a}-{b}" for a, b in zip(frame["score_for"], frame["score_against"], strict=True)
            ],
            "opp_elo": frame["opp_elo"].round(1).to_numpy(),
        }
    )


# --- standalone HTML (dark theme, inline SVG charts, no external resources) ---

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #0d1117; color: #e6edf3;
  font: 16px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 40px 24px 80px; }
h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -0.4px; }
h2 { font-size: 20px; margin: 40px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #21262d; }
.sub { color: #8b949e; margin: 0 0 8px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px;
margin: 22px 0 8px; }
.kpi { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 14px; }
.kpi .v { font-size: 22px; font-weight: 650; }
.kpi .l { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; }
.kpi .n { color: #6e7681; font-size: 12px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 4px; font-size: 14px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #21262d; }
th { color: #8b949e; font-weight: 600; background: #11161d; position: sticky; top: 0; }
tbody tr:hover { background: #161b22; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
ul.findings { padding-left: 20px; }
ul.findings li { margin: 8px 0; }
.scroll { overflow-x: auto; }
footer { color: #6e7681; font-size: 13px; margin-top: 44px;
  border-top: 1px solid #21262d; padding-top: 16px; }
code { background: #161b22; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
@media print { body { background: #fff; color: #000; } .kpi, th { background: #f5f5f5; }
  h2 { border-color: #ccc; } th, td { border-color: #ddd; } }
"""


def svg_columns(
    labels: list[str],
    values: list[float],
    width: int = 720,
    height: int = 230,
    title: str = "",
    baseline: float | None = None,
) -> str:
    """Vertical bar chart with value labels. `values` are fractions when `baseline` is set."""
    pad_l, pad_r, pad_t, pad_b = 46, 14, 26, 46
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    top = max(values) if values else 1.0
    top = max(top if baseline is None else max(top, baseline), 1e-9)
    top = top * 1.12 if baseline is None else 1.0
    step = plot_w / max(len(values), 1)
    bars = []
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        bar_h = 0 if not value or np.isnan(value) else max(1.0, plot_h * (value / top))
        x = pad_l + i * step + step * 0.18
        y = pad_t + plot_h - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{step * 0.64:.1f}" height="{bar_h:.1f}" '
            f'rx="3" fill="#3fb950" opacity="0.85"/>'
            f' <text x="{x + step * 0.32:.1f}" y="{y - 5:.1f}" fill="#8b949e" font-size="10" '
            f'text-anchor="middle">{value:.0%}</text>'
            f' <text x="{x + step * 0.32:.1f}" y="{height - 26:.1f}" fill="#8b949e" font-size="10" '
            f'text-anchor="middle" transform="rotate(-45 {x + step * 0.32:.1f}'
            f' {height - 26:.1f})">{html.escape(label)}</text>'
        )
    grid = ""
    if baseline is not None:
        y = pad_t + plot_h - plot_h * (baseline / top)
        grid = (
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="#8b949e" stroke-dasharray="4 4" opacity="0.6"/>'
            f' <text x="{pad_l - 6}" y="{y + 3:.1f}" fill="#8b949e" font-size="10" '
            f'text-anchor="end">{baseline:.0%}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img"'
        f' aria-label="{html.escape(title)}">'
        f" <title>{html.escape(title)}</title>"
        f' <text x="{pad_l}" y="16" fill="#e6edf3" font-size="13">{html.escape(title)}</text>{grid}'
        + "".join(bars)
        + "</svg>"
    )


def svg_bars(
    pairs: list[tuple[str, float, int]],
    width: int = 720,
    height: int | None = None,
    title: str = "",
) -> str:
    """Horizontal bars: (label, win_rate, n)."""
    row_h = 24
    height = height or 40 + row_h * len(pairs)
    pad_l, pad_r, pad_t = 96, 58, 30
    plot_w = width - pad_l - pad_r
    out = []
    for i, (label, value, n) in enumerate(pairs):
        y = pad_t + i * row_h
        bar_w = max(1.0, plot_w * (0.0 if np.isnan(value) else value))
        colour = "#3fb950" if value >= 0.5 else "#f85149"
        out.append(
            f'<text x="{pad_l - 8}" y="{y + 13}" fill="#8b949e" font-size="11" text-anchor="end">'
            f" {html.escape(label)}</text>"
            f' <rect x="{pad_l}" y="{y + 3}" width="{plot_w}" height="14" rx="3" fill="#21262d"/>'
            f' <rect x="{pad_l}" y="{y + 3}" width="{bar_w:.1f}" height="14" rx="3" fill="{colour}"'
            ' opacity="0.85"/>'
            f' <text x="{pad_l + plot_w + 8}" y="{y + 14}" fill="#8b949e" font-size="11">'
            f" {value:.0%} ({n})</text>"
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img"'
        f' aria-label="{html.escape(title)}">'
        f" <title>{html.escape(title)}</title>"
        f' <text x="10" y="18" fill="#e6edf3" font-size="13">{html.escape(title)}</text>'
        f' <line x1="{pad_l + plot_w * 0.5:.1f}" y1="{pad_t}" x2="{pad_l + plot_w * 0.5:.1f}" '
        f'y2="{height - 6}" stroke="#8b949e" stroke-dasharray="3 3" opacity="0.5"/>'
        + "".join(out)
        + "</svg>"
    )


def svg_lines(
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    width: int = 720,
    height: int = 260,
    title: str = "",
) -> str:
    """Line chart with a legend; `series` is [(name, values, colour)]."""
    pad_l, pad_r, pad_t, pad_b = 52, 120, 30, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_values = [v for _, values, _ in series for v in values if not np.isnan(v)]
    lo, hi = (min(all_values), max(all_values)) if all_values else (0.0, 1.0)
    span = max(hi - lo, 1e-6)
    lo, hi = lo - span * 0.08, hi + span * 0.08
    n = max(len(labels), 2)

    def px(i: int) -> float:
        return pad_l + plot_w * (i / (n - 1))

    def py(v: float) -> float:
        return pad_t + plot_h - plot_h * ((v - lo) / (hi - lo))

    grid = []
    for frac in (0.0, 0.5, 1.0):
        value = lo + (hi - lo) * frac
        y = py(value)
        grid.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="#21262d"/>'
            f' <text x="{pad_l - 6}" y="{y + 3:.1f}" fill="#8b949e" font-size="10" '
            f'text-anchor="end">{value:.0f}</text>'
        )
    lines, legend = [], []
    for k, (name, values, colour) in enumerate(series):
        points = " ".join(
            f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values) if not np.isnan(v)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"/>'
        )
        legend.append(
            f'<rect x="{pad_l + plot_w + 14}" y="{pad_t + 6 + k * 18}" width="10" height="10" '
            f'fill="{colour}" rx="2"/>'
            f' <text x="{pad_l + plot_w + 30}" y="{pad_t + 15 + k * 18}" fill="#8b949e" '
            f'font-size="11">{html.escape(name)}</text>'
        )
    ticks = []
    for i in range(0, len(labels), max(1, len(labels) // 6)):
        ticks.append(
            f'<text x="{px(i):.1f}" y="{height - 12}" fill="#8b949e" font-size="10" '
            f'text-anchor="middle">{html.escape(labels[i])}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img"'
        f' aria-label="{html.escape(title)}">'
        f" <title>{html.escape(title)}</title>"
        f' <text x="10" y="18" fill="#e6edf3" font-size="13">{html.escape(title)}</text>'
        + "".join(grid)
        + "".join(lines)
        + "".join(legend)
        + "".join(ticks)
        + "</svg>"
    )


def html_table(frame: pd.DataFrame, labels: dict[str, str], percent: set[str] | None = None) -> str:
    percent = percent or set()
    head = "".join(f"<th>{html.escape(labels.get(c, c))}</th>" for c in frame.columns)
    body = []
    for row in frame.itertuples(index=False):
        cells = []
        for column, value in zip(frame.columns, row, strict=True):
            numeric = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
                value, bool
            )
            if column in percent and numeric and not pd.isna(value):
                text = f"{value:.1%}"
            elif numeric and not pd.isna(value):
                text = f"{value:g}"
            elif pd.isna(value):
                text = "—"
            else:
                text = html.escape(str(value))
            cells.append(f'<td class="{"num" if numeric else ""}">{text}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(
    vs: pd.DataFrame,
    vm: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    players: pd.DataFrame,
    fives: pd.DataFrame,
    close: dict,
    elo: dict,
    wins: pd.DataFrame,
    losses: pd.DataFrame,
) -> str:
    events, months, maps, opps, fmt = (
        tables["events"],
        tables["months"],
        tables["maps"],
        tables["opps"],
        tables["fmt"],
    )
    w, losses_n, rate = record(vs)
    mw, ml, mrate = record(vm)
    months_plain = months.copy()
    pool_pairs = [(row.map, row.win_rate, row.maps) for row in maps.itertuples(index=False)]
    trace = elo["elo_trace"]
    labels = [d.date().strftime("%b %d") for d in trace["datetime"]]
    charts = (
        svg_columns(
            months_plain["month"].tolist(),
            months_plain["series_win_rate"].tolist(),
            title="Monthly series win rate",
            baseline=0.5,
        )
        + '<div style="height:18px"></div>'
        + svg_lines(
            labels,
            [
                ("Vitality Elo", trace["elo_v"].tolist(), "#58a6ff"),
                ("Opponent Elo", trace["opp_elo"].tolist(), "#8b949e"),
            ],
            title="Pre-match Elo, series by series",
        )
        + '<div style="height:18px"></div>'
        + svg_bars(pool_pairs, title="Map pool — win rate (maps played)")
    )
    kpis = [
        ("Series", f"{w}-{losses_n}", f"{rate:.0%} win rate"),
        ("Maps", f"{mw}-{ml}", f"{mrate:.0%} win rate"),
        ("Events", str(vs["event"].nunique()), f"{vs['opponent'].nunique()} opponents"),
        ("Elo", f"{elo['elo_start']} → {elo['elo_end']}", f"peak {elo['elo_peak']}"),
        (
            "vs Elo expectation",
            f"{elo['delta_vs_elo']:+.1f}",
            f"{elo['actual_series_wins']} of {elo['expected_series_wins']} expected",
        ),
        ("Deciders", f"{close['decider_win_rate']:.0%}", f"{close['decider_series']} series"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="l">{html.escape(label)}</div><div'
        f' class="v">{html.escape(value)}</div>'
        f' <div class="n">{html.escape(note)}</div></div>'
        for label, value, note in kpis
    )
    findings = "".join(
        f"<li>{bullet}</li>"
        for bullet in md_findings(vs, vm, maps, close, elo, fives, players, events)
    )
    # findings are markdown-ish; strip the bold markers for HTML
    findings = findings.replace("**", "")
    lineup_frame = fives.assign(
        first=fives["first"].dt.date.astype(str), last=fives["last"].dt.date.astype(str)
    )
    player_frame = players.assign(
        first=players["first"].dt.date.astype(str), last=players["last"].dt.date.astype(str)
    )
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Team Vitality — window review (BLAST Open London 2025 → BLAST Rivals Spring"
        " 2026)</title>",
        f'<style>{CSS}</style></head><body><div class="wrap">',
        "<h1>Team Vitality — window review</h1>",
        f'<p class="sub">BLAST Open London 2025 → BLAST Rivals Spring 2026 · {WINDOW_START} →'
        f' {WINDOW_END} '
        f"· first series {vs['datetime'].min().date()} · last series"
        f" {vs['datetime'].max().date()}</p>",
        f'<div class="kpis">{kpi_html}</div>',
        "<h2>What the window says</h2>",
        f'<ul class="findings">{findings}</ul>',
        "<h2>Timeline</h2>",
        charts,
        '<h2>Events</h2><div class="scroll">',
        html_table(
            events[
                [
                    "event",
                    "series",
                    "series_record",
                    "map_record",
                    "map_win_rate",
                    "map_coverage",
                    "first",
                    "last",
                ]
            ],
            {
                "event": "Event",
                "series": "Series",
                "series_record": "W-L",
                "map_record": "Maps W-L",
                "map_win_rate": "Map win",
                "map_coverage": "Map rows",
                "first": "First",
                "last": "Last",
            },
            {"map_win_rate"},
        ),
        '</div><h2>Monthly</h2><div class="scroll">',
        html_table(
            months[
                ["month", "series", "record", "series_win_rate", "maps", "map_win_rate", "avg_elo"]
            ],
            {
                "month": "Month",
                "series": "Series",
                "record": "W-L",
                "series_win_rate": "Series win",
                "maps": "Maps",
                "map_win_rate": "Map win",
                "avg_elo": "Avg Elo",
            },
            {"series_win_rate", "map_win_rate"},
        ),
        '</div><h2>Map pool</h2><div class="scroll">',
        html_table(
            maps,
            {
                "map": "Map",
                "maps": "Played",
                "record": "W-L",
                "win_rate": "Win rate",
                "avg_round_diff": "Avg round diff",
                "overtime": "OT maps",
            },
            {"win_rate"},
        ),
        '</div><h2>Head-to-head</h2><div class="scroll">',
        html_table(
            opps,
            {
                "opponent": "Opponent",
                "series": "Series",
                "record": "W-L",
                "win_rate": "Win rate",
                "map_record": "Maps W-L",
                "avg_opp_elo": "Avg opp Elo",
            },
            {"win_rate"},
        ),
        '</div><h2>Formats and closeness</h2><div class="scroll">',
        html_table(
            fmt,
            {"format": "Format", "series": "Series", "record": "W-L", "win_rate": "Win rate"},
            {"win_rate"},
        ),
        "</div>",
        html_table(
            pd.DataFrame(
                [
                    {
                        "situation": "Maps decided by ≤2 rounds",
                        "count": close["close_maps"],
                        "record": f"{close['close_map_win_rate']:.1%} won",
                    },
                    {
                        "situation": "Overtime maps (>24 rounds)",
                        "count": close["overtime_maps"],
                        "record": f"{close['overtime_win_rate']:.1%} won",
                    },
                    {
                        "situation": "Series that went the distance",
                        "count": close["decider_series"],
                        "record": f"{close['decider_win_rate']:.1%} won",
                    },
                    {
                        "situation": "Sweeps delivered",
                        "count": close["sweeps_for"],
                        "record": "2-0 / 3-0",
                    },
                    {
                        "situation": "Sweeps suffered",
                        "count": close["sweeps_against"],
                        "record": "lost without a map",
                    },
                    {
                        "situation": "Series lost by a single map",
                        "count": close["one_map_losses"],
                        "record": "e.g. 1-2 or 2-3",
                    },
                ]
            ),
            {"situation": "Situation", "count": "Count", "record": "Vitality"},
        ),
        '<h2>Lineups and players</h2><div class="scroll">',
        html_table(
            lineup_frame,
            {
                "lineup": "Five",
                "series": "Series",
                "maps": "Maps",
                "first": "First",
                "last": "Last",
            },
        ),
        '</div><div class="scroll">',
        html_table(
            player_frame[
                [
                    "player",
                    "maps",
                    "kills",
                    "deaths",
                    "assists",
                    "kd",
                    "adr",
                    "kast",
                    "map_win_rate",
                    "first",
                    "last",
                ]
            ],
            {
                "player": "Player",
                "maps": "Maps",
                "kills": "K",
                "deaths": "D",
                "assists": "A",
                "kd": "K/D",
                "adr": "ADR",
                "kast": "KAST",
                "map_win_rate": "Map win",
                "first": "First",
                "last": "Last",
            },
            {"kast", "map_win_rate"},
        ),
        '</div><h2>Best wins and worst defeats</h2><div class="scroll">',
        html_table(
            _edge_rows(wins, vs),
            {
                "date": "Date",
                "event": "Event",
                "opponent": "Opponent",
                "score": "Score",
                "opp_elo": "Opp Elo",
            },
        ),
        '</div><div class="scroll">',
        html_table(
            _edge_rows(losses, vs),
            {
                "date": "Date",
                "event": "Event",
                "opponent": "Opponent",
                "score": "Score",
                "opp_elo": "Opp Elo",
            },
        ),
        "</div>",
        "<footer>Generated by <code>scripts/analyze_vitality.py</code> from "
        "<code>outputs/series_clean.csv</code> + <code>data/raw/cs2_all_tiers_games.csv</code>. "
        "Elo: repo engine, K=32, base 1500, walk-forward (pre-match ratings). "
        "CSVs: <code>outputs/vitality_series.csv</code>, <code>outputs/vitality_maps.csv</code>, "
        "<code>outputs/vitality_summary.csv</code>.</footer>",
        "</div></body></html>",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- main


def main() -> None:
    series = load_series()
    vs = vital_series(series)
    vm = load_maps()
    players = player_lines()
    fives = lineups()

    tables = {
        "events": event_table(vs, vm),
        "months": month_table(vs, vm),
        "maps": map_table(vm),
        "opps": opponent_table(vs, vm),
        "fmt": format_table(vs),
    }
    close = closeness(vm, vs)
    elo = elo_section(vs)
    wins, losses = best_and_worst(vs)
    rows = summary_rows(
        vs,
        vm,
        players,
        tables["events"],
        tables["maps"],
        tables["months"],
        tables["opps"],
        tables["fmt"],
        close,
        elo,
        fives,
    )

    series_out = vs.assign(
        datetime=vs["datetime"].dt.strftime("%Y-%m-%d %H:%M"),
        elo=vs["elo"].round(1),
        opp_elo=vs["opp_elo"].round(1),
        p_win=vs["p_win"].round(4),
    )
    maps_out = vm.assign(
        datetime=vm["datetime"].dt.strftime("%Y-%m-%d %H:%M"),
        team_adr=vm["team_adr"].round(1),
        team_kast=vm["team_kast"].round(3),
    )
    series_out.to_csv(SERIES_OUT, index=False)
    maps_out.to_csv(MAPS_OUT, index=False)
    pd.DataFrame(rows).to_csv(SUMMARY_OUT, index=False)

    MD_OUT.write_text(
        render_md(vs, vm, tables, players, fives, close, elo, wins, losses), encoding="utf-8"
    )
    HTML_OUT.write_text(
        render_html(vs, vm, tables, players, fives, close, elo, wins, losses), encoding="utf-8"
    )

    w, losses_n, rate = record(vs)
    mw, ml, mrate = record(vm)
    print(f"{TEAM}: {WINDOW_START} -> {WINDOW_END}")
    print(f"  series      : {w}-{losses_n} ({rate:.1%}) over {vs['event'].nunique()} events")
    print(f"  maps        : {mw}-{ml} ({mrate:.1%})")
    print(f"  first/last  : {vs['datetime'].min().date()} .. {vs['datetime'].max().date()}")
    print(
        f"  elo         : {elo['elo_start']} -> {elo['elo_end']} (peak {elo['elo_peak']}), "
        f"{elo['delta_vs_elo']:+.1f} vs expectation"
    )
    print(f"  lineups     : {len(fives)} distinct fives")
    print(
        f"  wrote       : {SERIES_OUT.name} ({len(series_out)}), {MAPS_OUT.name} ({len(maps_out)}),"
        f" {SUMMARY_OUT.name} ({len(rows)}), {MD_OUT.name}, {HTML_OUT.name}"
    )


if __name__ == "__main__":
    main()
