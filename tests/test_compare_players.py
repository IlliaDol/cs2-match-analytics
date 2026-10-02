"""Contracts for scripts/compare_players.py — synthetic frames, no real data.

The orientation regression matters most here: `team1_win` is a side-relative flag,
so a player on team2 keeps their stats and their "won" flag is the flag's negation.
Getting that wrong silently swaps two players' win rates (the same bug class the
Vitality report caught).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_players as cp  # noqa: E402

A = "Alpha (A), XX"
B = "Bravo (B), YY"
C = "Charlie (C), ZZ"


def map_row(game_id, team1, team2, team1_win, s1, s2, players, stats=None, map_name="Dust2",
            event="Test Cup", when="2025-09-01", match_id=1):
    """One synthetic map row with five player slots per side."""
    row = {
        "match_id": match_id, "game_id": game_id, "datetime": pd.Timestamp(when, tz="UTC"),
        "tournament": event, "map_name": map_name, "team1": team1, "team2": team2,
        "team1_win": team1_win, "score1_game": s1, "score2_game": s2,
        "is_total": False,
    }
    for side, names in (("team1", players[0]), ("team2", players[1])):
        for i in range(1, 6):
            row[f"{side}_player{i}"] = names[i - 1]
            for stat in ("kills", "deaths", "assists", "adr", "kast"):
                row[f"{side}_player{i}_{stat}"] = (stats or {}).get((side, i - 1, stat), 0)
    return row


def build(rows):
    return cp.player_frame(pd.DataFrame(rows))


def test_team2_player_keeps_stats_and_gets_the_inverted_win_flag():
    """Regression: team2 stats must not be mirrored, and `won` is the negated flag."""
    rows = [
        map_row(10, "Red", "Blue", team1_win=False, s1=7, s2=13,
                players=([C, "x1", "x2", "x3", "x4"], [A, "y1", "y2", "y3", "y4"]),
                stats={("team2", 0, "kills"): 25, ("team2", 0, "deaths"): 10,
                       ("team2", 0, "adr"): 95.5}),
    ]
    long = build(rows)
    a = long[long["player"] == A].iloc[0]
    assert a["kills"] == 25 and a["deaths"] == 10 and a["adr"] == 95.5   # not mirrored
    assert bool(a["won"]) is True                                        # team1 lost => A won
    assert a["opponent"] == "Red" and a["team"] == "Blue"


def test_team1_player_win_flag_is_the_flag_itself():
    rows = [map_row(11, "Red", "Blue", team1_win=True, s1=13, s2=9,
                    players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))]
    long = build(rows)
    assert bool(long[long["player"] == A].iloc[0]["won"]) is True
    assert bool(long[long["player"] == B].iloc[0]["won"]) is False


def test_five_slots_per_side_are_all_present():
    rows = [map_row(12, "Red", "Blue", True, 13, 5,
                    players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))]
    long = build(rows)
    assert len(long) == 10
    assert long["player"].nunique() == 10


def test_empty_slots_are_dropped():
    rows = [
        map_row(
            13,
            "Red",
            "Blue",
            True,
            13,
            5,
            players=([A, None, None, None, None], [B, "y1", "y2", "y3", "y4"]),
        )
    ]
    long = build(rows)
    assert long["player"].notna().all()
    assert set(long["player"]) == {A, B, "y1", "y2", "y3", "y4"}   # the 4 empty slots are gone


def test_kast_is_rescaled_to_a_fraction():
    rows = [map_row(14, "Red", "Blue", True, 13, 5,
                    players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]),
                    stats={("team1", 0, "kast"): 77.44})]
    long = build(rows)
    assert long[long["player"] == A].iloc[0]["kast"] == pytest.approx(0.7744)


def test_aggregates_use_real_rounds_for_kpr():
    rows = [
        map_row(20, "Red", "Blue", True, 13, 11,
                players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]),
                stats={("team1", 0, "kills"): 20, ("team1", 0, "deaths"): 10,
                       ("team1", 0, "adr"): 80.0, ("team1", 0, "kast"): 70.0}),
    ]
    agg = cp.aggregates(build(rows)[lambda f: f["player"] == A])
    assert agg["maps"] == 1
    assert agg["rounds"] == 24                       # 13 + 11, the map's real rounds
    assert agg["kpr"] == pytest.approx(20 / 24)      # a rate, not a per-map mean
    assert agg["kd"] == pytest.approx(2.0)
    assert agg["kast"] == pytest.approx(0.70)


def test_head_to_head_counts_only_shared_maps():
    rows = [
        map_row(30, "Red", "Blue", True, 13, 5,
                players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]),
                stats={("team1", 0, "adr"): 100.0, ("team2", 0, "adr"): 50.0}),
        map_row(31, "Red", "Green", True, 13, 5,
                players=([A, "x1", "x2", "x3", "x4"], [C, "y1", "y2", "y3", "y4"]),
                stats={("team1", 0, "adr"): 90.0}),
    ]
    long = build(rows)
    h2h = cp.head_to_head(long[long["player"] == A], long[long["player"] == B])
    assert h2h["maps"] == 1                      # the Green map is not a duel
    assert h2h["a_map_wins"] == 1 and h2h["b_map_wins"] == 0
    assert h2h["a_adr"] == pytest.approx(100.0) and h2h["b_adr"] == pytest.approx(50.0)
    assert h2h["same_team_share"] == 0.0


def test_head_to_head_flags_shared_team_maps():
    rows = [map_row(40, "Red", "Red", True, 13, 5,
                    players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))]
    long = build(rows)
    h2h = cp.head_to_head(long[long["player"] == A], long[long["player"] == B])
    assert h2h["same_team_share"] == 1.0         # honest: not a duel at all


def test_resolve_accepts_exact_and_unique_case_insensitive():
    long = build([map_row(50, "Red", "Blue", True, 13, 5,
                          players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))])
    assert cp.resolve(A, long) == A
    assert cp.resolve(A.lower(), long) == A


def test_resolve_refuses_ambiguous_and_unknown_names():
    long = build([
        map_row(60, "Red", "Blue", True, 13, 5,
                players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"])),
        map_row(61, "Red", "Blue", True, 13, 5,
                players=([C, "x1", "x2", "x3", "x4"],
                         ["Alpha (Other), QQ", "y1", "y2", "y3", "y4"])),
    ])
    # a fragment hitting two players must refuse, not guess...
    with pytest.raises(SystemExit, match="matches 2 players"):
        cp.resolve("alpha", long)
    # ...and the same error must say how to fix it
    with pytest.raises(SystemExit, match="be more specific"):
        cp.resolve("Alpha (", long)
    # a fragment that is unique in the data is accepted (convenience, no guessing)
    assert cp.resolve("Bra", long) == B
    with pytest.raises(SystemExit, match="no player matches"):
        cp.resolve("Nobody At All", long)


def test_resolve_accepts_a_nickname_on_its_own():
    """What people actually type: the nickname, in any case, without the real name."""
    long = build([map_row(70, "Red", "Blue", True, 13, 5,
                          players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))])
    assert cp.resolve("Alpha", long) == A
    assert cp.resolve("alpha", long) == A
    assert cp.resolve("bravo", long) == B
    # a real name without the nickname resolves too
    assert cp.resolve("(A), XX", long) == A


def test_resolve_ignores_surrounding_whitespace():
    long = build([map_row(71, "Red", "Blue", True, 13, 5,
                          players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))])
    assert cp.resolve("  Bravo (B), YY  ", long) == B
    assert cp.resolve("\tBravo\t", long) == B


def test_shared_nickname_error_names_both_players():
    """A nickname two people share must refuse and say WHICH two — not list every
    substring hit, which is what made 'NiKo' look like 12 unrelated players."""
    long = build([
        map_row(72, "Red", "Blue", True, 13, 5,
                players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"])),
        map_row(73, "Red", "Blue", True, 13, 5,
                players=([C, "x1", "x2", "x3", "x4"],
                         ["Alpha (Other), QQ", "y1", "y2", "y3", "y4"])),
    ])
    with pytest.raises(SystemExit) as exc:
        cp.resolve("alpha", long)
    message = str(exc.value)
    assert A in message and "Alpha (Other), QQ" in message
    assert "x1" not in message          # the irrelevant substring hits are gone


# ------------------------------------------------------- curated native names (sheet)

SHEET_COLUMNS = ["country", "language", "nickname", "dataset_string", "latin_name_in_data",
                 "maps", "native_name", "liquipedia_romanized", "liquipedia_country",
                 "corrected_latin", "note"]


def _sheet(tmp_path, rows):
    """Write a review sheet from dicts — csv.writer produces the quoting the real file has."""
    path = tmp_path / "player_names.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SHEET_COLUMNS})
    return path


def test_curated_sheet_gives_native_and_corrected_aliases(tmp_path):
    """A Cyrillic or corrected name typed by hand resolves to the dataset's string."""
    long = build([map_row(80, "Red", "Blue", True, 13, 5,
                          players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))])
    sheet = _sheet(tmp_path, [
        {"country": "XX", "language": "xx", "nickname": "Alpha", "dataset_string": A,
         "latin_name_in_data": "A", "maps": "5", "native_name": "Альфа А",
         "corrected_latin": "Alfa A"},
        {"country": "YY", "language": "yy", "nickname": "Bravo", "dataset_string": B,
         "latin_name_in_data": "B", "maps": "5", "native_name": "Браво Б"},
    ])
    aliases = cp.load_name_aliases(sheet)
    assert aliases[A] == {"Альфа А", "Alfa A"}
    assert aliases[B] == {"Браво Б"}
    assert cp.resolve("Альфа А", long, aliases) == A
    assert cp.resolve("Альфа", long, aliases) == A                         # a fragment
    assert cp.resolve("Alfa A", long, aliases) == A                        # corrected Latin
    assert cp.resolve("  Браво  ", long, aliases) == B


def test_written_form_with_quoted_nickname_still_matches(tmp_path):
    """'Данил «donk» Крышковец' — the way a name is written — must resolve."""
    long = build([map_row(81, "Red", "Blue", True, 13, 5,
                          players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))])
    sheet = _sheet(tmp_path, [
        {"country": "RU", "language": "ru", "nickname": "Alpha", "dataset_string": A,
         "latin_name_in_data": "A", "maps": "5", "native_name": "Данил Крышковец",
         "liquipedia_romanized": "Danil Kryshkovets", "liquipedia_country": "RU"},
    ])
    aliases = cp.load_name_aliases(sheet)
    assert cp.resolve("Данил «donk» Крышковец", long, aliases) == A
    assert cp.resolve("Данил Крышковец", long, aliases) == A


def test_patronymic_in_the_sheet_does_not_block_a_two_part_name(tmp_path):
    """Liquipedia often records 'Дмитрий Эдуардович Соколов'; typing two of the three
    parts must still find the player."""
    long = build([map_row(82, "Red", "Blue", True, 13, 5,
                          players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))])
    sheet = _sheet(tmp_path, [
        {"country": "RU", "language": "ru", "nickname": "Alpha", "dataset_string": A,
         "latin_name_in_data": "A", "maps": "5", "native_name": "Дмитрий Эдуардович Соколов"},
    ])
    aliases = cp.load_name_aliases(sheet)
    assert cp.resolve("Дмитрий Соколов", long, aliases) == A
    assert cp.resolve("Соколов", long, aliases) == A


def test_absent_or_empty_sheet_changes_nothing(tmp_path):
    """The sheet is optional: without it the resolver behaves exactly as before."""
    long = build([map_row(83, "Red", "Blue", True, 13, 5,
                          players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))])
    assert cp.load_name_aliases(tmp_path / "does-not-exist.csv") == {}
    blank = _sheet(tmp_path, [                        # row present but not filled in
        {"country": "RU", "language": "ru", "nickname": "Alpha", "dataset_string": A,
         "latin_name_in_data": "A", "maps": "5"},
    ])
    assert cp.load_name_aliases(blank) == {}
    assert cp.resolve("Bravo", long, {}) == B
    with pytest.raises(SystemExit, match="no player matches"):
        cp.resolve("Браво", long, {})


def test_corrected_latin_overrides_the_displayed_name(tmp_path):
    """Once corrected_latin is filled in, the output shows THAT spelling — the point of the
    sheet. Without it the dataset's own spelling is used, exactly as before."""
    long = build([map_row(90, "Red", "Blue", True, 13, 5,
                          players=([A, "x1", "x2", "x3", "x4"], [B, "y1", "y2", "y3", "y4"]))])
    sheet = _sheet(tmp_path, [
        {"country": "XX", "nickname": "Alpha", "dataset_string": A,
         "latin_name_in_data": "A", "corrected_latin": "Alfa Corrected"},
        {"country": "YY", "nickname": "Bravo", "dataset_string": B,
         "latin_name_in_data": "B"},                       # only matched, not corrected
    ])
    corrected = cp._load_corrected(sheet)
    assert corrected == {cp.key_of(A): "Alfa Corrected"}     # B has no correction
    assert cp.label_for(long, cp.key_of(A), corrected) == "Alfa Corrected"
    assert cp.label_for(long, cp.key_of(B), corrected) == B
    assert cp.label_for(long, cp.key_of(A), {}) == A         # no sheet -> unchanged


def test_corrected_sheet_absent_or_empty_yields_nothing(tmp_path):
    assert cp._load_corrected(tmp_path / "does-not-exist.csv") == {}
    blank = _sheet(tmp_path, [
        {"country": "XX", "nickname": "Alpha", "dataset_string": A, "corrected_latin": "  "},
    ])
    assert cp._load_corrected(blank) == {}                   # whitespace is not a spelling


# --------------------------------------------------------------------------- filters


def _filter_fixture() -> pd.DataFrame:
    """Three maps across two events and two dates, for the filter tests."""
    return pd.DataFrame([
        map_row(70, "Red", "Blue", True, 13, 5, players=([A, "x1", "x2", "x3", "x4"],
                                                        [B, "y1", "y2", "y3", "y4"]),
                map_name="Mirage", event="IEM Katowice 2026", when="2026-02-01", match_id=7),
        map_row(71, "Red", "Blue", False, 9, 13, players=([A, "x1", "x2", "x3", "x4"],
                                                         [B, "y1", "y2", "y3", "y4"]),
                map_name="Dust2", event="IEM Katowice 2026", when="2026-02-02", match_id=7),
        map_row(72, "Red", "Green", True, 13, 11, players=([A, "x1", "x2", "x3", "x4"],
                                                          [C, "y1", "y2", "y3", "y4"]),
                map_name="Nuke", event="BLAST Rivals 2026", when="2026-04-30", match_id=8),
    ])


def test_load_window_filters_by_map(monkeypatch):
    monkeypatch.setattr(cp, "load_map_rows", lambda path=None: _filter_fixture())
    assert cp.load_window(maps=["Mirage"])["game_id"].tolist() == [70]
    assert cp.load_window(maps=["mirage", "DUST2"])["game_id"].tolist() == [70, 71]


def test_load_window_filters_by_event_substring(monkeypatch):
    monkeypatch.setattr(cp, "load_map_rows", lambda path=None: _filter_fixture())
    assert cp.load_window(events=["Katowice"])["game_id"].tolist() == [70, 71]
    assert cp.load_window(events=["rivals"])["game_id"].tolist() == [72]
    assert len(cp.load_window(events=["BLAST", "Katowice"])) == 3


def test_load_window_filters_by_tier_and_dates(monkeypatch):
    monkeypatch.setattr(cp, "load_map_rows", lambda path=None: _filter_fixture())
    monkeypatch.setattr(cp, "tier_map", lambda: {"7": "1", "8": "3"})
    assert cp.load_window(tiers=[1])["game_id"].tolist() == [70, 71]
    assert cp.load_window(tiers=[1, 3])["game_id"].tolist() == [70, 71, 72]
    assert cp.load_window(start="2026-04-01")["game_id"].tolist() == [72]
    assert cp.load_window(end="2026-02-01")["game_id"].tolist() == [70]


def test_load_window_combines_filters(monkeypatch):
    monkeypatch.setattr(cp, "load_map_rows", lambda path=None: _filter_fixture())
    monkeypatch.setattr(cp, "tier_map", lambda: {"7": "1", "8": "3"})
    both = cp.load_window(start="2026-01-01", end="2026-03-01", maps=["Dust2"], tiers=[1])
    assert both["game_id"].tolist() == [71]


def test_interactive_pick_cancels_cleanly_on_empty_input(monkeypatch):
    monkeypatch.setattr(cp, "load_map_rows", lambda path=None: _filter_fixture())
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    args = argparse.Namespace(a=None, b=None, start=None, end=None, maps=[], events=[], tiers=[])
    assert cp.interactive_pick(args) is None          # no exception, no traceback


def test_interactive_pick_collects_players_and_filters(monkeypatch):
    """Drives the real prompts: search 'Alpha', take #1, then filter to Mirage."""
    monkeypatch.setattr(cp, "load_map_rows", lambda path=None: _filter_fixture())
    answers = iter(["Alpha", "1", "Bravo", "1", "Mirage", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    args = argparse.Namespace(a=None, b=None, start=None, end=None, maps=[], events=[], tiers=[])
    picked = cp.interactive_pick(args)
    assert picked is not None
    assert picked.a == A and picked.b == B
    assert picked.maps == ["Mirage"]


def test_md_table_renders_percentages_and_hides_missing():
    frame = pd.DataFrame([{"map": "Dust2", "maps": 12, "kd": 1.5, "adr": float("nan"),
                           "kast": 0.774}])
    text = cp.md_table(frame, {"map": "Map", "maps": "Maps", "kd": "K/D", "adr": "ADR",
                               "kast": "KAST"}, percent={"kast"})
    assert "| Map | Maps | K/D | ADR | KAST |" in text
    assert "77.4%" in text and "—" in text and "1.50" in text
