"""Contracts for the two CS:GO adapters — synthetic frames, no downloads.

The regression that matters most here is the join in `adapt_csgo_history`. Its two files
are keyed differently: `results.csv` is keyed by map name (`_map`), while a player's
per-map numbers live in `m1_*` / `m2_*` / `m3_*`, which line up with `map_1` / `map_2` /
`map_3` — *not* with the order of the result rows. Reading the wrong block would silently
attach one map's kills to another map, which is exactly the kind of error that produces a
confident, wrong report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import adapt_csgo_dataset as ds  # noqa: E402
import adapt_csgo_history as hp  # noqa: E402

# --------------------------------------------------------------- the join (history)

MAPS = pd.Series({"map_1": "Overpass", "map_2": "Nuke", "map_3": "Inferno"})


def test_map_index_follows_the_name_not_the_row_order():
    """`Nuke` is map_2 here — a mis-join would call it map_1."""
    assert hp.map_index(MAPS, "Overpass") == 1
    assert hp.map_index(MAPS, "Nuke") == 2
    assert hp.map_index(MAPS, "Inferno") == 3


def test_map_index_returns_nothing_for_a_map_that_was_not_played():
    assert hp.map_index(MAPS, "Mirage") is None


def _history_frames():
    """One match, two maps listed in the *reverse* order of map_1/map_2/map_3."""
    players = []
    for index, team in enumerate(["Red", "Red", "Red", "Red", "Red",
                                  "Blue", "Blue", "Blue", "Blue", "Blue"]):
        row = {"date": "2018-05-01", "player_name": f"p{index}", "team": team,
               "opponent": "other", "country": "Ukraine", "match_id": 7,
               "event_name": "Test Cup 2018", "best_of": 3,
               "map_1": "Overpass", "map_2": "Nuke", "map_3": "Inferno"}
        for position, kills in ((1, 11), (2, 22), (3, 33)):
            row.update({f"m{position}_kills": kills, f"m{position}_deaths": 10,
                        f"m{position}_assists": 4, f"m{position}_adr": 70.5,
                        f"m{position}_kast": 71.5})
        players.append(row)
    results = pd.DataFrame([                                # Inferno (map_3) listed first
        {"match_id": 7, "map_name": "Inferno", "team_1": "Red", "team_2": "Blue",
         "result_1": 16, "result_2": 12, "map_winner": 1},
        {"match_id": 7, "map_name": "Overpass", "team_1": "Red", "team_2": "Blue",
         "result_1": 9, "result_2": 16, "map_winner": 2},
    ])
    return pd.DataFrame(players), results


def test_one_map_row_per_result_with_the_matching_m_block():
    players, results = _history_frames()
    frame = hp.build(players, results)
    assert len(frame) == 2
    inferno = frame[frame["map_name"] == "Inferno"].iloc[0]
    # Inferno is map_3, so its numbers must be the m3 block (33 kills), never m1 (11)
    assert inferno["team1_player1_kills"] == 33
    assert inferno["team1_player1_deaths"] == 10
    assert inferno["team1_player1_assists"] == 4
    assert inferno["team1_player1_adr"] == 70.5
    assert inferno["team1_player1_kast"] == 71.5
    assert inferno["score1_game"] == 16 and inferno["score2_game"] == 12
    assert inferno["rounds"] == 28
    assert bool(inferno["team1_win"]) is True
    assert inferno["game"] == "csgo"
    assert inferno["datetime"] == "2018-05-01"
    assert inferno["tournament"] == "Test Cup 2018"
    assert inferno["team1_player1"] == "p0, UA"          # nickname + code, no real name
    assert frame[frame["map_name"] == "Overpass"].iloc[0]["team1_player1_kills"] == 11


def test_a_map_without_a_full_line_up_is_skipped():
    players, results = _history_frames()
    frame = hp.build(players[players["team"] == "Red"], results)   # only one team present
    assert frame.empty


# --------------------------------------------------------------- labels

def test_history_label_is_nickname_and_country_code():
    assert hp.player_label("s1mple", "Ukraine") == "s1mple, UA"
    assert hp.player_label("s1mple", "Wakanda") == "s1mple"        # unknown country: no code


def test_dataset_label_strips_the_nickname_packed_into_the_real_name():
    """The source writes "Oleksandr 's1mple' Kostyliev"; the label must not repeat it."""
    assert ds.player_label("s1mple", "Oleksandr 's1mple' Kostyliev", "Ukraine") == \
        "s1mple (Oleksandr Kostyliev), UA"
    assert ds.player_label("donk", "Danil Kryshkovets", "Russia") == \
        "donk (Danil Kryshkovets), RU"


# --------------------------------------------------------------- slug heuristics

def test_slug_game_labels_the_cs2_era_only_after_the_transition():
    assert ds.slug_game("1/a-vs-b-iem-katowice-2024") == "cs2"
    assert ds.slug_game("1/a-vs-b-blast-premier-fall-2023") == "cs2"   # CS2 shipped 2023-09
    assert ds.slug_game("1/a-vs-b-blast-premier-spring-2023") == "csgo"
    assert ds.slug_game("1/a-vs-b-iem-katowice-2019") == "csgo"
    assert ds.slug_game("1/a-vs-b-esea-advanced-season-46-europe") == "csgo"  # no year


def test_slug_date_is_an_event_midpoint_and_empty_when_no_year_is_named():
    assert ds.slug_date("1/a-vs-b-blast-premier-fall-2023") == "2023-10-15"
    assert ds.slug_date("1/a-vs-b-iem-katowice-2024") == "2024-07-01"   # year, no season
    assert ds.slug_date("1/a-vs-b-esea-advanced-season-46-europe") == ""
