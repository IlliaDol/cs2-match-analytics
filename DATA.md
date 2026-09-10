# DATA.md — dataset documentation

Generated 2026-09-10 by profiling the raw CSVs (Python 3.12.10, pandas 2.x). Real numbers, measured — not copied from dataset descriptions.

## Files (in `data/raw/`, all git-ignored)

| File | Rows | Cols | What it is |
|---|---|---|---|
| `cs2_all_tiers_games.csv` | 20,676 | 98 | **Primary.** Every match, one row per map + one series-summary row per match (`is_total`) |
| `cs2_tier1_games.csv` | 11,151 | 98 | Same schema, Tier-1 events only |
| `cs2_tier2_games.csv` | 8,305 | 98 | Tier-2 only |
| `cs2_tier3_games.csv` | 1,220 | 98 | Tier-3 only |
| `teams.csv` | 19,846 | 2 (`team_id`, `team_name`) | id → name lookup |
| `players.csv` | 1,398 | 2 (`player_id`, `player_name`) | id → name lookup (lineup columns reference ids) |
| `tournaments.csv` | 344 | 1 (`tournament`) | tournament names |
| `cs2_2025_top50_matches.csv` | 4,600 | 7 | Secondary clean 2025 slice (top-50 teams, Jan 12 – Nov 26 2025, date format `DD/MM/YY`) |

## Coverage (primary file)

- **Dates:** 2023-01-10 → 2026-06-28. **Tournaments:** 344. **Teams:** 793 unique names.
- **Row types:** 10,753 map-level (`is_total=False`) + 9,923 series-level (`is_total=True`).
- `cs2_all_tiers` = tier1+tier2+tier3 concatenation (same shape; row order differs).

## Schema (main columns; 98 total — the rest are per-player stats, see Quirk 7)

| Column | Meaning (verified against the data, not the description) |
|---|---|
| `match_id` | one match/series; constant across all its rows |
| `game_id` | one map. **Positive** on map rows, **negative** on older series rows (abs value = a map id), **NaN** on newer series rows (4,750 rows) |
| `is_total` | `True` → this row summarizes the whole series (map/game fields empty or bogus) |
| `score1_match`, `score2_match` | **winner-sorted**: (winner's series score, loser's) — NOT team1/team2. Max = 3 on Bo5 ✓ |
| `score1_game`, `score2_game` | **winner-sorted** rounds on that map: (map winner's rounds, loser's). Overtime exists (max 16–19). NaN on series rows |
| `team1_win` | on series rows: 1 if **team1 won the series**. On map rows: 1 if team1 won **that map** (except Quirk 4 noise) |
| `bestOf` | 1 / 3 / 5 (5 missing on map rows, 7 in tier1) |
| `map_name` | missing on 78 map-level rows (mostly tier-2/3 forfeits) |
| `team1/2_playerN`, `..._kills/deaths/assists/adr/kast/kddiff` | full lineups + stats per map |

## The 5 sanity checks (real output)

1. Row count: **20,676** rows, 98 cols (all_tiers).
2. Date range: **2023-01-10 09:30 → 2026-06-28 20:00**, 0 datetime parse failures (ISO format).
3. Duplicates: **0** full-row; **0** on `(match_id, game_id)`; 10,753 on `match_id` alone (expected — one row per map + 1 series row).
4. Top tournaments by rows: PGL CS2 Major Copenhagen 2024 Closed Qualifiers (315), IEM Cologne Major 2026 (295), BLAST.tv Austin Major 2025 (272).
5. Teams: **793** unique (union of team1/team2).

## Data quirks — read before using anything

1. **Score columns are winner-sorted, not team-sorted.** Verified: series pairs are only ever (2,0), (2,1), (3,0), (3,1), (3,2) — never (0,2). On map rows, 10,543/10,714 have `score1_game > score2_game` exactly when `team1_win=1`. To get team-specific series scores: `t1 = wl_w if team1_win else wl_l`.
2. **~20% of series rows are Bo1s wearing a Bo3 costume.** 1,753 rows have `bestOf=3` but `games_played=1` and series score (1,0); another 2,020 rows have series score (1,0) with `bestOf` missing. Rule: trust `games_played` and the (winner,loser) score pair, never `bestOf` alone. True Bo1s must be identified as `games_played==1`.
3. **Two generations of series rows.** Older: `game_id` negative, sometimes a leftover map name + 0–0 scores on the series row. Newer (~2,020+ matches, mostly 2026, incl. 1,182 in tier1): `game_id` NaN, no map fields at all. A match has either kind — never both.
4. **Per-map winner flag is 98.4–98.5% reliable.** 171/10,714 map rows (tier1: 99/6,695 = 1.48%) have `team1_win` contradicting the round scores; 11 whole matches disagree everywhere, 154 mixed (partly forfeits/unplayed maps recorded as 0–0). Treatment: **the series row is authoritative for the series winner**; drop or down-weight the contradicting map rows.
5. **Missingness is structural, two flavors:** (a) series rows have empty map fields by design (~34% of `map_name` missing overall); (b) genuine gaps — 20 rows with `bestOf` NaN, 78 map-level rows without map name, player stats missing up to 56% in tier3.
6. **Team-name identity is a minefield.** `teams.csv` has 19,846 id→name rows (historic org renames). Raw `team1`/`team2` strings have **0 whitespace issues** but **−1 case collisions** in all_tiers (a name pair differing only by case maps to 2 different ids — to be resolved in M6 with the id columns). Case-collision count: all_tiers −1, tier2 −1, tier1/tier3 0.
7. **Lineups are per-map and per-slot** (`team1_player1_id` … `player5_kddiff`) — roster-stability and stand-in features (M7) come from comparing player-id sets across a team's consecutive matches. `players.csv` (1,398 players) maps the ids; note the 4,750 series rows without game_ids correspond to matches with NO map rows in this file at all (summary-only records).
8. **Cross-dataset overlap is partial.** 53/100 team names from the 2025 top-50 dataset appear in all_tiers — the two sources use different team-name conventions. Until M6 resolves names via `teams.csv`, don't join them blindly.

## Sanity conclusions for the build order

- The **series row** (`is_total=True`) is the single source of truth for "who won": use `team1_win` + winner-sorted `score1_match/score2_match`.
- **Map-level rows** are for map-specific features only — and after dropping the ~1.5% flag noise.
- A clean matches table = one row per `match_id` with: date, tournament, tier, team1, team2, winner (as team name), series score (t1, t2), bestOf (corrected by games_played).
