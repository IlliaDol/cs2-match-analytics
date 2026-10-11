# DATA.md — dataset documentation

Generated 2026-10-11 by profiling the raw CSVs (Python 3.12.10, pandas 3.x). Real numbers, measured — not copied from dataset descriptions. Previous profile: 2026-09-10 (20,676 rows, coverage to 2026-06-28).

## Files (in `data/raw/`, all git-ignored)

| File | Rows | Cols | What it is |
|---|---|---|---|
| `cs2_all_tiers_games.csv` | 22,948 | 98 | **Primary.** Every match, one row per map + one series-summary row per match (`is_total`). **Correction (2026-09-11):** `score1_match`/`score2_match` on series rows are **TEAM-sorted** (team1's / team2's maps won), NOT winner-sorted; `team1_win` on series rows is broken (88% zeros) and must never be used for the winner. Winner = team with more maps. See `tests/test_loader_contract.py`. |
| `cs2_tier1_games.csv` | 12,535 | 98 | Same schema, Tier-1 events only |
| `cs2_tier2_games.csv` | 9,044 | 98 | Tier-2 only |
| `cs2_tier3_games.csv` | 1,369 | 98 | Tier-3 only |
| `teams.csv` | 22,112 | 2 (`team_id`, `team_name`) | id → name lookup |
| `players.csv` | 1,525 | 2 (`player_id`, `player_name`) | id → name lookup (lineup columns reference ids) |
| `tournaments.csv` | 386 | 1 (`tournament`) | tournament names |
| `cs2_2025_top50_matches.csv` | 4,600 | 7 | Secondary clean 2025 slice (top-50 teams, Jan 12 – Nov 26 2025, date format `DD/MM/YY`) |

## Coverage (primary file)

- **Dates:** 2023-01-10 → 2026-10-03. **Tournaments:** 386. **Teams:** 964 unique names.
- **Row types:** 11,888 map-level (`is_total=False`) + 11,060 series-level (`is_total=True`), covering 11,056 matches (4 matches carry two series rows — Quirk 9).
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

1. Row count: **22,948** rows, 98 cols (all_tiers).
2. Date range: **2023-01-10 09:30 → 2026-10-03 22:00**, 0 datetime parse failures (ISO format).
3. Duplicates: **0** full-row; **0** on `(match_id, game_id)`; 11,892 on `match_id` alone (expected — one row per map + 1 series row; plus 4 matches with a doubled series row, Quirk 9).
4. Top tournaments by rows: PGL CS2 Major Copenhagen 2024 Closed Qualifiers (315), IEM Cologne Major 2026 (295), BLAST.tv Austin Major 2025 (272).
5. Teams: **964** unique (union of team1/team2).

## Data quirks — read before using anything

1. **Series scores are TEAM-sorted, and the series `team1_win` flag is broken.** *(Corrected 2026-09-11 — the original Quirk 1 claim was wrong; re-verified 2026-10-11.)* `score1_match`/`score2_match` = (team1's maps won, team2's maps won). Verified by (a) both orderings occurring (~55/45, no dominant order), (b) `score1+score2 == games_played` on **100%** of decided rows, (c) 99.3% winner-agreement with strong map-level evidence, (d) two external Major finals (Copenhagen 2024: NAVI 2-1 FaZe, match_id 1048414; Shanghai 2024: Spirit 2-1 FaZe, match_id 1859211 — both still resolve correctly in the refreshed snapshot). The series-row `team1_win` flag is 88% zeros and contradicts all of the above — a dataset defect; **the winner is the team with MORE maps**. On map rows, `score1_game`/`score2_game` are team-sorted as well (same finals: 6/6 map scores follow the team columns). Team-specific series scores need NO disentangling: `t1 = score1_match`, `t2 = score2_match`.
2. **~20% of series rows are Bo1s wearing a Bo3 costume.** 255 rows have `bestOf=3` but `games_played=1` and series score (1,0); another 29 rows have series score (1,0) with `bestOf` missing (both down sharply from the 2026-09-10 snapshot — upstream reporting improved). Rule: trust `games_played` and the (winner,loser) score pair, never `bestOf` alone. True Bo1s must be identified as `games_played==1` (2,227 series).
3. **Two generations of series rows.** Older: `game_id` negative, sometimes a leftover map name + 0–0 scores on the series row. Newer (~5,300 series rows without game_ids, mostly 2026): `game_id` NaN, no map fields at all. A match has either kind — never both — except the 4 doubled matches in Quirk 9.
4. **Per-map winner flag: only rounds+flag AGREEMENT is trustworthy.** On usable map rows, round-score direction and `team1_win` disagree on 381/11,853 (3.2%; tier1: 240/7,473) under the naive direction comparison. On those rows neither source is trustworthy; forfeited maps (0-0 rounds) have no usable direction. Treatment: attribute a map only when rounds and flag AGREE (strong evidence); the series row stays authoritative for the series winner and for forfeit maps absent as map rows.
5. **Missingness is structural, two flavors:** (a) series rows have empty map fields by design (~36% of `map_name` missing overall); (b) genuine gaps — 140 rows with `bestOf` NaN, 83 map-level rows without map name, player stats missing up to 56% in tier3.
6. **Team-name identity is a minefield.** `teams.csv` has 22,112 id→name rows of historic org renames (838 case-folded alias groups — e.g. the BETBOOM spelling split in Quirk 9). Raw `team1`/`team2` strings have **0 whitespace issues** (checked 2026-10-11, 45,894 names) but must not be joined blindly across sources — resolution via the id columns is still pending (M6).
7. **Lineups are per-map and per-slot** (`team1_player1_id` … `player5_kddiff`) — roster-stability and stand-in features (M7) come from comparing player-id sets across a team's consecutive matches. `players.csv` (1,525 players) maps the ids; note the 5,300 series rows without game_ids correspond to matches with NO map rows in this file at all (summary-only records).
8. **Cross-dataset overlap is partial.** 53/100 team names from the 2025 top-50 dataset appear in all_tiers — the two sources use different team-name conventions. Until M6 resolves names via `teams.csv`, don't join them blindly.
9. **Four matches carry two series rows each (new in the 2026-10-11 snapshot).** Same `match_id`, old-generation negative `game_id`s two apart (e.g. match 7287899: K27 vs WW Team, 2-0, twice). One pair even spells the team differently across its two rows (`BETBOOM TEAM` vs `BetBoom Team`, match 7298778). Treatment: `scripts/build_interim.py` dedupes cross-tier by `match_id` (keep first) and `scripts/build_features_v1.py` dedupes the roster frame the same way — series-level tables are unaffected, raw all_tiers row counts overstate matches by 4.

## Sanity conclusions for the build order

- The **series row** (`is_total=True`) is the single source of truth for "who won": the winner is the team with MORE maps (`score1_match`/`score2_match` are team-sorted). Never use the series-row `team1_win` flag.
- **Map-level rows** are for map-specific features only — and only where rounds and the flag agree (Quirk 4).
- A clean matches table = one row per `match_id` with: date, tournament, tier, team1, team2, winner (as team name), series score (t1, t2), bestOf (corrected by games_played).
