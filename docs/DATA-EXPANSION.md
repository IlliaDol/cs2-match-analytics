# DATA-EXPANSION.md — how to get a lot more data into cs2-match-analytics

Written 2026-09-14. Question asked: *"look for more datasets and more data — I need a lot of
data to work with, really a lot."* This is the honest map of what exists, what each source
unlocks, and what the machine can actually hold.

## Where we are today

| File | Rows | Note |
|---|---|---|
| `data/raw/cs2_all_tiers_games.csv` | 20,676 map rows (9,923 series) | the main table — one row per map |
| `data/raw/cs2_tier1_games.csv` | 11,151 | tier split |
| `data/raw/cs2_tier2_games.csv` | 8,305 | tier split |
| `data/raw/cs2_tier3_games.csv` | 1,220 | tier split |
| `data/raw/cs2_2025_top50_matches.csv` | 4,600 | top-50 event slice |
| `data/raw/teams.csv` | 19,846 | team lookup |
| `data/raw/players.csv` | 1,398 | player lookup |
| `data/raw/tournaments.csv` | 344 | tournament lookup |
| **Total distinct maps** | **10,675** | 2023-10-25 → 2026-06-28, 1,189 players |

Source: Kaggle **`ektarr/counter-strike-pro-matches`** (the tier/tier1/tier2 file names match
it exactly). 98 columns per map row: 10 players × (kills, deaths, assists, ADR, KAST, K/D diff)
plus map/scores/date/tournament. **This is already a wide dataset** — what it is *not* is deep:
no round-by-round events, no weapons, no economy, no positions.

### CS:GO sources added 2026-09-15

The CS2-only picture above was the whole story until the CS:GO eras were wired in. Both are
converted to the same map-row schema (plus a `game` column) by an adapter, and both live in
git-ignored `data/interim/`:

| Source | Maps | Window | Adapter |
|---|---|---|---|
| Kaggle `mateusdmachado/csgo-professional-matches` | 43,944 | 2015-11-03 → 2020-02-27 | `scripts/adapt_csgo_history.py` |
| Kaggle `fernandopy/csgo-data-set` | 20,004 | 2021–2023 (43% of rows dated) | `scripts/adapt_csgo_dataset.py` |

That brings the corpus to **74,300 map rows** (63,948 CS:GO + 10,675 CS2, 2015–2026), and
`compare_players.py --game csgo|cs2` selects an era. `docs/DATA-COVERAGE.md` is the living
audit of what is still missing (pre-2015, the 2020-03→2021 hole, assists in the 2021–2023
file, CS:GO tier labels).

## Tier A — refresh + finish the source we already use (cheap, minutes)

- **Refresh `ektarr/counter-strike-pro-matches`.** It is a living dataset (19,000+ games
  advertised; our copy ends 2026-06-28). A newer snapshot adds months of matches at the same
  98-column schema — zero code changes, the ingest path already reads it.
- **Use the lookups we already ship but never join.** `players.csv` (1,398) and
  `teams.csv` (19,846) are sitting unused. Joining them unlocks: true player IDs (so the same
  person across name changes), team rosters over time, nationality splits.
- Size: ~25 MB. Disk risk: none.
- ~~Blocked by: no Kaggle API token on this machine~~ **Token present and working** (verified
  2026-09-15: `kaggle` 2.2.4 pulled two CS:GO datasets). When adding a dataset, register it in
  `scripts/fetch_datasets.py` so the download stays reproducible.

## Tier B — additional Kaggle datasets, same style (cheap, MB-scale)

| Dataset | What it adds | Why it matters here |
|---|---|---|
| **BLAST Rivals 2026 – CS2 Match Statistics** (`eupeutro`, CC0-1.0) | per-player/per-map per-side stats **plus HLTV Rating 3.0 and Swing** | lets us *validate our own metrics* against a published rating on overlapping matches — the single most useful external check we can do |
| Other tournament/season sets (CS2 majors, 2024–2026) | more matches for specific events | widens the event-level analyses (the Vitality review's per-event table) |
| Any CS2 set with **pistol-round / side-split** columns | CT vs T performance | our data cannot answer side-specific questions at all today |

These are CSV-scale (a few MB each) and join cleanly on team/date/map. Disk risk: none.

### Tier B result — our metrics validated against HLTV's published Rating 3.0 (2026-09-14)

The BLAST Rivals 2026 set was fetched and used for exactly this check. Over the players present
in both sets (n = 38) for that event:

| Our metric | Spearman vs their Rating 3.0 |
|---|---|
| K/D (pooled) | **+0.761** |
| ADR (pooled) | **+0.736** |

So our simple, transparent metrics rank players essentially the way the published composite
rating does — which is the strongest external check available without inventing our own rating
formula. Absolute values differ, and the comparison showed exactly why:

- **their rows are map-*sides*, ours are maps** (a Bo3 map is two rows for them: CT and T) —
  ZywOo shows 22 of their rows against our 9 maps in the same event;
- **their event coverage is wider than ours** — Twistzz has 20 of their rows against our 6 maps,
  and we already know the raw table is missing maps elsewhere (148/159 in the Vitality window).

Conclusion: keep reporting the ingredients (K/D, ADR, KAST, KPR) with their sample sizes, and
cite the Rating 3.0 correlation as the external sanity check — not a claim that our numbers are
interchangeable with HLTV's.

## Tier C — round-event level from demos (this is the "really a lot")

This is the real jump: `.dem` files give **every round of every map** — kills with weapon and
position, first bloods, clutches, economy/equipment value, bomb plants, per-round players
alive. One map ≈ 24 rounds × 10 players ≈ 240 player-rounds plus ~50 kill events; our current
table has **1 row per map**.

| Source | Content | Access |
|---|---|---|
| **CS2-10k (Reka AI)** | 600,000+ player-round videos, 10,000+ hours 720p/48fps, HLTV-sourced demos, plus the open-source `cs2-dem-renderer` pipeline | dataset download + rendering; vision/ML oriented |
| **`demoparser2` / `awpy`** (Python) | parse `.dem` → round events as DataFrames | pip install; needs demo files |
| Public demo archives (bo3.gg, event hubs) | downloadable `.dem` per match | per-match downloads |

**The disk math, because it decides everything:**

```
free space on C:            24 GB   (476 GB total, 96% used)
one .dem file             100–300 MB
→ ~80–200 demos fit          ≈ 2,000–5,000 maps of round-level data
→ thousands of demos       need 0.5–2 TB  → an external drive, or Colab + keep only aggregates
```

So "really a lot" splits into two honest options:
- **on this laptop:** a *curated* set of 100–200 demos (recent tier-1 events), parsed to
  parquet — round-level depth for a specific window;
- **at scale:** process demos in Colab (the repo already uses Colab for the Spark module) and
  commit only the *aggregates* (parquet), or buy/attach external storage.

Also note: OneDrive now syncs this folder, so a 100 GB demo cache would fight sync — demos
belong outside OneDrive (e.g. `D:\cs2-demos`) with only derived parquet in the repo.

## Tier D — structured feeds (live, ongoing)

| Source | Shape | Note |
|---|---|---|
| **bo3.gg** | public CS2 API: matches, teams, players, events | no key for basic use; check ToS |
| **Liquipedia** | tournament/match data via dumps/API | good for event metadata, dates, prize pools |
| **PandaScore / Abios** | commercial esports APIs | paid tiers; free tier is rate-limited |
| Ground rules from the plan | *no scraping* (HLTV pages, etc.) | keep using mirrors/downloads, not scrapers |

## What I recommend, in order

1. **Join the lookups we already have** (`players.csv`, `teams.csv`) — pure local work, more
   signal from data already on disk, no downloads, no disk cost. **Do now.**
2. **Add Tier B**: fetch the BLAST Rivals 2026 set and use its Rating 3.0 to sanity-check our
   K/D/ADR/KAST/KPR on overlapping matches. Small, high value.
3. **Refresh Tier A** when a Kaggle token exists (or a manual download).
4. **Then decide Tier C**: curated 100–200 demos here, or demo processing in Colab. This is
   the only step that needs a decision about disk/storage.

## What I need from you

- A **Kaggle API token** (kaggle.json) — put it in `%USERPROFILE%\.kaggle\` and then
  `.venv/Scripts/python.exe scripts/fetch_datasets.py --all` fetches everything registered
  below into `data/raw/new/` (the v1 files in `data/raw/` are never overwritten). Without a
  token the script refuses with the exact steps; `--check-token` verifies it, `--list` shows
  the registry.
- A **decision on Tier C storage**: external drive, Colab-only aggregates, or skip.
- Confirmation that round-level parsing is worth the pipeline work (it changes the repo's
  shape: new ingest, new features, new tests — a real module, not a script).
