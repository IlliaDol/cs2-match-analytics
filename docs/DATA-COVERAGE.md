# CS:GO / CS2 data coverage — audit and source notes

Status 2026-10-11 (Tier A refresh applied; CS:GO sections unchanged since 2026-09-15).
Companion to `docs/DATA-EXPANSION.md`. Keeps the state of the data
explicit: what we have, how good it is, and what is still missing.

## What the tool reads

| era | source | maps | window | per-player metrics | dates |
|---|---|---|---|---|---|
| cs2 | `data/raw/cs2_all_tiers_games.csv` — Kaggle `ektarr/counter-strike-pro-matches` | 11,888 | 2023-01-10 → 2026-10-03 | kills, deaths, assists, ADR, KAST | exact |
| csgo 2015–2020 | `data/interim/csgo_history_games.csv` — Kaggle `mateusdmachado/csgo-professional-matches` | 43,944 | 2015-11-03 → 2020-02-27 | kills, deaths, assists, ADR, KAST | exact |
| csgo 2021–2023 | `data/interim/csgo_adapter_games.csv` — Kaggle `fernandopy/csgo-data-set` | 20,004 | 43% carry a year, all 2021–2023 | kills, deaths, ADR, KAST (**no assists**) | event slug, approximate |

**75,836 map rows** in total, **63,948 of them CS:GO**, across ten years.

Rebuild the CS:GO files (both are git-ignored, regenerable, and skip themselves when their
raw source is absent):

    .venv/Scripts/python.exe scripts/adapt_csgo_history.py   # 2015-2020 (needs data/raw/new/players.csv + results.csv)
    .venv/Scripts/python.exe scripts/adapt_csgo_dataset.py   # 2021-2023 (needs data/raw/new/data/)

Then compare, optionally narrowed to one era:

    .venv/Scripts/python.exe scripts/compare_players.py --a s1mple --b donk --game csgo

`compare_players.py` tags every row with `game` and merges the sources per **identity**
(nickname + country code, case-folded), because the sources disagree on spelling —
`Aleksandr`/`Oleksandr` Kostyliev, `Donk`/`donk`, and the short `Nick, CC` form the 2015–2020
file uses. Without that, one career splits into two players. `--game` accepts repeat values;
omitting it keeps everything.

## Quality per source

* **2015–2020** — the best of the three: 43,944 maps, every one with a complete ten-player
  line-up, null shares of 0% (kills/deaths/assists) to 1% (ADR/KAST), real match dates and
  event names. No player real names (nickname + country only) and no tier label.
* **2021–2023** — 20,004 maps with complete line-ups, but **assists are absent**, 57% of
  rows carry no year (a `--from/--to` window silently drops them), and dates are
  season-derived rather than match dates. `TeamRank` is the only quality proxy.
* **CS2** — 11,888 maps with everything including assists and exact dates, refreshed
  2026-10-11 (was a stale snapshot; see gap 5, now closed). Tiers 1/2/3 available.

## Roster / transfer dates — the gap worth closing (2026-09-15)

`roster_stability_diff` and `standin_diff` are the two strongest features in the model, but
nothing in the match data says *when* a player joined or left a team — a stand-in looks
identical to a permanent member. Those dates live on Liquipedia's transfer pages, one
structured template per move:

    {{Transfer Row|name=dev1ce|flag=dk|team1=astralis|date=2025-12-31|ref=...}}       leaving
    {{Transfer Row|name=Plopski|flag=se|team1=metizport|team2=metizport|role1=Inactive|...}}

`scripts/fetch_liquipedia_transfers.py` walks them (one page per month) into
`data/interim/roster_transfers.csv` — `date, player, flag, from_team, to_team, role, page`.
A single template can cover a whole line-up (`name2`…`name5`), and those get expanded.

Verified by hand before writing the script:

| page | HTTP | transfer templates |
|---|---|---|
| `Player_Transfers/2025/December` | 200 | 237 → 387 person-records once multi-player rows expand |
| `Player_Transfers/2020/January` | 200 | 451 |
| `Player_Transfers/2016/January` | 200 | 147 |
| `Player_Transfers/2014/January` | 404 | — |

So coverage runs from ~2015 to the present: both games, **and 2020–2021, the window where our
*match* data has a hole**. Politeness: 1 request / 2 s, descriptive User-Agent, every page
cached under the git-ignored `data/interim/`.

Not done yet: joining these dates onto the line-ups to rebuild "who was on the roster on each
match day". That is where the feature improvement actually lands.

## Gaps that remain

1. **Match data: 2012 → 2015-11, and 2020-03 → 2021.** The second one could not be closed by
   any per-player per-map source either I or a second reviewer found; 2020-07→2021 does not
   appear to exist in structured form, and pre-2015 structured stats start around 2015 on HLTV
   too. The *roster* dates for that window do exist (§ above).
2. **Assists missing in the 2021–2023 file**, so assist-derived metrics still cannot be
   compared across every era (both other sources have them).
3. **57% of 2021–2023 rows are undated** — only fixable with a source that carries dates.
4. **No tier label for either CS:GO source.** `rank_1`/`rank_2` (2015–2020) and `TeamRank`
   (2021–2023) could be mapped onto the tier scheme.
5. ~~**CS2 snapshot is stale** — refresh `ektarr/counter-strike-pro-matches` for the months
   after 2026-06-28; upstream is alive.~~ **Closed 2026-10-11:** refreshed to 2026-10-03
   (22,948 rows, 11,056 matches); models retrained, docs re-measured.
6. **Rating 3.0 / Swing have no open reimplementation**, so our ingredient metrics cannot be
   checked against HLTV's published rating. `awpy` computes ADR/KAST independently but only
   from individual demo files, which will not scale on this disk.
   `eupeutro/blast-rivals-2026-cs2-match-statistics` (in `scripts/fetch_datasets.py`, not
   wired) carries Rating 3.0 for one event.

## Checked and not adopted (second-pass research, 2026-09-15)

I sent a second reviewer to search beyond what I had found, including in Portuguese, Chinese,
Russian and Korean. What I could **verify myself**:

* the PyPI packages it named all exist — `ggpyscraper` 0.0.3.3 (Liquipedia scraper; its
  description mentions `parse_transfers` and `counterstrike`), `cs2api` 0.1.3 (MIT, wraps
  BO3.gg), `awpy` 2.0.2;
* the Liquipedia transfer structure and its ~2015-onward depth (table above);
* **HLTV is now bot-blocked, measured rather than assumed** (below).

### HLTV scraping: measured, 2026-09-15

Three endpoints, requested with an ordinary browser User-Agent:

| URL | result |
|---|---|
| `https://www.hltv.org/matches` | `403 Forbidden`, `Cf-Mitigated: challenge`, `Server: cloudflare` |
| `https://www.hltv.org/ranking/teams` | same |
| `https://www.hltv.org/results` | same |

A Brazilian thesis project hit the identical wall and moved to **Liquipedia** for the same
reason; two Chinese HLTV plugins ship a Cloudflare proxy or state outright that real-time data
cannot be fetched. So HLTV is not a grey area — it is actively challenged, and passing it needs
a headless browser (~150 MB) or a paid residential proxy, both against their terms.

**Consequence:** the three permanent gaps below (pre-2015, per-round events, an independent
Rating 3.0) stay permanent unless we pay. Liquipedia remains the only free, sanctioned route.

What I could **not** verify, and therefore do not treat as fact — these are someone else's
reading of vendor pages: PandaScore's historical tier costing €400/month, Esports Charts being
viewership-only, PureSkill.gg being public matchmaking rather than pro play, Bayes Esports'
August 2025 bankruptcy and whoever succeeded it, and bo3.gg having no documented API. Check
each before acting on it.

## Measured dead ends, 2026-09-15 — read this before chasing more data

Three promising directions were tried to the end. All three are now closed **with numbers**,
and the reason they failed is the same one every time.

### 1. Dated rosters — sourced successfully, then measured as useless

Liquipedia's transfer history was fetched in full (59,127 dated moves, 2002→2026, 12,867
people) and turned into three strictly pre-match features: `tenure_min_diff`,
`tenure_avg_diff` (days since the newest / average member joined, team1−team2) and
`recent_join_diff` (how many joined within 30 days). Player names matched **97%** against the
match rows, so the join was sound. `scripts/roster_dates_experiment.py` trains both feature
sets on identical rows over the same split:

| | acc | logloss | ece |
|---|---|---|---|
| current 9 features | **0.6328** | **0.6377** | 0.0176 |
| + dated roster (3) | 0.6311 | 0.6384 | 0.0146 |
| delta | **−0.0017** | +0.0007 | −0.0030 |
| shipped model (replication check) | 0.6367 | 0.6377 | 0.0185 |

The baseline reproduces the shipped model (0.6328/0.6377/0.0176 vs 0.6367/0.6377/0.0185), so
the comparison is trusted — **and the new features are not an improvement.** Two reasons:
only 26.8% of series have both line-ups dated (the rest get 0 = "no information", diluting the
signal), and the information is largely **redundant** — `roster_stability_diff` already sees
that a line-up changed. "He joined three days ago" adds little on top of "the line-up changed".

### 2. `picks.csv` (map veto order) — the right data, the wrong universe

`mateusdmachado/csgo-professional-matches` ships a perfect veto table (`t1_removed_1..3`,
`t1_picked_1`, `left_over`, `system`, `best_of`) — exactly the signal the model lacks. It
cannot be used:

| | source | match_id range |
|---|---|---|
| `picks.csv` | CS:GO 2015–2020, 16,035 series | 2,302,057 → 2,340,461 |
| `features_v1` | CS2 2023–2026, 9,920 series | 487,653 → 10,094,514 |
| intersection | | **2 (0.0%)** |

And the CS2 source (`ektarr`) has no pick/veto column at all. Different era, different id space.

### 3. CS2 veto from Liquipedia — exists, but unjoinable today

`Template:MapVeto` does not exist (`"missing"`), yet a search returns **102 pages** containing
"Mapveto", so Liquipedia does record it in some other structure (and its API requires gzip —
solvable with `--compressed`). But even with the data, matching it to our 9,920 series means
matching on (team names, date), and **exact team-name matching is only 54%** (measured).

### The conclusion worth keeping

The bottleneck is **identifiers, not volume**. `ektarr` uses a private `match_id` space that
nothing else shares, and team-name agreement is 54%. That is why three separate data hunts
ended the same way. So:

* **No further data hunt is worth starting** until either (a) we predict at **map level**, where
  the same file gives 10,753 maps instead of 5,173 series and there is **no join problem at
  all**, or (b) a **team-id crosswalk** (`ektarr team_id` ↔ canonical name) is built once, which
  would unlock every external source at the same time — veto, rosters and line-ups alike.
* Until then the 9 features look **saturated for this kind of information**, in the same way
  `dl_embedding` loses to logistic: an honest negative, not a failure.

## Closed on 2026-09-15

* **CS:GO 2015–2020 added.** Found `mateusdmachado/csgo-professional-matches` (383,317
  player-match rows, 2015-10-07 → 2020-02-27), verified the join — `results.csv._map` is
  matched to `players.csv.map_1/2/3` **by name, never by row order** — and wired it through
  `scripts/adapt_csgo_history.py`. s1mple went from 196 to 919 CS:GO maps and donk from 0 to
  234, which is what makes a fair donk-vs-s1mple question possible at all.
* **Cross-era identity.** The same player now merges across all three sources instead of
  splitting on spelling (`key_of` handles both label shapes).
* **Roster dates sourced.** Liquipedia's transfer pages give dated moves from ~2015, including
  the 2020–2021 match-data hole. Fetcher written and verified on one month.

## If more data is ever needed

The hard remainder is pre-2015, per-round granularity (economy, buy types, positions) and any
independent reimplementation of HLTV's ratings — those need demo files (disk-heavy) or a paid
provider. Never fabricate a metric to fill a gap: leave it NaN and record it here.
