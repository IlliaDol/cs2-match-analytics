# Prompt for Claude — find every usable CS2 / CS:GO match data source

Copy everything below the line. It is written to be pasted into a fresh Claude conversation.
It gives Claude the real numbers so it can check its findings against a baseline, and it asks
for sources *with evidence* rather than guesses.

---

## What I need

I maintain a personal data-science project that predicts **who wins a professional Counter-Strike
match**. I want **a lot more training data**, and I want to know about datasets I have not
found. Your job is to **search widely and report concrete, verifiable sources** — not to give
me general advice.

Do not invent datasets, links, or row counts. If you are unsure whether a dataset exists,
say so explicitly. For every source you propose I need: the exact name, a working URL, what
it contains at what granularity, how many rows/matches it realistically has, the date range,
its licence, and whether it is downloadable without a paid account.

## My situation right now (verified numbers, 2026-10-11)

**The model.** A calibrated probabilistic model of match outcome: output = P(team1 wins), a
binary label at the *series* level (`team1_win`). Time-split evaluation: train on matches
before **2026-01-01** (8,109 series), test on matches after (2,943 series). Measured:
accuracy **0.6306**, logloss **0.6342**, Brier 0.2219, ECE 0.0159. The "always say 50%"
baseline is logloss 0.6931, so the model beats it but not by a landslide.

**The 9 features it uses:** `elo_diff`, `form5_diff`, `rest_days_diff`, `is_bo1`,
`tier_tier1`, `tier_tier2`, `tier_tier3`, `roster_stability_diff`, `standin_diff`.
So the things that matter to me are: team strength ratings, recent form, rest/schedule,
series format, tournament tier, and **lineup/roster information (stand-ins, roster changes)**.

**The training data I have.** One Kaggle dataset (`ektarr/counter-strike-pro-matches`,
98 columns, one row per map with all ten players' kills/deaths/assists/ADR/KAST):
**11,888 maps across 11,052 series in the modeling table (2023-01-10 → 2026-10-03)**. It is CS2-era only. All ten
players' per-map stats are present, which is what lets me build the roster features.

**Other data I have downloaded but am NOT training on yet** (map-level, player-level stats):
- Kaggle `fernandopy/csgo-data-set` — 20,004 CS:GO maps, mostly 2021–2023, no assists, 57% undated.
- Kaggle `mateusdmachado/csgo-professional-matches` — 43,944 CS:GO maps, 2015-11-03 → 2020-02-27,
  real dates, per-map kills/deaths/assists/ADR/KAST + HLTV rating, joinable via match_id.
Both are usable for a *separate* CS:GO model but are **per-map**, while my label is **per-series**,
so they need aggregation before they can train anything.

**Hard constraints.** My laptop's C: drive has **166 GB free**, so multi-gigabyte
downloads are fine unless they reach demo-archive scale. I am a student — paid
data providers are out unless there is a free tier. I have a working Kaggle API token.

## What I want you to search for, in this order

1. **The best CS2 sources beyond the one I use.** Competitors/alternatives to
   `ektarr/counter-strike-pro-matches` — other Kaggle datasets, GitHub scrapers, or archives
   that have **per-player per-map stats and a date** for the CS2 era. I want more months, more
   tiers, and more leagues than I currently have.

2. **CS:GO-era sources with per-player per-map stats AND dates**, covering 2012–2024, so a
   separate CS:GO model becomes possible. Especially: does anything cover **2020-03 → 2021**
   (the hole between my two CS:GO files) and **before 2015**?

3. **Roster / lineup / transfer data as its own source** — this is my weakest area and the
   highest-value addition. Stand-in appearances, roster changes with dates, player-to-team
   moves. If a dataset or API gives *dated* roster changes, that alone improves my two
   best-performing features. Tell me explicitly if such a source exists and where.

4. **Anything that would let me validate my metrics**, e.g. HLTV Rating 3.0, Swing, or KAST
   from an independent pipeline, so I can check whether my computed numbers agree.

5. **Non-Kaggle routes**: HLTV scraping legality/robots/rate limits, any official API and its
   price, bo3.gg / Liquipedia / Esports Charts / open-source scrapers on GitHub that are
   currently working (not abandoned). For each, say what it would take for one person to run
   it politely and how many matches it would realistically yield.

## What I do NOT want

- Generic advice like "try web scraping" or "use an API" without naming it.
- Datasets that are only match *results* with no player stats — I need player-level rows to
  build roster features.
- Anything you cannot point to with a URL.
- Made-up row counts. If the page does not state a count, say "unknown".

## How to format your answer

A table with one row per source: **name · link · granularity (map/player/series) · what it has ·
realistic size · date range · licence · free? · what it would add to my 9 features**. Then a
short ranked recommendation: which single source to add first and why, and which ones are dead
ends. Finally, list what you searched but found nothing for — I want to know where the gaps
genuinely are, since I may need to accept that the data simply does not exist.
