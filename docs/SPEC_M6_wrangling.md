# M6 BUILD SPEC — Core 4 Data Wrangling (EDA + cleaning + features)  · `DA DS`

**Book sync:** Think Like a Data Scientist Part 2 (Ch. 6–10) runs WITH this module —
every cleaning decision gets a one-line "what does this cost me" justification.

**Goal:** turn the raw 98-column Kaggle file into a **feature store** the M7 models
consume, with an EDA notebook that tells the data's story. Much of the base cleaning
already happened in the loader; M6 adds the *judgment* layer.

---

## §1 `notebooks/02_eda.ipynb` (yours)

On `outputs/series_clean.csv` + the map-level rows from `data/raw/cs2_all_tiers_games.csv`:
1. Profile: missingness by column (structural vs genuine — DATA.md Quirk 5), top-10
   tournaments, series-per-team distribution (log-y histogram).
2. **3 plots that surprised you** — literally label them that in the notebook. Candidates:
   win-share volatility by tier; Bo1 win-share vs Bo3 win-share for the same team
   (regression to the mean → format effect); sweep rate over time.
3. One Godsey-style paragraph per plot: what would you DO about it?

## §2 `src/cs2analytics/cleaning.py` (yours) — the decision log as code

Pure functions, unit-tested with tiny frames:

```python
def deduplicate_series(df) -> pd.DataFrame        # match_id unique, keep first, log count
def normalize_team_names(df, mapping) -> pd.DataFrame  # canonical ids via mapping dict
def flag_forfeits(df) -> pd.DataFrame             # 0-0 map scores, missing map_name rows
def winsorize_round_scores(df, lo=0, hi=25) -> pd.DataFrame
```

**Contract:** every function type-hinted, no silent mutation (return copies, log drops).
`tests/test_cleaning.py` pins behavior with tiny synthetic frames (tracked, CI runs them):
- `flag_forfeits` marks rows where `games_played >= 1` and both series scores are 0
- `normalize_team_names` maps `NAVI`/`Natus Vincere` to the same canonical id when a
  mapping `{natus vincere: navi}` is given, leaves unknown names untouched
- `winsorize_round_scores` clips series scores into `[lo, hi]` (default 0..25) and leaves
  the row count and winner column untouched

## §3 `src/cs2analytics/features/form.py` — rolling features (leakage law)

```python
def rolling_form(matches, team, n=5) -> float        # win share over team's last n BEFORE ref
def days_rest(matches, team, ref_ts) -> int          # days since team's previous series
def h2h_record(matches, team_a, team_b, before_ts) -> tuple[int, int]
```

All three take an explicit reference timestamp and must only look BACKWARDS.
`tests/test_form.py`: build a 6-row synthetic schedule where team X plays dates
D1..D6 with known results; assert `rolling_form(team, ref=D5) == exact fraction`,
`days_rest == expected`, `h2h == expected` — plus one leakage trap: calling with
`ref_ts` between two matches must return the pre-ref value, never the post.

## §4 Feature drafts exported → `outputs/features_v1.parquet`

One row per match, and **these exact column names** (M7's `build_feature_matrix` reads them):

```
match_id, datetime, tier, is_bo1,
elo_diff,            # elo_t1_pre − elo_t2_pre (from M4's run_elo_backtest)
form5_diff,          # rolling_form(team1, 5) − rolling_form(team2, 5), both pre-match
rest_days_diff,      # days_rest(team1) − days_rest(team2)
h2h_t1_win_share,    # team1's win share vs team2 before this match (0.5 if never met)
```

Notes:
- `is_bo1` is `games_played == 1` — a PRE-match fact only because the format is known
  before the series; `games_played` itself stays forbidden in the model (post-outcome).
- `elo_diff` needs M4 §3's backtest output; if that isn't built yet, compute it inline by
  calling `run_elo_backtest` on `series_clean.csv` — the engine is already shipped.
- M7's leakage guard will reject `days_rest_t1`/`days_rest_t2` style raw columns only if you
  add them to the feature list; the parquet may carry extra columns, the matrix is selective.

## Done when
- [ ] pytest green (cleaning ~6 + form ~8 new tests)
- [ ] EDA notebook tells raw→problems→fixes→3-surprises in order
- [ ] `DATA.md` decision log extended with every cleaning decision (date, decision, cost)
- [ ] Commit: `m6: cleaning + rolling features + EDA (module 6 done)`

## Checkpoint
1. Name one cleaning decision that IMPROVES model accuracy but makes the data less
   representative of the real world. When is that trade acceptable?
2. Rolling form: why `min_periods` matters (early-season NaN handling)?
3. What breaks if you compute `days_rest` with `>` instead of `>=`?

## Appendix — hints
- `rolling_form`: filter team's matches with `datetime < ref_ts`, tail(n), mean of wins.
- Mapping file: build from `teams.csv` + manual merges for the ~10 big orgs; don't
  attempt full fuzzy matching (scope law).
