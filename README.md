# CS2 Match Analytics

[![CI](https://github.com/IlliaDol/cs2-match-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/IlliaDol/cs2-match-analytics/actions/workflows/ci.yml)

How honest are win probabilities for professional CS2? This repo builds them from
scratch — Elo baseline, logistic/GBM on strictly pre-match features, Bayesian ratings —
and measures calibration on a fixed time split, not just accuracy.

![calibration](outputs/fig_calibration.png)

## Results

Time-split test set (train < 2026-01-01, n = 2,943 series). Lower logloss is better;
`constant_0.5` = always predict 0.5 (logloss ln 2 = 0.6931) — the floor every real
model must beat, shown on purpose. Refreshed 2026-10-11 on the Tier A data
(11,052 series); every row re-measured on the same split.

| model | logloss | brier | acc | ece |
|---|---|---|---|---|
| **lr+roster** (Elo + form/rest/h2h + roster-stability/stand-in) | **0.6342** | 0.2219 | 0.6306 | 0.0159 |
| lr (Elo diff + form/rest/h2h) | 0.6496 | 0.2290 | 0.6089 | 0.0176 |
| elo_k32 (from-scratch engine) | 0.6512 | 0.2297 | 0.6123 | 0.0301 |
| gbm | 0.6491 | 0.2286 | 0.6259 | 0.0214 |
| gbm_isotonic | 0.6529 | 0.2305 | 0.6188 | 0.0280 |
| constant_0.5 | 0.6931 | 0.2500 | 0.5671 | 0.0671 |
| dl_embedding (PyTorch) | 0.6681 | 0.2371 | 0.5916 | — |

`ece` is the Expected Calibration Error — the average gap between a stated probability and
what actually happened. The chart built from it is
[`outputs/fig_calibration.png`](outputs/fig_calibration.png); if the axes look cryptic,
[`docs/CALIBRATION-README.md`](docs/CALIBRATION-README.md) walks through how to read them —
including why the plain `gbm` is now the best-calibrated row while per-tier
`isotonic` refits lose on both metrics (rejected; measured below).

Reads: a from-scratch Elo engine gets 0.651; adding form/rest/head-to-head features to a
logistic model edges it to 0.6496; adding **roster-stability + stand-in** (the per-map
lineups that sat unused) drops the linear model to **0.6342** — still the biggest
feature-family win in the repo, and the gradient booster (0.6491) still trails it,
so the linear story survives the refresh with ~60% more test series. The
deep-learning variant still *loses* to logistic (0.6681) — at this data size the signal
stays linear-ish in Elo space, and that honest negative is a finding. Per-regime
calibration still flags tier-3: ECE 0.047 (vs 0.02–0.03 elsewhere).
Bayesian ratings (PyMC Bradley-Terry, `outputs/bayesian_ratings.csv`) quantify what Elo
cannot: per-team uncertainty.

## Method

- **Data:** 11,052 professional CS2 series (2023-01 → 2026-10, 3 tiers) from Kaggle; winners
  verified against map-level evidence and two real Major finals.
- **Elo:** the update rule derived as one SGD step on logistic log-loss; replayed
  time-ordered with pre-match ratings only (K-sweep peaks at K=32).
- **Features:** strictly pre-match — Elo gap, rolling 5-series form, rest days,
  head-to-head share, format flag; a leakage guard rejects any post-outcome column.
- **Model:** logistic regression vs gradient boosting vs a team-embedding MLP, all on a
  fixed time split; isotonic calibration tested and honestly reported as a negative.
- **Calibration:** reliability diagrams, ECE, Brier decomposition; drift monitored via
  monthly PSI on the Elo gap.

## Limitations

- No market odds: the v1-subset of bookmaker odds is too small for the model-vs-market
  chart, so calibration is vs *outcomes* only.
- Bo1s are a different regime (map-veto-free); the format flag helps but doesn't fully
  capture it — and the Bo1/Bo3 scatter shows format specialists exist.
- Roster/stand-in features are now **built and used** (lr+roster, see Results). But the
  raw signal is skewed: 92.6% of lineups are fully stable between consecutive series, so
  `roster_stability_diff` is mostly zeros and `standin` fires ~2% of rows — a rare-but-real
  signal, not a strong continuous one.
- Tier-3 forfeit noise: ~1.5% of map rows carry unreliable winner flags; 2 corrupt
  series rows were dropped rather than repaired.
- Predictions are symmetric-by-construction and calibrated on the aggregate —
  per-regime calibration is tested: ECE is honest (0.02–0.03) at tier1/2 and
  bo1/bo3, and **tier-3 remains the weak spot** (measured refits below, none
  enabled by default).
### Per-tier calibration refits (measured)

`scripts/build_tier_calibration.py` on the test split (2,943 series). `before` is the shipped
model; the other two columns are the refit applied to the same rows.

| tier | n | ECE before | ECE isotonic | ECE platt | logloss before | logloss isotonic | logloss platt |
|---|---|---|---|---|---|---|---|
| tier1 | 1259 | 0.0335 | 0.0245 | **0.0229** | 0.6210 | 0.6441 ✗ | **0.6194** |
| tier2 | 1327 | 0.0238 | 0.0272 | **0.0229** | 0.6411 | 0.6413 | 0.6424 |
| tier3 | 357 | 0.0474 | 0.0715 ✗ | 0.0530 | 0.6546 | 0.7479 ✗ | 0.6572 |
| all | 2943 | 0.0159 | 0.0224 | 0.0172 | 0.6342 | 0.6555 | 0.6344 |

**Verdict.** Per-tier *isotonic* is rejected again, harder: tier-3 ECE worsens
0.047 → 0.072 while logloss blows out 0.655 → 0.748, and the aggregate is worse on
both (ECE 0.016 → 0.022, logloss 0.634 → 0.656) — the map fits bin edges, not signal.
Per-tier **Platt** is the only map that helps anywhere (tier-1 better on both:
ECE 0.034 → 0.023 with logloss 0.621 → 0.619) but costs tier-3 on both, leaving the
aggregate essentially flat (ECE 0.016 → 0.017, logloss 0.6342 → 0.6344). None of
this is enabled in serving by default. The actionable finding is the one already in
`docs/DECISIONS.md`: **tier-3 needs more data** (529 train / 357 test series), not a fancier map.
The selective "calibrate only suspicious tiers" variant selected every tier (nothing
skipped), so it degenerates to plain Platt.

## Reproducibility

```bash
pip install -e ".[dev,ml,serve]"
# data (manual, git-ignored): download the Kaggle CSVs into data/raw/ — see DATA.md
python scripts/build_interim.py                        # series table
make r-m3                                              # R 01-03 (or call Rscript directly)
python scripts/build_features_v1.py                    # feature store (parquet)
python -m cs2analytics.models.train                    # artifacts + MLflow run
python -m cs2analytics.serve.monitor                   # drift report
```

`Rscript` is resolved from PATH by the Makefile (falling back to the local
Windows install) — no hardcoded path anywhere else.

The trained artifact is **`lr-roster-2026-10-11`**: the exact `lr+roster` model
from the Results table (logloss 0.6342), including the roster features. The
serving API and the Streamlit demo both report `model_version` from
`artifacts/features.json`, so the live demo is verifiably the paper's model —
not a stale `lr` from before the roster work.

The analysis notebooks (`notebooks/`) run top-to-bottom with `jupyter execute` and
produce every chart/table in `outputs/`. Tests: `pytest -q` (data-gated contract
tests skip when the private data is absent; `tests/test_model_fitting.py` runs the
actual model-fitting code on synthetic data so a fresh clone executes the
modeling layer too). CI installs `.[dev,ml]`.

## Going deeper

- [`notebooks/07_break_effect_did.ipynb`](notebooks/07_break_effect_did.ipynb) — does a
  ≥30-day break hurt a team's next match? A DiD with three specifications; the naive
  "rust" story is **not** supported, selection into breaks dominates.
  Results: [`outputs/m13_did_results.csv`](outputs/m13_did_results.csv).
- [`docs/CAUSAL-ROSTER-STUDY.md`](docs/CAUSAL-ROSTER-STUDY.md) — a separate roster-change
   cohort pipeline with deterministic controls, event-study leads/lags, DiD, and an
   event-cluster bootstrap. The current run shows a strong pre-treatment lead difference,
   so its +0.1545 DiD is a diagnostic and **not a publishable causal claim** until the
   identification strategy is redesigned.
- [`r/04_rating_forecast.R`](r/04_rating_forecast.R) — Holt's linear trend vs a naive
  random walk on monthly Elo: the random walk wins for 9 of 10 teams (ratings are
  near-martingale). Results: [`outputs/rating_forecast_metrics.csv`](outputs/rating_forecast_metrics.csv).
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — every design decision a reviewer would
  question, one line each.
- [`docs/DATA-COVERAGE.md`](docs/DATA-COVERAGE.md) — the data audit: what we have, what is
  missing, and the **three dead ends we closed with numbers** (dated rosters, map-veto order,
  HLTV scraping) plus why the bottleneck is identifiers rather than volume. Read it before
  hunting for more data.
- **Two players, head to head** — [`scripts/compare_players.py`](scripts/compare_players.py)
  builds an HLTV-style comparison from our own map rows (K/D, ADR, KAST, KPR, per-map and
  per-event splits, plus the maps the two actually shared):

      .venv/Scripts/python.exe scripts/compare_players.py --a donk --b s1mple
      .venv/Scripts/python.exe scripts/compare_players.py --a "Олександр Костилєв" --b donk

  Names resolve by nickname, full name, or native Cyrillic (see `data/player_names.csv`).
  Fill that sheet's `corrected_latin` column with your own spelling and the report shows it
  instead of the dataset's — e.g. the Ukrainian `Oleksandr` over the shipped `Aleksandr`.
  The comparison spans **both games**: `--game csgo|cs2` picks an era, default both, so
  donk-vs-s1mple is no longer a post-2023-only question. Coverage, per-source quality and
  the remaining holes are in [`docs/DATA-COVERAGE.md`](docs/DATA-COVERAGE.md) — rebuild the
  CS:GO eras with `scripts/adapt_csgo_history.py` (2015–2020) and
  `scripts/adapt_csgo_dataset.py` (2021–2023).

## Data & ethics

Match data comes from a public Kaggle dataset (git-ignored here; see DATA.md for schema
and the documented quirks, including a broken winner flag caught by cross-checking two
Major finals). This project is a **calibration study**: it measures how well
probabilistic models can predict outcomes and how honest their uncertainty is. It is
not betting advice; nothing here should be used to place wagers, and the repo takes no
position on gambling.