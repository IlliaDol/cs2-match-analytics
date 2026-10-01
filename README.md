# CS2 Match Analytics

[![CI](https://github.com/IlliaDol/cs2-match-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/IlliaDol/cs2-match-analytics/actions/workflows/ci.yml)

A from-scratch probabilistic model of professional CS2 match outcomes — Elo baseline,
calibrated logistic/GBM, Bayesian team ratings, validated walk-forward — built as a
walkthrough of the whole data-science stack.

![calibration](outputs/fig_calibration.png)

## Results

Time-split test set (train < 2026-01-01, n = 1,811 series). Lower logloss is better;
`constant_0.5` = always predict 0.5 (logloss ln 2 = 0.6931) — the floor every real
model must beat, shown on purpose.

| model | logloss | brier | acc | ece |
|---|---|---|---|---|
| **lr+roster** (Elo + form/rest/h2h + roster-stability/stand-in) | **0.6377** | 0.2237 | 0.6367 | 0.0185 |
| lr (Elo diff + form/rest/h2h) | 0.6464 | 0.2278 | 0.6190 | 0.0242 |
| elo_k32 (from-scratch engine) | 0.6472 | 0.2282 | 0.6179 | 0.0251 |
| gbm | 0.6511 | 0.2298 | 0.6218 | 0.0255 |
| gbm_isotonic | 0.6560 | 0.2321 | 0.6102 | 0.0190 |
| constant_0.5 | 0.6931 | 0.2500 | 0.5721 | 0.0721 |
| dl_embedding (PyTorch) | 0.6632 | 0.2349 | 0.6052 | — |

`ece` is the Expected Calibration Error — the average gap between a stated probability and
what actually happened. The chart built from it is
[`outputs/fig_calibration.png`](outputs/fig_calibration.png); if the axes look cryptic,
[`docs/CALIBRATION-README.md`](docs/CALIBRATION-README.md) walks through how to read them —
including why `gbm_isotonic` wins on `ece` while *losing* on logloss.

Reads: a from-scratch Elo engine gets 0.647; adding form/rest/head-to-head features to a
logistic model edges it to 0.6464; adding **roster-stability + stand-in** (the per-map
lineups that sat unused) drops it to **0.6377** — the single biggest feature-family win in
the repo, and it was the one thing the Limitations section had promised to try. The
deep-learning variant *loses* to logistic — at this data size the signal is linear-ish in
Elo space, and that honest negative is a finding. Per-regime calibration shows the
aggregate number also hides a tier-3 problem: ECE 0.126 (vs 0.02–0.03 elsewhere).
Bayesian ratings (PyMC Bradley-Terry, `outputs/bayesian_ratings.csv`) quantify what Elo
cannot: per-team uncertainty.

## Method

- **Data:** 9,920 professional CS2 series (2023–2026, 3 tiers) from Kaggle; winners
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

`scripts/build_tier_calibration.py` on the test split (1,811 series). `before` is the shipped
model; the other two columns are the refit applied to the same rows.

| tier | n | ECE before | ECE isotonic | ECE platt | logloss before | logloss isotonic | logloss platt |
|---|---|---|---|---|---|---|---|
| tier1 | 643 | 0.0202 | 0.0411 ✗ | 0.0326 ✗ | 0.6227 | 0.6242 | 0.6228 |
| tier2 | 926 | 0.0344 | 0.0322 | **0.0229** | 0.6522 | 0.6525 | **0.6517** |
| tier3 | 242 | 0.0788 | 0.0616 | **0.0669** | 0.6220 | 0.7517 ✗ | **0.6199** |
| all | 1811 | 0.0185 | 0.0317 | 0.0233 | 0.6377 | 0.6557 | **0.6372** |

**Verdict.** Per-tier *isotonic* is rejected: it buys tier-3 ECE by wrecking tier-3 logloss
(0.622 → 0.752) — the map fits bin edges, not signal. Per-tier **Platt** fixes the tiers that
actually needed it (tier-3 ECE −15% *with* better logloss; tier-2 better on both) but costs
tier-1 ECE, so the aggregate ECE rises while the aggregate logloss improves slightly. None of
this is enabled in serving by default. The actionable finding is the one already in
`docs/DECISIONS.md`: **tier-3 needs more data** (529 train / 242 test series), not a fancier map.
A train-only "calibrate only the tiers that look miscalibrated" rule was tested too and cannot
discriminate here — pooled train ECE 0.0066 vs 0.0124 / 0.0220 / 0.0325 per tier, because the
tiers' miscalibrations partly cancel in the mixture — so it degenerates to plain Platt.

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

The trained artifact is **`lr-roster-2026-09-12`**: the exact `lr+roster` model
from the Results table (logloss 0.6377), including the roster features. The
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
  so its +0.1295 DiD is a diagnostic and **not a publishable causal claim** until the
  identification strategy is redesigned.
- [`r/04_rating_forecast.R`](r/04_rating_forecast.R) — Holt's linear trend vs a naive
  random walk on monthly Elo: the random walk wins for all 10 teams (ratings are
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