# Decision Log

One entry per design decision that a reviewer would question. Format: date, decision, why.

| Date | Decision | Why |
|---|---|---|
| 2026-09-10 | src/ layout + hatchling | import-safe packaging, modern standard |
| 2026-09-10 | ruff for lint+format | one tool, zero config fights |
| 2026-09-10 | toy CSV tracked, real data git-ignored | CI must run without local data |
| 2026-09-10 | time-based splits (fixed cutoff) | teams/rosters/meta drift; random splits leak |
| 2026-09-10 | Spark via Colab, not local — Windows + no JVM, and the dataset doesn't justify a cluster; the notebook is the artifact. *(Superseded 2026-09-11: Java 25 Temurin found installed, so M9 runs Spark locally with `PYSPARK_PYTHON` pointed at the repo venv; notebook + timing CSV are the artifacts either way.)* | dataset doesn't justify a cluster; the notebook is the artifact |
| 2026-09-11 | M9 window span rowsBetween(-5, -1) | the spec's rowsBetween(-4, -1) hint is off by one: pandas `rolling(5).mean().shift(1)` covers up to 5 pre-match rows; correctness proof (max |diff| < 1e-9) pins the right span |
| 2026-09-11 | m9 timing kept full-precision | the timing contract asserts speedup == pandas_sec/spark_sec exactly; rounding before dividing breaks the identity |
| 2026-09-11 | M11: CI needs no workflow change | the trainer + API tests are artifact/data-gated and skip in CI by the existing skip-guards; the trainer runs locally where the data lives |
| 2026-09-11 | M11: /predict symmetrizes the model over both team orientations | the fitted scaler carries the training mean of elo_diff (~+23.6, a seeding artifact: team1 wins 55% of rows); serving raw output would make predict(a,b) + predict(b,a) != 1. Averaging over (1,2) and (2,1) restores the symmetry law at zero training cost |
| 2026-09-11 | M11: MLflow sqlite backend (mlflow.db, git-ignored) | mlflow 3.x deprecation mode blocks the legacy filesystem store; sqlite is the sanctioned local backend |
| 2026-09-11 | M11 deploy = local uvicorn; Render/HF Space left as a documented follow-up | no Docker on this machine and no secrets/persistence story needed for the portfolio demo; artifacts load from artifacts/ |
| 2026-09-12 | Per-tier isotonic recalibration rejected | tier-3 ECE improved 0.0788→0.0616 but logloss worsened 0.622→0.752 and tier-1 got worse on both metrics; isotonic with ~8k train rows per tier overfits bin edges (same failure mode as M7's GBM+isotonic negative). Kept as an artifact (`outputs/m11_tier_calibration.csv`, `models/calibrate_tier.py`) — the honest result is "aggregate calibration is fine; per-tier refits need more tier-3 data". |
| 2026-09-14 | Per-tier **Platt** recalibration measured, not enabled | the v1.1 candidate the plan named. Result: tier-3 ECE 0.0788→0.0669 **with** logloss 0.6220→0.6199 (both better), tier-2 better on both — but tier-1 ECE 0.0202→0.0326, so aggregate ECE rises 0.0185→0.0233 while aggregate logloss improves 0.6377→0.6372. A train-only "calibrate only the tiers that look miscalibrated" rule cannot discriminate here (pooled train ECE 0.0066 vs 0.0124/0.0220/0.0325 per tier — the tiers' errors partly cancel), so it degenerates to plain Platt; it stays a tested knob with default `select="all"`. Verdict unchanged: tier-3 needs more data (529 train / 242 test series), not a fancier map. Numbers: `scripts/build_tier_calibration.py`, README table. |
| 2026-10-11 | Retrained + re-measured everything except DL on the Tier A refresh (test 1,811 → 2,943) | same cutoff, same code paths; GBM caught up (0.6349 ≈ lr+roster 0.6342), isotonic rejected harder, dl row labeled stale (no torch here) |
| 2026-09-12 | Veto/pick sequencing model is impossible with this schema | the 98-column raw schema has NO veto/pick/side columns (verified by column scan) — a map-veto model needs external pick/ban data, documented as out of scope rather than faked. |
