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
