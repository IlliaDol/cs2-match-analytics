# SAUCE INDEX — where every module's spec and tests live

**The deal:** the human writes every implementation; the tests here verify it mechanically.
Specs are dense on purpose (formulas, exact numbers, contracts). Hints are at the bottom of
each spec — attempt first, read hints second.

## Module ladder

| Module | Spec | Test file | Builds (in `src/`) | State |
|---|---|---|---|---|
| M4 Elo + Bayes | `docs/SPEC_M4_elo_engine.md` | `tests/test_elo.py`, `tests/test_elo_backtest.py` | `features/elo.py` ✅, `evaluation/metrics.py` ✅, `models/backtest.py` ✅, `models/bayes.py` ✅, `notebooks/elo_derivation.ipynb` ✅, `notebooks/m4_backtest_report.ipynb` ✅, `notebooks/m4_bayesian_ratings.ipynb` ✅ → `outputs/m4_backtest_results.csv`, `outputs/bayesian_ratings.csv` | **complete** |
| M5 SQL | `docs/SPEC_M5_sql.md` | `tests/test_sql_queries.py` | **yours:** `db/build_db.py`, `db/queries.sql`, `db/run_queries.py` | sauce pushed |
| M6 Wrangling | `docs/SPEC_M6_wrangling.md` | `tests/test_cleaning.py` | **yours:** `cleaning.py`, `features/form.py`, `notebooks/02_eda.ipynb` | sauce pushed |
| M7 Calibrated model | `docs/SPEC_M7_calibrated_model.md` | `tests/test_m7_model.py` | **yours:** `features/matrix.py`, `models/logistic.py`, `models/gbm.py`, `evaluation/calibration.py`, `notebooks/03_model_experiments.ipynb` | sauce pushed |
| M8 Deep learning | `docs/SPEC_M8_deep_learning.md` | `tests/test_m8_net.py` | **yours:** `models/net.py`, `models/train_dl.py`, `notebooks/04_embedding_experiments.ipynb` | sauce pushed |
| M9 Spark | `docs/SPEC_M9_spark.md` | `tests/test_m9_artifacts.py` | **yours:** `notebooks/05_spark_features.ipynb` (Colab) | sauce pushed |
| M10 LLM analyst | `docs/SPEC_M10_llm_analyst.md` | `tests/test_m10_llm.py` | **yours:** `db/gold_questions.csv`, `llm/qa.py`, `scripts/eval_llm.py`, `notebooks/06_llm_analyst.ipynb` | sauce pushed |
| M11 MLOps | `docs/SPEC_M11_mlops.md` | `tests/test_m11_serve.py` | **yours:** `models/train.py`, `serve/app.py`, `serve/monitor.py` | sauce pushed |
| M12 Capstone | `docs/SPEC_M12_capstone.md` | `tests/test_m12_readme.py` | **yours:** `README.md`, `reports/method_note.md`, `serve/dashboard.py`, `docs/presentation.md` | sauce pushed |
| M13 Gaps | `docs/SPEC_M13_gaps.md` | `tests/test_m13_gaps.py` | **yours:** `r/04_rating_forecast.R`, `notebooks/07_break_effect_did.ipynb` | sauce pushed |

## How the skip-guards work

Every not-yet-built module's tests **skip** instead of failing, so CI stays green while you
work module by module. Two guard styles:

- **Module import guard** — `try: import cs2analytics.X except ModuleNotFoundError:
  pytestmark = skip`. Fires the moment you create the module file: tests stop skipping and
  start checking. (M6, M7, M8, M10, M11.)
- **Data/artifact guard** — `skipif not PRIMARY.exists()` / `pytest.skip(...)` inside a
  fixture. CI's runner has no Kaggle data (it's git-ignored) so these skip there and run
  on your machine. (M4 backtest, M5, M9, M12, M13.)

So: `pytest -q` locally tells you exactly which module you're inside.

## Environment upgrades (install per module, not upfront)

```bash
pip install duckdb                        # M5
pip install scikit-learn matplotlib shap  # M7
pip install torch --index-url https://download.pytorch.org/whl/cpu   # M8
pip install openai                        # M10
pip install fastapi uvicorn joblib mlflow httpx  # M11
pip install streamlit                     # M12
```

R: 4.6.1 at `C:/Program Files/R/R-4.6.1/bin/Rscript.exe` (M3 ✅, M13 pending).

## Repo layout additions these specs assume

```
artifacts/     # M11: model.pkl, features.json, elo_ratings.json (git-ignored except .gitkeep)
db/            # M5 queries + M10 gold questions
notebooks/     # M4 elo_derivation, m4_backtest_report, 02_eda, 03_model_experiments, ...
reports/       # M12 method_note.md
scripts/       # M10 eval_llm.py
```
