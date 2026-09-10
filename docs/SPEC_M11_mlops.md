# M11 BUILD SPEC — MLOps: train → serve → monitor  · `MLE DE`

**Install:** `pip install fastapi uvicorn joblib mlflow httpx` (`httpx` powers TestClient).
**No Docker on this laptop** — deployment target is Render or a HuggingFace Space (both
already in your stack). CI does the "build" work instead of a local image.

---

## §1 `src/cs2analytics/models/train.py` (yours) — the artifact producer

`python -m cs2analytics.models.train` must:
1. Build the M7 feature matrix, fit the LR pipeline on TRAIN only, evaluate on TEST,
   and print the 5-number summary (logloss, brier, acc, ece, n_test).
2. Save to `artifacts/` (create, git-ignored except a `.gitkeep`):
   - `model.pkl` (joblib) — the fitted pipeline
   - `features.json` — ordered `feature_names`, the cutoff timestamp, and `model_version`
     (e.g. `"lr-2026-09-10"`), plus the test metrics
   - `elo_ratings.json` — final Elo per team from `run_elo_backtest` (for the live API)
3. Log to MLflow (local `mlruns/`, git-ignored): params (model type, cutoff, n_train),
   metrics (logloss/brier/acc/ece), and the model artifact. One run per invocation.
4. Deterministic: fixed seeds, and `features.json` must be byte-identical across re-runs
   (no dict ordering surprises — sort keys).

## §2 `src/cs2analytics/serve/app.py` (yours)

FastAPI, two endpoints:
- `GET /health` → `{"status": "ok", "model_version": "..."}`
- `POST /predict` body `{"team1": str, "team2": str, "best_of": int}` →
  `{"p_team1": float, "model_version": str, "elo_t1": float, "elo_t2": float, "n_train": int}`
  - Loads model + features.json + elo_ratings.json at startup (module-level lazy cache).
  - Unknown team → Elo 1500 (documented fallback), and `"unknown_team": [names]` in the response.
  - `best_of` → `is_bo1` feature consistently with training (Bo1 ⇔ best_of == 1).
  - Reject `best_of` not in {1,3,5} with HTTP 422.

## §3 `src/cs2analytics/serve/monitor.py` (yours)
```python
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float
def drift_report(...) -> pd.DataFrame   # per-month Elo-gap PSI vs training distribution
```
Writes `outputs/m11_drift.csv`. The README gets one sentence: "PSI on the Elo-gap feature
flags when the team landscape has shifted enough that the model needs refitting."

## §4 `tests/test_m11_serve.py` (CI-runs the unit parts; API tests skip without artifacts)
- PSI: identical distributions → ≈ 0; shifted distributions → > 0.25
- `predict` returns `0 < p_team1 < 1`; symmetry: `predict(a,b) + predict(b,a) ≈ 1` within
  0.02 (Elo asymmetry is allowed only via form/rest features — document any deviation)
- `best_of=2` → 422
- `/health` reports the version string from features.json
- unknown team → 1500 fallback + flag in the response

## §5 CI + deploy
- Extend `.github/workflows/ci.yml` with a step that runs `python -m cs2analytics.models.train`
  **only if** the data is present (it isn't in CI) — so instead: add a step that runs the
  pure-function tests and skips. Concretely: no workflow change needed; the existing
  skip-guards cover it. Add a `# M11` note in DECISIONS.md explaining that.
- Deploy: `uvicorn cs2analytics.serve.app:app` locally, then Render (Web Service) or a
  HF Space with a `Dockerfile`-free `app.py` + `requirements.txt`. Put the live URL in the
  README + a screenshot in `docs/`.

## Checkpoint
1. What breaks in production if you skip monitoring? Give a CS2-specific answer (roster
   changes, meta patches, tier-3 noise leaking into features).
2. Why is `features.json` versioned with the model and not re-derived at serve time?
3. Your API returns p=0.63 for NAVI vs FaZe. Name the three things a bettor would ask next.

## Appendix — hints
- PSI: `sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))` with epsilon guards
  for zero bins.
- FastAPI TestClient: `from fastapi.testclient import TestClient; c = TestClient(app)`.
- If joblib + FastAPI pickling fights your sklearn version, pin sklearn in requirements.txt
  and say so in the README (a classic real-world MLOps note).
