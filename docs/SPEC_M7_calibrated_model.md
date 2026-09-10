# M7 BUILD SPEC — Statistical Machine Learning: the calibrated model  · [RA] flagship

**This is the module the repo exists for.** A logistic/GBM model with honest time-based
validation, a calibration story, and the leakage demo. Everything runs on
`outputs/features_v1.parquet` (M6) + `outputs/series_clean.csv` (M3).

**Install:** `pip install scikit-learn matplotlib shap` (shap optional).

---

## §1 `src/cs2analytics/features/matrix.py` (yours)

```python
def build_feature_matrix(
    features_path,
    extra_features: list[str] | None = None,   # appended to the default feature set
    cutoff: pd.Timestamp = pd.Timestamp("2026-01-01"),
) -> FeatureSet
```

`FeatureSet`: dataclass with `X`, `y`, `feature_names`, `dates`, `split` (Series of
"train"/"test"), and `cutoff` (Timestamp).

- Default features (ALL pre-match, leakage law): `elo_diff` (elo_t1_pre − elo_t2_pre),
  `form5_diff`, `rest_days_diff`, `is_bo1` (games_played==1), `tier` one-hot.
- `extra_features` lets a caller append columns — and the guard below is what makes that
  parameter dangerous by design (see the test).
- **NEVER** use `games_played`, `margin`, `swept`, or any post-outcome column.
  (They're literally in the same table — this is the trap the tests check.)
- Time split: `cutoff` → train = strictly before, test = on/after.
- Raises `LeakageError` (a ValueError subclass) if any name in `feature_names` — default
  set or `extra_features` — is on the forbidden list. The check runs against the FINAL
  feature list, so `extra_features=["games_played"]` must raise.

## §2 `src/cs2analytics/models/logistic.py` + `models/gbm.py` (yours)

```python
def fit_logistic(X_tr, y_tr) -> Pipeline          # StandardScaler + LogisticRegression
def fit_gbm(X_tr, y_tr) -> Pipeline               # HistGradientBoostingClassifier
def predict_proba(model, X) -> np.ndarray         # column of P(y=1)
```

Rules: no tuning on test; if you tune, use `TimeSeriesSplit` on train only. Fixed
`random_state=42` everywhere.

## §3 `src/cs2analytics/evaluation/calibration.py` (yours)

```python
def reliability_table(y, p, bins=10) -> pd.DataFrame   # [bin_lo, bin_hi, n, mean_pred, obs_rate]
def expected_calibration_error(y, p, bins=10) -> float
def brier_decomposition(y, p, bins=10) -> dict          # reliability, resolution, uncertainty
```

Brier decomposition identity to satisfy: `brier ≈ reliability − resolution + uncertainty`.

## §4 `notebooks/03_model_experiments.ipynb` — the report (yours)

**Required contents (checked by tests):**
1. **The money chart** → `outputs/fig_calibration.png`: reliability diagram, 3 series
   (LR, GBM, GBM+isotonic) + the 45° line + `constant_0.5` baseline. Save at dpi=150.
2. **Leakage demo table** → `outputs/m7_leakage_demo.csv`: rows
   `[model, split_mode, logloss, brier, acc]` for {lr, gbm} × {time, random}.
   The random-split numbers WILL look better. Explain in a markdown cell HOW MUCH better
   and why (team-level memorization: the same team appears in both sets → the model
   learns team identity, not team strength).
3. Isolation: `CalibratedClassifierCV(gbm, method="isotonic", cv=TimeSeriesSplit(5))` —
   report the logloss change. If isotonic HURTS, say so (honest negative).
4. Feature importance: permutation importance (sklearn) on the GBM. Expect `elo_diff` on
   top — state the ranking in text.
5. **Comparison table** → `outputs/m7_model_comparison.csv` with rows for
   `constant_0.5`, `elo_k32` (reuse M4's number), `lr`, `gbm`, `gbm_isotonic` and columns
   `[model, logloss, brier, acc, ece]` — the single table M8/M11/M12 all extend.

## §5 Tests (`tests/test_m7_model.py`, data-gated, CI skips)
- `build_feature_matrix` raises on a forbidden column; no NaN in X; split obeys the cutoff
  (train max date < cutoff ≤ test min date)
- `reliability_table`: bins sum to n; `mean_pred` monotone sanity
- ECE of a perfectly calibrated synthetic set (`y` sampled from `p`) < 0.05
- the two CSVs + the PNG exist with the required columns/rows
- `m7_model_comparison.csv`: every model's logloss < 0.6932, and `lr`/`gbm` beat `elo_k32`
  OR the notebook explains why not (assert only "≤ elo + 0.01" to allow honest ties)

## Checkpoint
1. Compute log loss by hand for a 3-match test set: p = [0.7, 0.4, 0.9], y = [1, 0, 1].
2. Why is accuracy the wrong headline here? What does a 0.5-threshold rule buy you when
   the market already prices favourites?
3. State your identifying assumption for the time split in ONE sentence.

## Appendix — hints
- `log_loss([0.7,0.4,0.9], [1,0,1])` = (−ln0.7 + −ln0.6 + −ln0.9)/3 = (0.3567+0.5108+0.1054)/3 ≈ 0.3243
- ECE = Σ (n_b/N)·|obs_rate_b − mean_pred_b|
- If GBM beats LR by <0.005 logloss, LR is the better repo story (interpretability) — say so.
