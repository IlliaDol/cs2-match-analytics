# M4 BUILD SPEC — Elo Engine + Bayesian Ratings  · [HE] flagship · [RA]

**The heart of the project.** You build a rating model FROM SCRATCH: derive the math in a
notebook, implement it as pure functions, backtest it walk-forward, then rebuild the same
idea Bayesian in PyMC. Every number below is computed from OUR data (9,922 series,
`outputs/series_clean.csv`), so there is zero room for tutorial-copying.

**Files you create:**
- `notebooks/elo_derivation.ipynb` — the math (see §1)
- `src/cs2analytics/features/elo.py` — pure functions (see §2)
- `src/cs2analytics/models/backtest.py` — walk-forward backtest engine (see §3)
- `notebooks/m4_backtest_report.ipynb` — runs the backtest, produces the results table
- `src/cs2analytics/models/bayes.py` + `notebooks/m4_bayesian_ratings.ipynb` — PyMC (see §4)

**Tests:** `tests/test_elo.py` (runs in CI — pure math, no data needed)
**Contract tests for the backtest are in `tests/test_elo_backtest.py` (local, data-gated)**

---

## §1 `notebooks/elo_derivation.ipynb` — the math, by hand (Helsing signal #1)

Cells must contain, IN THIS ORDER, with real LaTeX (`\(...\)` delimiters):

1. **The logistic assumption.** Assume win probability depends only on the rating
   difference \(d = R_A - R_B\): \(P(A\ \text{beats}\ B) = \sigma(\beta d)\). State WHY
   (only-transitivity + monotonicity → logistic is the canonical choice; reference
   Bradley–Terry).
2. **Derive the Elo expected-score formula** from that assumption by rescaling:
   \(E_A = \dfrac{1}{1+10^{(R_B-R_A)/400}}\). Show the algebra: 400 = ln(10)/β·... —
   you derive which β gives the 400-point convention. NO copying — derive.
3. **Worked example with OUR numbers:** R_A=1650, R_B=1750 → E_A = 1/(1+10^{100/400})
   = 1/(1+10^{0.25}) ≈ 0.360. NAVI win, K=32: R_A' = 1650 + 32(1−0.360) ≈ 1670.5.
4. **Update rule as gradient step.** Show \(R' = R + K(S - E)\) is one step of SGD on the
   log-loss \(-[S\ln E + (1-S)\ln(1-E)]\). Compute the derivative \(d\mathcal{L}/dR_A\)
   by hand and show K = β·lr·... equivalence. This is THE derivation that makes you
   sound like a modeler, not a copier.
5. **K as learning rate:** plot (matplotlib) the rating trajectory of a team with true
   strength 1700 that starts at 1500, for K ∈ {8, 16, 32, 64} — 200 simulated games vs
   average opposition. Mark the "convergence window".
6. **Scale equivalence note:** σ(x) with x=(ΔR/400)·ln10 vs σ(Δs/173) — show
   400/ln(10) ≈ 173.7 and write the sentence "Elo is Bradley–Terry on a log10 scale".

**Test (`tests/test_elo.py`, runs in CI):** the notebook file exists AND contains the
required formulas as raw LaTeX strings (grep-level checks on the .ipynb JSON), plus the
pure-function tests from §2. That's the mechanical proof you did the derivation.

## §2 `src/cs2analytics/features/elo.py` — pure functions (TDD: tests exist FIRST)

```python
def expected_score(ra: float, rb: float) -> float
def update_rating(ra: float, rb: float, result: float, k: float = 32.0) -> float
def bo3_win_probability(p_map: float) -> float     # p^2 * (3 - 2p), from spec derivation
def run_elo_backtest(df, k: float, base: float = 1500.0, map_specific: bool = False) -> pd.DataFrame
```

Contract (ALL tested in `tests/test_elo.py` with EXACT numbers — no data needed):
- `expected_score(1500, 1500) == 0.5` (exactly)
- `expected_score(1650, 1750)` ≈ 0.359935 (assert `pytest.approx(..., abs=1e-6)`)
- `expected_score(2800, 1500) > 0.99` (dominant champ vs newcomer)
- `update_rating(1650, 1750, result=1.0, k=32)` ≈ 1670.48 (winner)
- `update_rating(1650, 1750, result=0.0, k=32)` ≈ 1629.52 (loser symmetric)
- `bo3_win_probability(0.5) == 0.5` exactly; `bo3_win_probability(0.55)` ≈ 0.575
  (verify by the binomial formula p²(3−2p)); `bo3_win_probability(0.0) == 0.0`;
  `bo3_win_probability(1.0) == 1.0`
- Symmetry law: `expected_score(a, b) == 1 - expected_score(b, a)` for a grid of values.
- Monotonicity law: expected_score strictly increasing in (ra − rb).
- `run_elo_backtest` on the toy 10-row dataset: returns per-match rows with pre-match
  ratings + predictions, no NaN, ratings updated chronologically (assert on the toy
  fixture, which is tracked in git — CI runs this).

Rules: pure functions only (no I/O in this module), type hints, docstrings with the
formula in LaTeX, no global state. The backtest function must process matches in
datetime order and record PRE-match ratings (leakage law).

## §3 `src/cs2analytics/models/backtest.py` — walk-forward evaluation

Time-split law: **train = everything before 2025-08-01, test = on/after.** Fixed constant
in the module: `TIME_SPLIT = pd.Timestamp("2025-08-01")`.

`run_elo_backtest(df, k, ...)` must:
1. sort by `datetime` (stable, mergesort),
2. walk chronologically: for each match, record both teams' pre-match ratings →
   `expected` = expected_score → apply update with the actual `result`
   (1 if winner == team1 else 0),
3. return df with columns `[match_id, datetime, team1, team2, winner, elo_t1_pre,
   elo_t2_pre, p_t1, result, logloss, brier]` (logloss/brier per match vs the true label).

Metrics in `src/cs2analytics/evaluation/metrics.py`:
```python
def log_loss(p, y) -> float       # mean of -(y ln p + (1-y) ln(1-p)), clip p to [1e-15, 1-1e-15]
def brier_score(p, y) -> float    # mean (p - y)^2
def accuracy_from_probs(p, threshold=0.5) -> float
```

**Tested with exact values** (`tests/test_elo.py`): `log_loss([0.75],[1])` ≈ 0.2877,
`log_loss([0.75],[0])` ≈ 1.3863, `brier_score([0.75],[1])` == 0.0625.

The backtest report (`notebooks/m4_backtest_report.ipynb`) must produce a results table:

| model | split | logloss | brier | acc |
|---|---|---|---|---|
| elo_k32 | train | ... | ... | ... |
| elo_k32 | test | ... | ... | ... |
| elo_k16 | test | ... | ... | ... |
| elo_k8 | test | ... | ... | ... |
| elo_mapspecific_k32 | test | ... | ... | ... |
| constant_0.5 | test | 0.6931 | 0.25 | 0.5* |

`constant_0.5` row is REQUIRED — the "always predict 0.5" baseline that makes every
other number interpretable. Elo must beat it on logloss or something is wrong.

Checkpoint (yours, by hand): compute p_t1 for the first test match of the toy dataset
by hand and confirm the engine agrees.

## §4 `src/cs2analytics/models/bayes.py` — Bayesian ratings (PyMC) · [HE]

Notebook `notebooks/m4_bayesian_ratings.ipynb` (PyMC, optional dep — see pyproject):

- Model: strengths \(s_i \sim \mathcal{N}(1500, 350^2)\) on the logit scale;
  \(P(i\ \text{beats}\ j) = \sigma((s_i - s_j)/173)\) (Bradley–Terry). Write down why
  173 (§1's scale note).
- Fit on a **subset** (tier1, 2025 only, teams with ≥ 20 series — keeps sampling fast).
- Posterior: report per-team posterior mean + 94% HDI. Save
  `outputs/bayesian_ratings.csv` with columns `[team, posterior_mean, hdi_3, hdi_97]`.
- Compare: for 200 random test pairs, posterior-P vs Elo-P — correlation plot +
  MAE. One paragraph: WHEN do they disagree? (few-match teams → wide HDIs.)
- The Helsing sentence to write in the notebook: "the posterior IS the Bayesian update
  after each match; Elo is a degenerate online approximation of it."

**Tested (`tests/test_elo_backtest.py`, data-gated):** `outputs/bayesian_ratings.csv`
exists, ≥ 50 teams, HDI_3 < posterior_mean < HDI_97 for all rows.

## Done when
- [ ] `pytest -q` green locally (new tests: ~15 pure-math + 2 data-gated)
- [ ] CI green (pure-math tests run there; data-gated skip)
- [ ] `elo_derivation.ipynb` contains the 6 §1 elements (checked by test)
- [ ] Backtest table shows elo beating constant_0.5 on test logloss
- [ ] Commit: `m4: elo engine + derivation + walk-forward backtest + bayesian ratings (module 4 done)`

## Appendix — hints (after a real attempt only)
- σ(x) = 1/(1+e^(−x)); E_A with ΔR = R_A−R_B: E_A = 1/(1+10^(−ΔR/400)).
- dℒ/dR_A = −(S − E)·ln10/400 → K(S−E) = −lr·dℒ/dR_A with lr = 400K/ln10.
- Bo3: P = p²(3−2p); at p=0.55 → 0.3025·1.9 = 0.5748.
- If your backtest logloss < 0.60 on test: something leaks (you used post-match info).
  Expected ballpark: 0.66–0.69 (logloss of a ~55–60% accurate predictor).
