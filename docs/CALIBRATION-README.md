# How to read the reliability diagram

`outputs/fig_calibration.png` — the calibration chart this project is built around. Written
because the axes confuse almost everyone the first time: neither one is a score, and neither
one says whether a prediction was "right".

![reliability diagram](../outputs/fig_calibration.png)

## The question it answers

The models output a **probability**, not a verdict: "team1 wins with probability 0.63".
A probability is easy to produce and hard to trust. A reliability diagram asks the only
question that matters about one:

> When the model says 40%, does it actually happen 40% of the time?

Binary outcome here: team1 wins, or team2 wins. 2,943 test matches.

## How each axis is built

1. Score every test match → 2,943 numbers between 0 and 1.
2. Sort them and cut into **10 equally-populated groups** (`reliability_table(..., bins=10)`).
3. For each group compute two things:
   * **X** = the mean of the predicted probabilities in that group — what the model promised.
   * **Y** = the fraction of those matches actually won — what happened.
4. Plot one point per group. Note that each point stands for **~294 matches**, not one.

Worked example from the figure: the orange point at **x ≈ 0.36, y ≈ 0.33** means "in the group
where the model averaged 36%, the team actually won 33%".

## Why both axes run 0 → 1

Because both are probabilities — X is what was *said*, Y is what *occurred*. A flawless model
would have X = Y everywhere, so every point would sit on the diagonal labelled **perfect**:

| where the point sits | what it means |
|---|---|
| on the diagonal | honest: promised 40%, delivered 40% |
| **above** it | the model **under**-estimated — reality beat the promise |
| **below** it | the model **over**-estimated — reality fell short |

The points are ordered left→right by predicted probability, so a sane curve **rises**:
matches it called 20% are won rarely, matches it called 80% are won often. A curve that does
not rise means the probabilities are close to random.

## What the three lines are

| line | model | shape |
|---|---|---|
| **Logistic** (blue) | logistic regression on Elo + form/rest/h2h/roster | tracks the diagonal closely |
| **GBM** (orange) | gradient boosting | close, with visible misses at both ends (x ≈ 0.16 → y ≈ 0.20, n=40; x ≈ 0.84 → y ≈ 0.79) |
| **GBM + isotonic** (green) | the same GBM with a post-hoc isotonic recalibration | close in the middle, thin at the extremes (bottom bin 5 matches at x ≈ 0.17 → y ≈ 0.00; top bin empty) |
| **perfect** | the diagonal itself | a reference, not a model |
| **constant 0.5** | predicting 0.5 for everything | the baseline — a single point at x = 0.5, y = the base win rate |

## The thin extremes, and what they prove

The green line's top bin is **empty** and its bottom bin holds 5 matches
(x ≈ 0.17 → y ≈ 0.00). That is not a plotting bug — it is what isotonic regression
does with 2,943 test points: to stay monotonic it pools the rare, extreme-probability
matches into blocks too small to trust, which is exactly why the per-tier isotonic
refits in the README lose on both metrics and stay rejected.

Re-measured on the refreshed split:

    GBM logloss 0.6491 -> isotonic 0.6529     worse  (lower is better)
    GBM ECE     0.0214 -> isotonic 0.0280     worse

* **logloss** grades the predictions themselves — how good they are.
* **ECE** (Expected Calibration Error) grades their honesty — the average gap from the diagonal.

So on this data isotonic buys neither honesty nor sharpness — a cleaner rejection
than the old snapshot's tradeoff story. An ECE of ≈0.016 (the shipped lr+roster row)
means the stated probability is off by under **2 percentage points** on average,
which is why calibrated uncertainty can be claimed as a real result rather than a hope.

## How to regenerate it

The figure is produced by `notebooks/03_model_experiments.ipynb` (cells using
`cs2analytics.evaluation.calibration.reliability_table`). Run the notebook; it writes
`outputs/fig_calibration.png` at 150 dpi.
