# M13 BUILD SPEC — The Gaps: forecasting + causal  · `DA DS` · [RA] [LMU]

Pick 3 of the roadmap's 7 gaps. For THIS repo the three that pay off:
**time series** (LMU signal), **causal inference** (RA signal), **dashboarding** (already
covered by M12's Streamlit). If you want a different third, take Kaggle's Intro to AI Ethics
and add one paragraph about gambling-adjacent ML to the README's ethics section.

---

## §1 `r/04_rating_forecast.R` — team-rating forecasting (your LMU evidence #2)

Input: `outputs/elo_monthly.csv` (produced by you: monthly mean posterior/Elo rating per
team for the top-10 teams, from the M4 backtest output). Steps:

1. `library(tidyverse)`; read the monthly ratings.
2. For each of the top 10 teams, build a `ts()` object (monthly) and fit
   **Holt's linear trend** (`stats::HoltWinters(x, beta = TRUE, gamma = FALSE)`) — base R,
   no extra packages (fpp3's fable is the textbook but `stats` runs everywhere).
3. Forecast 3 months ahead; write `outputs/rating_forecast.csv` with
   `[team, month, actual, fitted, forecast, lo80, hi80]` where `lo80/hi80` are the
   forecast ±1.28·residual_sd (state the assumption in a comment).
4. One plot: `outputs/fig_rating_forecast.png` — actual line, fitted line, forecast +
   ±80% band for ONE team with a title naming the team, plus a facet version for all 10.
5. **Backtest row:** for each team, hold out the last 3 months, fit on the rest, and report
   MAPE in `outputs/rating_forecast_metrics.csv` `[team, mape, n_months]`. Say in a comment
   whether a random walk would beat you (compute that too: naive forecast = last value).

## §2 `notebooks/07_break_effect_did.ipynb` — the causal question (RA signal)

Question: **does a long competitive break (≥30 days) causally hurt a team's next match?**

- Treatment: team comes into a match after a gap ≥ 30 days (from `days_rest` logic).
- Control: same team (or similar-Elo teams) with gap ≤ 14 days.
- Design: difference-in-differences on win rate with team fixed effects (simplest credible
  version: within-team comparison before/after the break vs before/after a normal gap);
  if you prefer, use a matched control group on Elo (±50).
- Outputs: `outputs/m13_did_results.csv` `[specification, estimate, se, ci_lo, ci_hi, n]`
  with ≥ 2 specifications (e.g. all teams / tier1 only), and a pre-trend plot →
  `outputs/fig_did_pretrend.png`.
- **Required in a markdown cell: the identifying assumption in ONE sentence** (parallel
  trends: absent the break, treated and control would have moved the same). Then: what would
  violate it (roster changes coinciding with the break — say how you'd check).

## §3 README hookup
Add a "Going deeper" section linking both notebooks + the two output tables, one sentence each.

## Tests (`tests/test_m13_gaps.py`, skipif artifacts missing)
- `rating_forecast.csv`: ≥ 10 teams × 3 months, `lo80 < forecast < hi80` for all rows
- `rating_forecast_metrics.csv`: MAPE > 0 and the naive row exists (so the comparison is real)
- `fig_rating_forecast.png` exists, > 20 KB
- `m13_did_results.csv`: ≥ 2 specifications, CI brackets the estimate, `n` > 100
- DiD notebook contains the literal phrase "parallel trends" (the assumption must be stated)

## Checkpoint
1. Why is Holt's trend a reasonable baseline for a rating series but a terrible one for
   match outcomes?
2. DiD: if teams systematically schedule a break right before a major, what does that do to
   your estimate?
3. Which is a bigger threat here: confounding or measurement error in "break"? Why?

## Appendix — hints
- Monthly rating matrix: `pd.pivot_table(df, index="month", columns="team", values="elo")`.
- HoltWinters needs ≥ 2 full periods (24 months) — you have 42; if a team has < 24 months,
  drop it and say so.
- Pre-trend plot: mean win rate of treated vs control in the 6 matches BEFORE the break,
  by match index — if the lines diverge before treatment, DiD is compromised.
