# M3 BUILD SPEC — Statistical Computing with R  · [LMU] evidence module

**Goal:** one R script that takes the series table through a tidyverse pipeline and answers
three real inferential questions about CS2 match data — t-test, ANOVA, proportion test —
plus two ggplot2 figures. Everything runs against the exported `data/interim/matches_series.csv`
(9,922 series, tier-tagged, produced by the Python loader).

**You write the R. The tests (test_r_inference.py) verify your outputs mechanically.**
Hints in Appendix — attempt first, 30-min rule applies.

---

## File 1: `r/01_wrangle.R`

Tidyverse pipeline, top-to-bottom, no functions needed:

1. `library(tidyverse)` — read `data/interim/matches_series.csv` with `read_csv()`
   (datetime parses automatically as `<dttm>`; verify with `glimpse()`).
2. **Feature engineering (the wrangle core):**
   - `total_maps = t1_series_score + t2_series_score`
   - `margin = abs(t1_series_score - t2_series_score)` (domination measure)
   - `is_bo1 = games_played == 1`, `is_upset` — you define it ONE way, documented:
     suggest: the winner is the team that did NOT have the better pre-series score? No —
     there's no pre-series rating yet (that's M4). **Use `bestOf` upsets instead:** a
     "seeded upset" = Bo5 that ended in minimum games (3–0) — flag `bo5_sweep`.
   - `decisive = margin == 2 | (games_played == 5 & margin == 1)` → sweep flag:
     `swept = (t1_series_score == 0 | t2_series_score == 0) & games_played >= 2`
3. **Tier filter**: keep all tiers but compute per-tier summaries; tier is a factor with
   levels tier1 > tier2 > tier3.
4. Write `outputs/series_clean.csv` — the wrangled table every later module reads.
5. Print: n rows, n teams, date range, Bo1 share (expect ~20%, DATA.md Quirk 2).

**Contract (tested):** `outputs/series_clean.csv` exists, has all input columns plus
`total_maps`, `margin`, `is_bo1`, `swept`, `bo5_sweep`; nrow == 9922; no NA in
`total_maps`/`margin`; `is_bo1 == (games_played == 1)` for all rows.

## File 2: `r/02_inference.R` — the three tests (LMU-test core)

Each test = hypothesis stated in a comment → run → effect size → one-sentence interpretation
printed to console with `cat()`. Numbers land in `outputs/inference_results.csv`.

### Test A — two-sample t-test (Welch): does a stand-in matter? No lineup data here —
instead: **Do Bo1s differ from Bo3+ series in decisiveness?**
- Groups: `margin` for `is_bo1 == TRUE` vs `is_bo1 == FALSE` (exclude Bo2 draws: margin==0 & total_maps==2)
- `t.test(margin ~ is_bo1, data = ...)` — Welch is default, good (unequal n: 2020 vs ~7900).
- Effect size: Cohen's d by hand from the t output: d = t * sqrt(1/n1 + 1/n2).
- Report means, CI of difference, p-value, d.

### Test B — one-way ANOVA + Tukey HSD: does margin differ by tier?
- `aov(margin ~ tier, data = ...)`; `summary()`; `TukeyHSD()`.
- Check assumptions in-script: `plot(ao, which = 2)` residuals QQ normality comment;
  Levene-style sanity via `car::leveneTest` is OPTIONAL — skip if `car` not installed,
  note it in the interpretation instead.
- If ANOVA's F is significant, Tukey tells you WHICH tiers differ. Tier3 has the most
  forfeits/forfeit-noise → expect tier3 margin to be the outlier.

### Test C — prop.test: is P(Bo1 | 2-0 sweep) different from 50%?
Actually cleaner with this data: **among decided non-Bo1 series, what share sweeps (2-0)?**
- x = n(swept), n = n(decided non-Bo1 series), `prop.test(x, n, p = 0.5)`.
- Hand-compute the 95% Wald CI FIRST in a comment (p̂ = x/n, SE = sqrt(p(1-p)/n)),
  then let R produce its interval — the numbers differ (Wilson vs Wald), say why.

**Contract (tested):** `outputs/inference_results.csv` exists with columns
`test, statistic, p_value, effect_size, interpretation`; 3 rows; p-values in [0,1];
each `interpretation` is non-empty.

## File 3: `r/03_plots.R` — two ggplot2 figures
1. `outputs/fig_winshare_top15.png`: top-15 teams by series win share (min 20 series),
   horizontal bar chart, sorted, fill = tier of their most frequent tier (or overall mean
   margin annotation). Titles/labels in English.
2. `outputs/fig_margin_by_tier.png`: boxplot of margin by tier with the group means as
   points — visual companion to Test B.

**Contract (tested):** both PNG files exist, each > 20 KB (real renders, not blanks).

## Rules
- R 4.6.1 at `C:/Program Files/R/R-4.6.1/bin/Rscript.exe` (verified). Run scripts from repo root:
  `"C:/Program Files/R/R-4.6.1/bin/Rscript.exe" r/01_wrangle.R`
- Packages: only `tidyverse`. If a package is missing, `install.packages("tidyverse")` once.
- Set the working directory INSIDE each script defensively:
  `setwd()` is banned; use `here`-style relative paths from repo root — scripts are run from root.
- No randomness anywhere → no seeds needed. No loops over dataframes → vectorized dplyr only.
- Every test interpretation must be ONE sentence, plain language, printed via `cat()`.
  This is the LMU-test muscle: state → test → interpret.

## Checkpoint (do these BEFORE looking at Appendix)
1. By hand: p̂ = 0.68, n = 412 → 95% Wald CI. Then compare with prop.test's interval.
2. Explain why Welch (default) rather than Student's t for Test A.
3. ANOVA assumptions — which one does the tier boxplot let you eyeball immediately?

## Appendix — hints (after a genuine attempt)
- Wald: SE = sqrt(0.68·0.32/412) = 0.0230 → [0.635, 0.725]. prop.test uses Wilson — different
  because Wilson borrows strength from the null/normal approximation differently near edges.
- Cohen's d from t: d = t·sqrt(n1+n2)/(sqrt(n1·n2)) — equivalent form.
- Bo1 share in this data: 2020/9922 ≈ 20.4% (DATA.md Quirk 2) — if your is_bo1 count
  differs, you filtered wrong.
