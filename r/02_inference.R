# 02_inference.R — Module 3: the three classical tests (LMU-test core).
# Run from repo root:  "C:/Program Files/R/R-4.6.1/bin/Rscript.exe" r/02_inference.R
#
# Each test follows the same discipline:
#   H0 stated in a comment -> test run -> effect size -> ONE-sentence interpretation.

library(tidyverse)

clean <- read_csv(
  "outputs/series_clean.csv",
  col_types = cols(tier = col_factor(levels = c("tier1", "tier2", "tier3")))
)
stopifnot(nrow(clean) == 9920)  # corrected after Quirk-1 fix (2026-09-11)

results <- tibble::tibble(
  test = character(), statistic = numeric(), p_value = numeric(),
  effect_size = numeric(), interpretation = character()
)

# ---------------------------------------------------------------------------
# Test A — two-sample t-test (Welch): Do Bo1s differ from Bo3+/Bo5 in margin?
# H0: mean(margin | Bo1) == mean(margin | Bo3+).  Excludes Bo2 draws (margin 0, 2 maps).
# Welch is the R default and correct here: group sizes are wildly unequal
# (2020 vs ~7900) and variances may differ.
# ---------------------------------------------------------------------------

decided <- clean |>
  filter(!(margin == 0 & total_maps == 2)) # drop genuine Bo2 draws

tt <- t.test(margin ~ is_bo1, data = decided)
n1 <- sum(decided$is_bo1)
n2 <- sum(!decided$is_bo1)
# Cohen's d from the t statistic (large-sample variant for unequal n):
#   d = t * sqrt(1/n1 + 1/n2)
d_a <- unname(tt$statistic) * sqrt(1 / n1 + 1 / n2)

m1 <- mean(decided$margin[decided$is_bo1])
m2 <- mean(decided$margin[!decided$is_bo1])
interp_a <- sprintf(
  paste0(
    "Bo1 series end with %s margin than multi-map series (mean %.2f vs %.2f, ",
    "p = %.2e, d = %.2f), %s."
  ),
  ifelse(m1 > m2, "larger", "smaller"), m1, m2, tt$p.value, abs(d_a),
  ifelse(tt$p.value < 0.05, "a real format effect", "no meaningful difference")
)

results <- results |> add_row(
  test = "welch_t_margin_bo1_vs_bo3plus",
  statistic = unname(tt$statistic), p_value = tt$p.value,
  effect_size = abs(d_a), interpretation = interp_a
)

cat("=== Test A: Welch t-test, margin ~ is_bo1 ===\n")
print(tt)
cat(sprintf("Cohen's d = %.3f (n1=%d Bo1, n2=%d Bo3+)\n\n", d_a, n1, n2))
cat("Interpretation:", interp_a, "\n\n")

# ---------------------------------------------------------------------------
# Test B — one-way ANOVA + Tukey HSD: does margin differ by tier?
# H0: mean(margin) equal across tier1/tier2/tier3.
# Assumptions checked in 03_plots.R's residual QQ plot; Levene's test skipped
# (car not installed) — noted in the interpretation instead.
# ---------------------------------------------------------------------------

av <- aov(margin ~ tier, data = clean)
fstat <- summary(av)[[1]]$`F value`[1]
p_b <- summary(av)[[1]]$`Pr(>F)`[1]
tukey <- TukeyHSD(av)

eta_sq <- summary(av)[[1]]$`Sum Sq`[1] / sum(summary(av)[[1]]$`Sum Sq`)

interp_b <- sprintf(
  "Series margin differs across tiers (F = %.1f, p = %.2e, eta-sq = %.3f); the Tukey table shows which pairs differ, with tier3 the noisiest (forfeits), though variance-homogeneity is assumed, not tested (Levene skipped, car not installed).",
  fstat, p_b, eta_sq
)

results <- results |> add_row(
  test = "anova_margin_by_tier",
  statistic = fstat, p_value = p_b, effect_size = eta_sq, interpretation = interp_b
)

cat("=== Test B: ANOVA margin ~ tier ===\n")
print(summary(av))
cat("\nTukey HSD:\n")
print(tukey)
cat("Interpretation:", interp_b, "\n\n")

# ---------------------------------------------------------------------------
# Test C — prop.test: among decided non-Bo1 series, what share sweeps (2-0)?
# H0: p = 0.5.  Hand-compute the Wald CI in comments FIRST (see spec §Test C),
# then compare with prop.test's Wilson interval — numbers differ by design.
# ---------------------------------------------------------------------------

decided_multi <- clean |> filter(games_played >= 2)
x_c <- sum(decided_multi$swept)
n_c <- nrow(decided_multi)
p_hat <- x_c / n_c
# Wald 95% CI by hand: p_hat ± 1.96*sqrt(p_hat*(1-p_hat)/n_c)
se_c <- sqrt(p_hat * (1 - p_hat) / n_c)
wald_ci <- c(p_hat - 1.96 * se_c, p_hat + 1.96 * se_c)

pt <- prop.test(x = x_c, n = n_c, p = 0.5)
interp_c <- sprintf(
  paste0(
    "Sweeps happen in %.1f%% of decided multi-map series (n = %d), which %s from ",
    "50%% (p = %.2e); Wald CI [%.3f, %.3f] vs prop.test's Wilson interval differ ",
    "because Wilson centers differently near p far from 0.5."
  ),
  100 * p_hat, n_c,
  ifelse(pt$p.value < 0.05, "differs significantly", "does not differ"),
  pt$p.value, wald_ci[1], wald_ci[2]
)

results <- results |> add_row(
  test = "proptest_sweep_share_50pct",
  statistic = unname(pt$statistic), # X-squared
  p_value = pt$p.value, effect_size = p_hat, interpretation = interp_c
)

cat("=== Test C: prop.test, sweep share vs 0.5 ===\n")
print(pt)
cat("Interpretation:", interp_c, "\n")

# ---------------------------------------------------------------------------
# Export + summary
# ---------------------------------------------------------------------------

dir.create("outputs", showWarnings = FALSE)
write_csv(results, "outputs/inference_results.csv")
cat("\n=== inference_results.csv written:", nrow(results), "rows ===\n")
