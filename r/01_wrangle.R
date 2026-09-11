# 01_wrangle.R — Module 3: read the Python loader's series table, engineer features,
# write outputs/series_clean.csv. Run from repo root:
#   "C:/Program Files/R/R-4.6.1/bin/Rscript.exe" r/01_wrangle.R

library(tidyverse)

series <- read_csv(
  "data/interim/matches_series.csv",
  col_types = cols(
    match_id = col_double(),
    datetime = col_datetime(),
    tournament = col_character(),
    team1 = col_character(),
    team2 = col_character(),
    winner = col_character(),
    t1_series_score = col_integer(),
    t2_series_score = col_integer(),
    games_played = col_double(),
    bestOf = col_integer(),
    tier = col_character()
  )
)

stopifnot(nrow(series) == 9920)  # 9,923 raw series - 1 no-teams - 1 undrawable (Quirk 1 fix, 2026-09-11)

wrangled <- series |>
  mutate(
    total_maps = t1_series_score + t2_series_score,
    margin = abs(t1_series_score - t2_series_score),
    is_bo1 = games_played == 1,
    # sweep = decided series with >= 2 maps where the loser won zero maps
    swept = (t1_series_score == 0 | t2_series_score == 0) & games_played >= 2,
    # Bo5 sweep = 3-0 (fastest possible Bo5 win)
    bo5_sweep = bestOf == 5 & margin == 3,
    tier = factor(tier, levels = c("tier1", "tier2", "tier3"))
  )

stopifnot(
  sum(wrangled$total_maps != wrangled$t1_series_score + wrangled$t2_series_score) == 0,
  all(!is.na(wrangled$total_maps)),
  all(!is.na(wrangled$margin))
)

dir.create("outputs", showWarnings = FALSE)
write_csv(wrangled, "outputs/series_clean.csv")

cat("=== 01_wrangle.R summary ===\n")
cat("rows:", nrow(wrangled), "\n")
n_teams <- length(unique(c(wrangled$team1, wrangled$team2)))
cat("teams:", n_teams, "\n")
cat("date range:", format(min(wrangled$datetime)), "->", format(max(wrangled$datetime)), "\n")
cat("Bo1 share:", sprintf("%.1f%%", 100 * mean(wrangled$is_bo1)),
    "(expect ~20.4%, DATA.md Quirk 2)\n")
cat("swept (non-Bo1):", sum(wrangled$swept), "\n")
cat("Bo5 sweeps:", sum(wrangled$bo5_sweep, na.rm = TRUE), "\n")
