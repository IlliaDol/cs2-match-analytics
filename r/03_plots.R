# 03_plots.R — Module 3: two ggplot2 figures for the inference story.
# Run from repo root:  "C:/Program Files/R/R-4.6.1/bin/Rscript.exe" r/03_plots.R

library(tidyverse)

clean <- read_csv(
  "outputs/series_clean.csv",
  col_types = cols(tier = col_factor(levels = c("tier1", "tier2", "tier3")))
)
stopifnot(nrow(clean) == 9922)

dir.create("outputs", showWarnings = FALSE)

# --- Figure 1: top-15 teams by series win share (min 20 series) --------------

team_stats <- clean |>
  mutate(
    won = winner == team1,
    team = ifelse(won, team1, team2)
  ) |>
  # long format: each match appears once per team (loser side too)
  bind_rows(
    clean |>
      mutate(
        won = winner == team2,
        team = ifelse(won, team2, team1)
      )
  ) |>
  group_by(team) |>
  summarise(
    n_series = n(),
    n_wins = sum(won),
    win_share = mean(won),
    mean_margin = mean(margin),
    .groups = "drop"
  ) |>
  filter(n_series >= 20)

top15 <- team_stats |>
  arrange(desc(win_share)) |>
  head(15)

p1 <- top15 |>
  ggplot(aes(x = win_share, y = reorder(team, win_share))) +
  geom_col(fill = "steelblue", alpha = 0.85) +
  geom_text(aes(label = sprintf("%.0f%% (n=%d)", 100 * win_share, n_series)),
            hjust = -0.15, size = 3) +
  scale_x_continuous(limits = c(0, 0.85), labels = scales::percent) +
  labs(
    title = "Top 15 CS2 Teams by Series Win Share (2023-2026)",
    subtitle = "Minimum 20 series; share = series won / series played",
    x = "Series win share", y = NULL
  ) +
  theme_minimal(base_size = 12)

ggsave("outputs/fig_winshare_top15.png", p1, width = 9, height = 5.5, dpi = 150)

# --- Figure 2: margin by tier (boxplot + group means) -------------------------

tier_means <- clean |>
  group_by(tier) |>
  summarise(mean_margin = mean(margin), .groups = "drop")

p2 <- ggplot(clean, aes(x = tier, y = margin, fill = tier)) +
  geom_boxplot(alpha = 0.6, outlier.alpha = 0.08) +
  geom_point(data = tier_means, aes(x = tier, y = mean_margin),
             color = "black", size = 3, shape = 18) +
  scale_fill_brewer(palette = "Blues") +
  labs(
    title = "Series Margin by Event Tier",
    subtitle = "Boxplot of |score difference|; black dots = group means (Test B companion)",
    x = "Event tier", y = "Series margin (maps)",
    fill = "Tier"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("outputs/fig_margin_by_tier.png", p2, width = 7, height = 5, dpi = 150)

cat("=== 03_plots.R: two figures written to outputs/ ===\n")
