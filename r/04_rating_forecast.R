# 04_rating_forecast.R — M13 §1: Holt's linear trend on monthly team Elo ratings.
# Run from repo root:  "C:/Program Files/R/R-4.6.1/bin/Rscript.exe" r/04_rating_forecast.R
#
# ±80% band assumption: forecast residuals are approximately normal, so the band is
# forecast ± 1.28 * residual_sd (the two-sided 80% normal quantile). predict.HoltWinters
# gives intervals directly; they are cross-checked against the manual normal approx.
# Naive baseline: random walk (forecast = last observed value); its MAPE is reported
# next to Holt's so the comparison is real, not implied.
#
# Implementation note: months are handled as plain consecutive vectors per team;
# ts() wrappers are used only where HoltWinters/predict need them.

library(tidyverse)

monthly <- read_csv(
  "outputs/elo_monthly.csv",
  col_types = cols(
    team = col_character(), month = col_character(),
    datetime = col_datetime(), elo = col_double()
  )
)
stopifnot(nrow(monthly) >= 250)

counts <- monthly |> count(team)
monthly <- monthly |> filter(team %in% counts$team[counts$n >= 24])
cat("teams with >= 24 monthly observations:", nrow(distinct(monthly, team)), "of", nrow(counts), "\n")
teams <- sort(unique(monthly$team))
stopifnot(length(teams) >= 10)

h <- 3  # forecast horizon (months)
forecast_rows <- list()
metric_rows <- list()
plot_rows <- list()

for (tm in teams) {
  df <- monthly |> filter(team == tm) |> arrange(month)
  n <- nrow(df)
  if (n < 24 + h) {
    cat(tm, ": only", n, "months — skipping (need >= 24 train + 3 held out)\n")
    next
  }

  ym <- as.integer(strsplit(df$month[1], "-")[[1]])
  x_all <- df$elo

  # ---- backtest: hold out the last 3 months --------------------------------
  x_tr <- x_all[1:(n - h)]
  actual_bt <- x_all[(n - h + 1):n]
  ym_tr <- as.integer(strsplit(df$month[n - h], "-")[[1]])
  ts_tr <- ts(x_tr, start = ym_tr, frequency = 12)
  hw_bt <- HoltWinters(ts_tr, beta = TRUE, gamma = FALSE)
  fc_bt <- predict(hw_bt, n.ahead = h, prediction.interval = TRUE, level = 0.8)
  mape_holt <- mean(abs((actual_bt - as.numeric(fc_bt[, "fit"])) / actual_bt)) * 100
  naive_pred <- rep(x_tr[length(x_tr)], h)
  mape_naive <- mean(abs((actual_bt - naive_pred) / actual_bt)) * 100
  metric_rows[[length(metric_rows) + 1]] <- tibble(team = tm, mape = mape_holt, n_months = h)
  metric_rows[[length(metric_rows) + 1]] <- tibble(team = "naive", mape = mape_naive, n_months = h)

  # ---- final fit on ALL months + 3-month forecast --------------------------
  ts_all <- ts(x_all, start = ym, frequency = 12)
  hw <- HoltWinters(ts_all, beta = TRUE, gamma = FALSE)
  fit_vals <- as.numeric(fitted(hw)[, "xhat"])
  resid_sd <- sd(as.numeric(residuals(hw)))
  fc <- predict(hw, n.ahead = h, prediction.interval = TRUE, level = 0.8)
  fc_point <- as.numeric(fc[, "fit"])
  lo80 <- as.numeric(fc[, "lwr"])
  hi80 <- as.numeric(fc[, "upr"])
  # manual normal-approximation cross-check for the band
  lo_manual <- fc_point - 1.28 * resid_sd
  hi_manual <- fc_point + 1.28 * resid_sd

  # forecast month labels: the h calendar months after the last observed one
  fc_dates <- seq(as.Date(paste0(df$month[n], "-01")) %m+% months(1), length.out = h, by = "month")
  fc_labels <- sprintf("%04d-%02d", year(fc_dates), month(fc_dates))

  plot_rows[[length(plot_rows) + 1]] <- tibble(
    team = tm,
    month = df$month,
    actual = x_all,
    fitted = c(NA_real_, fit_vals)[1:n]
  )

  forecast_rows[[length(forecast_rows) + 1]] <- tibble(
    team = tm,
    month = fc_labels,
    actual = NA_real_,
    fitted = NA_real_,
    forecast = fc_point,
    lo80 = ifelse(is.finite(lo80), lo80, lo_manual),
    hi80 = ifelse(is.finite(hi80), hi80, hi_manual)
  )
}

plot_df <- bind_rows(plot_rows)
fc_table <- bind_rows(forecast_rows) |>
  select(team, month, actual, fitted, forecast, lo80, hi80)
metrics_df <- bind_rows(metric_rows)

write_csv(fc_table, "outputs/rating_forecast.csv")
write_csv(metrics_df, "outputs/rating_forecast_metrics.csv")

# ---- figures -----------------------------------------------------------------
# one-team figure: the team with the BEST (lowest) backtest MAPE
one_team <- metrics_df |>
  filter(team != "naive") |>
  arrange(mape) |>
  slice(1) |>
  pull(team)
best_df <- plot_df |> filter(team == one_team)
fc_one <- fc_table |> filter(team == one_team)

p1 <- ggplot() +
  geom_ribbon(
    data = fc_one,
    aes(x = month, ymin = lo80, ymax = hi80, group = 1),
    fill = "steelblue", alpha = 0.2
  ) +
  geom_line(data = best_df, aes(x = month, y = actual, group = 1, color = "actual"), linewidth = 0.8) +
  geom_line(data = best_df |> filter(!is.na(fitted)),
            aes(x = month, y = fitted, group = 1, color = "fitted"),
            linetype = "dashed", linewidth = 0.6) +
  geom_line(data = fc_one, aes(x = month, y = forecast, group = 1, color = "forecast"), linewidth = 0.9) +
  labs(
    title = paste0("Monthly Elo — ", one_team, " (best Holt backtest MAPE)"),
    subtitle = "80% forecast band shaded; Holt's linear trend, gamma = FALSE",
    x = NULL, y = "Elo rating", color = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("outputs/fig_rating_forecast.png", p1, width = 9, height = 4.5, dpi = 150)

# facet version: all teams, actual (solid) + fitted (dashed)
p2 <- ggplot(plot_df, aes(x = month, group = team)) +
  geom_line(aes(y = actual, color = "actual"), alpha = 0.8, linewidth = 0.5) +
  geom_line(
    data = plot_df |> filter(!is.na(fitted)),
    aes(x = month, y = fitted, group = team, color = "fitted"),
    linetype = "dashed", alpha = 0.45, linewidth = 0.4
  ) +
  facet_wrap(~team, scales = "free_y") +
  labs(title = "Monthly Elo — all teams (actual solid, Holt fitted dashed)",
       x = NULL, y = "Elo rating") +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_blank())
ggsave("outputs/fig_rating_forecast_facets.png", p2, width = 12, height = 7, dpi = 130)

cat("=== rating_forecast_metrics.csv ===\n")
print(metrics_df, n = 22)
naive_mape <- metrics_df$mape[metrics_df$team == "naive"][1]
holt_mapes <- metrics_df$mape[metrics_df$team != "naive"]
cat("Holt beats the naive random walk for", sum(holt_mapes < naive_mape),
    "of", length(holt_mapes), "teams (naive MAPE:", round(naive_mape, 2), "%)\n")
cat("wrote outputs/rating_forecast.csv, rating_forecast_metrics.csv, fig_rating_forecast.png\n")