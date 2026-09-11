-- M5 §2 — five analytical queries against outputs/cs2.duckdb.
-- Each statement sits under a "-- @query: <name>" marker; run_queries.py
-- executes them in order and writes outputs/q<k>.csv.
-- Tables: matches (series table), teams (id/name lookup), team_stats
-- (per-team n_series, n_wins, win_share), m4_preds (Elo pre-match ratings).

-- @query: q1
WITH team_month AS (
    SELECT
        team,
        month,
        AVG(win) AS win_share
    FROM (
        SELECT team1 AS team, (winner = team1)::INT AS win,
               date_trunc('month', datetime) AS month
        FROM matches
        UNION ALL
        SELECT team2 AS team, (winner = team2)::INT AS win,
               date_trunc('month', datetime) AS month
        FROM matches
    )
    GROUP BY team, month
),
per_match AS (
    SELECT
        match_id,
        date_trunc('month', datetime) AS month,
        CASE WHEN winner = team1 THEN team1 ELSE team2 END AS w_team
    FROM matches
),
monthly AS (
    SELECT
        pm.month,
        COUNT(*) AS n_series,
        SUM(CASE WHEN COALESCE(tm.win_share, 0.0) < 0.5 THEN 1 ELSE 0 END) AS n_upsets
    FROM per_match pm
    LEFT JOIN team_month tm ON tm.team = pm.w_team AND tm.month = pm.month
    GROUP BY pm.month
),
bounds AS (
    SELECT
        date_trunc('month', MIN(datetime)) AS m0,
        date_trunc('month', MAX(datetime)) AS m1
    FROM matches
),
months AS (
    SELECT unnest(generate_series(m0, m1, INTERVAL 1 MONTH)) AS month FROM bounds
)
SELECT
    months.month,
    COALESCE(monthly.n_series, 0) AS n_series,
    COALESCE(monthly.n_upsets, 0) AS n_upsets,
    COALESCE(monthly.n_upsets::DOUBLE / NULLIF(monthly.n_series, 0), 0.0) AS upset_rate
FROM months
LEFT JOIN monthly ON monthly.month = months.month
ORDER BY months.month;

-- @query: q2
WITH matches_seq AS (
    SELECT
        match_id,
        datetime,
        team1 AS team,
        (winner = team1)::INT AS win
    FROM matches
    UNION ALL
    SELECT
        match_id,
        datetime,
        team2 AS team,
        (winner = team2)::INT AS win
    FROM matches
),
history AS (
    SELECT
        match_id,
        team,
        win,
        LAG(win, 1) OVER w AS w1,
        LAG(win, 2) OVER w AS w2,
        LAG(win, 3) OVER w AS w3
    FROM matches_seq
    WINDOW w AS (PARTITION BY team ORDER BY datetime, match_id)
)
SELECT
    match_id,
    team,
    CASE
        WHEN w1 IS NULL AND w2 IS NULL AND w3 IS NULL THEN 0.5
        ELSE (COALESCE(w1, 0) + COALESCE(w2, 0) + COALESCE(w3, 0))
            / (3.0 - (CASE WHEN w1 IS NULL THEN 1 ELSE 0 END
                        + CASE WHEN w2 IS NULL THEN 1 ELSE 0 END
                        + CASE WHEN w3 IS NULL THEN 1 ELSE 0 END))
    END AS form_win_share
FROM history
ORDER BY match_id, team;

-- @query: q3
-- gaps-and-islands: grp = rn_all - rn_win is constant exactly within a run of
-- consecutive wins (each win advances both counters by one); a loss advances
-- only rn_all, which breaks the run.
WITH matches_seq AS (
    SELECT
        team1 AS team, datetime, match_id, (winner = team1)::INT AS win
    FROM matches
    UNION ALL
    SELECT
        team2 AS team, datetime, match_id, (winner = team2)::INT AS win
    FROM matches
),
numbered AS (
    SELECT
        team,
        datetime,
        match_id,
        win,
        ROW_NUMBER() OVER (PARTITION BY team ORDER BY datetime, match_id) AS rn_all
    FROM matches_seq
),
wins AS (
    SELECT
        team,
        datetime,
        match_id,
        rn_all,
        ROW_NUMBER() OVER (PARTITION BY team ORDER BY datetime, match_id) AS rn_win
    FROM numbered
    WHERE win = 1
),
grouped AS (
    SELECT
        team,
        datetime,
        rn_all - rn_win AS grp,
        COUNT(*) OVER (PARTITION BY team, rn_all - rn_win) AS streak_length
    FROM wins
)
SELECT
    team,
    streak_length,
    MIN(datetime) AS streak_start,
    MAX(datetime) AS streak_end
FROM grouped
GROUP BY team, grp, streak_length
HAVING streak_length >= 5
ORDER BY streak_length DESC;

-- @query: q4
WITH preds AS (
    SELECT
        match_id,
        ABS(elo_t1_pre - elo_t2_pre) AS elo_gap,
        CASE
            WHEN elo_t1_pre >= elo_t2_pre THEN (winner = team1)::INT
            ELSE (winner = team2)::INT
        END AS favourite_won,
        CASE
            WHEN ABS(elo_t1_pre - elo_t2_pre) < 25 THEN 1
            WHEN ABS(elo_t1_pre - elo_t2_pre) < 50 THEN 2
            WHEN ABS(elo_t1_pre - elo_t2_pre) < 100 THEN 3
            ELSE 4
        END AS elo_bucket
    FROM m4_preds
)
SELECT
    elo_bucket,
    COUNT(*) AS n,
    1.0 - (SUM(favourite_won)::DOUBLE / COUNT(*)) AS upset_rate
FROM preds
GROUP BY elo_bucket
ORDER BY elo_bucket;

-- @query: q5
WITH pair_base AS (
    SELECT
        team1 AS a, team2 AS b, (winner = team1)::INT AS a_win FROM matches
    UNION ALL
    SELECT team2 AS a, team1 AS b, (winner = team2)::INT AS a_win FROM matches
)
SELECT
    a AS team_a,
    b AS team_b,
    SUM(a_win) AS a_wins,
    COUNT(*) - SUM(a_win) AS b_wins,
    COUNT(*) AS n
FROM pair_base
GROUP BY a, b
HAVING COUNT(*) >= 4
ORDER BY n DESC;