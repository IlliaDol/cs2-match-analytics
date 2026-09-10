# M5 BUILD SPEC — Enterprise SQL (DuckDB)  · `DA DS MLE DE`

**Goal:** the roadmap's SQL checkpoint, verified mechanically: "a query combining
multiple joins, a window function, and a subquery — without looking up syntax."
You write `db/queries.sql` (one statement per named marker) plus a small Python runner
that executes each query against DuckDB and saves outputs. Tests pin the query RESULTS.

**Install once:** `pip install duckdb` (add to your venv, NOT to pyproject deps —
M5 is a standalone analysis layer; we promote it to a dependency in M11 if serving needs it).

---

## §1 Setup: `db/build_db.py` (yours)

1. Read `outputs/series_clean.csv` (the wrangled M3 table — 9,922 rows).
2. Also read the raw `data/raw/teams.csv` (19,846 rows, `team_id, team_name`) — this is
   your second table so JOINs are real, not decorative.
3. Create `outputs/cs2.duckdb` with:
   - `matches` table: all series columns (`CREATE TABLE matches AS SELECT ...`)
   - `teams` table as-is
   - a `team_stats` table: per-team n_series, n_wins, win_share (write with your own
     aggregation — this doubles as the join target for Q4)
4. Print table row counts as the last statement.

**Contract (`tests/test_sql_queries.py::test_db_exists`):** `outputs/cs2.duckdb` exists;
`matches` has exactly 9,922 rows; `team_stats` has ≥ 700 rows.

## §2 `db/queries.sql` — five queries, each after a marker comment

Markers are how the runner finds statements. Format:

```sql
-- @query: q1_monthly_upsets
<one SELECT statement>
```

**Q1 — monthly upset rate (join + window + subquery in ONE query — the checkpoint):**
For each month: total series, and the share where the winner's PRE-month win-rate was
below 0.5 ("upset"). Use a CTE that computes per-team monthly win share with a window
function, join it back to matches, wrap the share in a subquery or second CTE.
Contract: returns ≥ 40 rows (37 months), columns `[month, n_series, n_upsets, upset_rate]`.

**Q2 — form going into a match:** for every match, each team's last-3 series win share
*before* that match. `LAG`/window over a self-join or correlated subquery. Contract:
no row uses future data (the runner verifies with a trap row — see tests).

**Q3 — longest win streaks (gaps-and-islands):** `(date - ROW_NUMBER())` trick per team
on their chronologically sorted results; output `[team, streak_length, streak_start,
streak_end]`, keep streaks ≥ 5. Contract: ≥ 20 rows; every `streak_length` ≥ 5.

**Q4 — Elo-gap vs upset bucket:** join matches to your M4 backtest output
(`outputs/m4_elo_predictions.csv` — produced by the M4 report notebook; if it doesn't
exist yet, Q4 degrades to `team_stats`-based favorites). Bucket |ΔElo| into
[0-25, 25-50, 50-100, 100+]; upset rate per bucket. Contract: 4 buckets, upset_rate
decreasing with |ΔElo| (assert monotone non-increasing — the football-logic check).

**Q5 — head-to-head:** all-time series record per team pair with ≥ 4 meetings:
`[team_a, team_b, a_wins, b_wins, n]` ordered by n desc. Contract: ≥ 50 pairs.

## §3 `db/run_queries.py` (yours)

Opens `outputs/cs2.duckdb`, executes each marked query, writes each result to
`outputs/q<k>.csv`, prints row counts. One file, ≤ 80 lines, argparse optional.

## Tests (`tests/test_sql_queries.py`, data-gated — CI skips)
- db exists with right row counts (§1 contract)
- each `outputs/qN.csv` exists with the §2 column contracts
- Q3's streak sanity: `streak_length == (streak_end - streak_start).days + 1` ≥ 5
- Q4 monotonicity of upset_rate across buckets
- Q1: sum(n_series) == 9,922 (the window/CTE didn't fan out rows — the classic bug)

## Checkpoint (by hand, before running)
1. For the toy dataset: what row count does Q1 produce if two teams have identical
   names in different cases? (data-hygiene trap — think JOIN explosion)
2. Why does `LAG(win, 3)` NOT give "last 3 matches' wins" and what does?
3. Q3: why does subtracting ROW_NUMBER() group consecutive wins?

## Appendix — hints (after a real attempt)
- Q1 skeleton: `WITH monthly AS (SELECT team, date_trunc('month', datetime) m,
  AVG(win) OVER (PARTITION BY team, date_trunc('month', datetime)) ...` — then join.
- Q3: `ROW_NUMBER() OVER (PARTITION BY team ORDER BY datetime)` minus the date ordinal
  is constant within a consecutive-win run; GROUP BY that.
- Q4 without M4 output yet: substitute team_stats favorite flag (higher win share =
  favorite). Upgrade later.
