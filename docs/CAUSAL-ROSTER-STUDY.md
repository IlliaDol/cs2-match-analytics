# Causal Roster-Change Study

This study extends the predictive CS2 model with a separate causal-analysis question:

> What is the estimated change in a professional team’s next five series after a qualifying roster change, relative to a comparable team-period without a roster change?

This is not a betting model and it is not evidence that roster changes cause a particular outcome unless the design diagnostics support the interpretation.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_causal_roster_study.py
```

Optional window settings:

```powershell
.\.venv\Scripts\python.exe scripts\run_causal_roster_study.py --pre-window 5 --post-window 5
```

Inputs:

- `outputs/series_clean.csv`: one row per series with winner and team-oriented scores;
- `data/raw/cs2_all_tiers_games.csv`: public raw source used to obtain lineup ids.

The command does not download data. It writes derived artifacts only:

- `outputs/causal_treated_windows.csv`;
- `outputs/causal_event_pairs.csv`;
- `outputs/causal_event_study.csv`;
- `outputs/causal_balance.csv`;
- `outputs/causal_did.json`;
- `reports/roster-causal-study.md`.

If no complete matched panel exists, the command reports that honestly and does not invent an estimate.

## Treatment and safeguards

A treatment event requires a known current lineup, a known previous lineup, a changed player set, and at least three consecutive prior matches with the previous lineup. Unknown lineups break continuity; they are not silently classified as “unchanged”. By default, a second roster change inside the five-series post window excludes the event.

Controls are selected greedily and deterministically from time-near, tier-matched, non-changing lineup observations. This is a transparent baseline, not a guarantee against unobserved confounding.

## Required review before publishing a result

1. Inspect the event-study **lead** coefficients in `causal_event_study.csv`.
2. Inspect standardized mean differences in `causal_balance.csv`.
3. Confirm enough treated/control events and effective coverage.
4. Run placebo treatment dates and alternative windows.
5. Review the deterministic event-cluster bootstrap interval in `causal_did.json`; it is uncertainty under the matched-event sampling design, not a cure for confounding.
6. Keep failed specifications in the research log.

The two-by-two DiD in `causal_did.json` is a descriptive baseline:

```text
(treated_post - treated_pre) - (control_post - control_pre)
```

The same file contains a deterministic event-cluster bootstrap 95% interval. It quantifies sampling variability across matched events; it does not remove selection bias or repair failed parallel-trends diagnostics.
