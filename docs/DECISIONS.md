# Decision Log

One entry per design decision that a reviewer would question. Format: date, decision, why.

| Date | Decision | Why |
|---|---|---|
| 2026-09-10 | src/ layout + hatchling | import-safe packaging, modern standard |
| 2026-09-10 | ruff for lint+format | one tool, zero config fights |
| 2026-09-10 | toy CSV tracked, real data git-ignored | CI must run without local data |
| 2026-09-10 | time-based splits (fixed cutoff) | teams/rosters/meta drift; random splits leak |
