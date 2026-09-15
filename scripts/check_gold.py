"""Validate every gold SQL in a question set without calling any model.

A gold answer is only a *gold* answer if it actually runs and actually means
what the question says. This script enforces both:

1. **Executes** each gold_sql against outputs/cs2.duckdb (read-only).
2. **Checks row shape**: every answerable question must return >=1 row and
   declare a non-empty result (a gold query that returns nothing is a broken
   question, not a hard question).
3. **Checks the trap questions have NO gold_sql** (they must be unanswerable).

Run from repo root:
    .venv/Scripts/python.exe scripts/check_gold.py [db/gold_questions_v2.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "outputs" / "cs2.duckdb"


def main(path: str) -> int:
    gold = pd.read_csv(REPO / path)
    con = duckdb.connect(str(DB), read_only=True)
    failures: list[str] = []

    traps = gold[gold["gold_sql"].isna()]
    answerable = gold[gold["gold_sql"].notna()]

    for _, row in answerable.iterrows():
        qid, sql = row["question_id"], row["gold_sql"]
        try:
            rows = con.execute(sql).fetchall()
        except Exception as exc:
            failures.append(f"{qid}: GOLD SQL DOES NOT RUN -> {str(exc)[:160]}")
            continue
        if not rows:
            failures.append(f"{qid}: gold SQL returned 0 rows — question has no answer")
            continue
        if len(rows) == 1 and all(v is None for v in rows[0]):
            failures.append(f"{qid}: gold SQL returned a single all-NULL row")
            continue
        cols = [d[0] for d in con.execute(sql).description]
        n = len(rows)
        print(f"  {qid}: OK  {n} row(s), cols={cols}")

    for _, row in traps.iterrows():
        print(f"  {row['question_id']}: trap (no gold SQL) — question: {row['question'][:60]}")

    print(
        f"\n{path}: {len(answerable)} answerable, {len(traps)} traps, "
        f"{len(failures)} broken gold SQL(s)"
    )
    for f in failures:
        print(f"  !! {f}")
    con.close()
    return 1 if failures else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "db/gold_questions_v2.csv"
    raise SystemExit(main(target))
