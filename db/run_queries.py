"""Run the five marked queries from db/queries.sql against outputs/cs2.duckdb.

Writes outputs/q1.csv ... outputs/q5.csv. Run from repo root:
    .venv/Scripts/python.exe db/run_queries.py
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DB_PATH = REPO / "outputs" / "cs2.duckdb"
QUERIES = REPO / "db" / "queries.sql"
OUT = REPO / "outputs"


def parse_markers(sql_text: str) -> list[tuple[str, str]]:
    """Split on '-- @query: name' markers, return [(name, statement), ...]."""
    pattern = re.compile(r"--\s*@query:\s*(\w+)\s*\n(.*?)(?=\n--\s*@query:|\Z)", re.DOTALL)
    return [(m.group(1), m.group(2).strip()) for m in pattern.finditer(sql_text)]


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Q4 needs the M4 Elo predictions as a table
    preds_path = OUT / "m4_elo_predictions.csv"
    if not preds_path.exists():
        raise SystemExit(
            "outputs/m4_elo_predictions.csv missing — run the M4 report notebook first"
        )
    preds = pd.read_csv(preds_path)
    preds["datetime"] = pd.to_datetime(preds["datetime"])
    con.register("m4_preds", preds)

    sql_text = QUERIES.read_text(encoding="utf-8")
    for name, statement in parse_markers(sql_text):
        df = con.execute(statement).fetchdf()
        dest = OUT / f"{name}.csv"
        df.to_csv(dest, index=False)
        print(f"{name}: {len(df)} rows -> {dest.name}")
    con.close()


if __name__ == "__main__":
    main()
