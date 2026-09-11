"""M10 §3 — run all 20 gold questions through the LLM analyst -> outputs/llm_eval.csv.

Run from repo root (DeepSeek-compatible endpoint):
    OPENAI_API_KEY=<key> .venv/Scripts/python.exe scripts/eval_llm.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cs2analytics.llm.qa import _run_sql, ask

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "db" / "gold_questions.csv"
DB = REPO / "outputs" / "cs2.duckdb"
OUT = REPO / "outputs" / "llm_eval.csv"


def result_set(rows: list[tuple] | None) -> set | None:
    """Tuples rounded to 4dp — aggregation float jitter must not fail equality."""
    if rows is None:
        return None
    return {tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows}


def main() -> None:
    gold = pd.read_csv(GOLD)
    trap_ids = set(gold.loc[gold["gold_sql"].isna(), "question_id"])
    records: list[dict] = []

    for _, q in gold.iterrows():
        qid = q["question_id"]
        res = ask(q["question"], str(DB))
        is_trap = qid in trap_ids

        if is_trap:
            # trap: correct = refused / errored / no SQL. A hallucinated table
            # that "runs" still counts as failure (the model guessed a table).
            refused_ok = (res["sql"] is None) or bool(res["error"])
            rec = {
                "question_id": qid,
                "executable": False,
                "result_match": False,
                "refused_ok": refused_ok,
                "error": (res["error"] or ("returned SQL: " + res["sql"] if res["sql"] else ""))[
                    :200
                ],
            }
        else:
            executable = res["rows"] is not None
            try:
                gold_rows = result_set(_run_sql(q["gold_sql"], DB))
            except Exception:
                gold_rows = None
            model_rows = result_set(res["rows"])
            result_match = bool(executable and gold_rows is not None and model_rows == gold_rows)
            rec = {
                "question_id": qid,
                "executable": executable,
                "result_match": result_match,
                "refused_ok": False,
                "error": (res["error"] or "")[:200],
            }
        records.append(rec)
        print(
            f"{qid}: match={rec['result_match']} refused={rec['refused_ok']} | {rec['error'][:70]}"
        )

    eval_df = pd.DataFrame(records)
    eval_df.to_csv(OUT, index=False)
    n_answerable = 20 - len(trap_ids)
    answered = int(eval_df.loc[~eval_df["question_id"].isin(trap_ids), "result_match"].sum())
    traps = int(eval_df.loc[eval_df["question_id"].isin(trap_ids), "refused_ok"].sum())
    n_total = len(eval_df)
    print(f"{answered}/{n_answerable} correct, {traps}/{len(trap_ids)} traps, {n_total}/20 total")


if __name__ == "__main__":
    main()
