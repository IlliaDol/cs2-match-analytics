"""M10 §3 — run the gold question set through the LLM analyst.

v2: multi-model benchmark. Each run writes outputs/llm_eval__<label>.csv plus a
token/cost summary row appended to outputs/llm_benchmark.csv, so several models
can be compared side by side.

Run from repo root:
    OPENAI_API_KEY=<key> OPENAI_BASE_URL=https://api.deepseek.com/v1 \
        EVAL_MODEL=deepseek-flash EVAL_LABEL=ds-flash \
        .venv/Scripts/python.exe scripts/eval_llm.py

Optional env:
    EVAL_QUESTIONS  path to question CSV (default db/gold_questions.csv)
    EVAL_MAX_TOKENS reasoning budget per call (default 3000, see qa.py)
    PRICE_IN / PRICE_OUT  $ per 1M tokens for the cost column (default 0)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cs2analytics.llm.qa import _run_sql, ask

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "db" / "gold_questions.csv"
DB = REPO / "outputs" / "cs2.duckdb"
BENCH = REPO / "outputs" / "llm_benchmark.csv"


def result_set(rows: list[tuple] | None) -> set | None:
    """Tuples rounded to 4dp — aggregation float jitter must not fail equality."""
    if rows is None:
        return None
    return {tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows}


def _norm_scalar(v):
    """One comparable scalar: floats rounded, strings trimmed/lowered."""
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def score_pair(model_rows, gold_rows) -> tuple[str, float]:
    """Grade a model result against gold: (verdict, partial_credit in [0,1]).

    `exact`   — identical row multiset (the old, brittle rule).
    `partial` — the gold's answer values are all present, but the shape differs
                (extra/renamed column, different row order). Semantically right,
                presentationally different.
    `wrong`   — the answer values never appear.

    Why partial exists: an LLM that returns (team, n_wins, n_losses) for "who won
    the most" *knows* the answer; exact-set equality scores it 0 and turns the
    benchmark into a column-naming lottery instead of a SQL-correctness measure.
    Credit is 0.5 so `exact` still dominates the headline number.
    """
    if model_rows is None or gold_rows is None:
        return "wrong", 0.0
    if result_set(model_rows) == result_set(gold_rows):
        return "exact", 1.0

    flat = {_norm_scalar(v) for row in model_rows for v in row}
    gold_flat = {_norm_scalar(v) for row in gold_rows for v in row}
    if not flat or not gold_flat:
        return "wrong", 0.0
    if gold_flat <= flat:  # every gold value reproduced somewhere in the output
        return "partial", 0.5
    return "wrong", 0.0


def main() -> None:
    gold_path = Path(os.environ.get("EVAL_QUESTIONS", GOLD))
    label = os.environ.get("EVAL_LABEL") or os.environ.get("EVAL_MODEL", "model")
    out_path = REPO / "outputs" / f"llm_eval__{label}.csv"
    price_in = float(os.environ.get("PRICE_IN", "0"))
    price_out = float(os.environ.get("PRICE_OUT", "0"))

    gold = pd.read_csv(gold_path)
    trap_ids = set(gold.loc[gold["gold_sql"].isna(), "question_id"])
    records: list[dict] = []
    ptoks = 0
    ctoks = 0

    for _, q in gold.iterrows():
        qid = q["question_id"]
        res = ask(q["question"], str(DB))
        ptoks += res.get("prompt_tokens") or 0
        ctoks += res.get("completion_tokens") or 0
        is_trap = qid in trap_ids

        if is_trap:
            # trap: correct = refused / errored / no SQL. A hallucinated table
            # that "runs" still counts as failure (the model guessed a table).
            refused_ok = (res["sql"] is None) or bool(res["error"])
            rec = {
                "question_id": qid,
                "executable": False,
                "result_match": False,
                "verdict": "refused" if refused_ok else "hallucinated",
                "credit": 1.0 if refused_ok else 0.0,
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
            # `model_rows` was computed here and never used after the WIP switched to
            # score_pair(), so it is dropped rather than left as dead code (ruff F841).
            verdict, credit = score_pair(res["rows"], gold_rows)
            result_match = verdict == "exact"
            rec = {
                "question_id": qid,
                "executable": executable,
                "result_match": result_match,
                "verdict": verdict,
                "credit": credit,
                "refused_ok": False,
                "error": (res["error"] or "")[:200],
            }
        rec["prompt_tokens"] = res.get("prompt_tokens")
        rec["completion_tokens"] = res.get("completion_tokens")
        rec["finish_reason"] = res.get("finish_reason")
        rec["repair_attempts"] = res.get("repair_attempts", 0)
        records.append(rec)
        print(
            f"{qid}: match={rec['result_match']} refused={rec['refused_ok']}"
            f" | {(rec['error'] or '')[:70]}"
        )

    eval_df = pd.DataFrame(records)
    eval_df.to_csv(out_path, index=False)
    n_answerable = len(gold) - len(trap_ids)
    answered = int(eval_df.loc[~eval_df["question_id"].isin(trap_ids), "result_match"].sum())
    traps = int(eval_df.loc[eval_df["question_id"].isin(trap_ids), "refused_ok"].sum())
    cost = ptoks / 1e6 * price_in + ctoks / 1e6 * price_out
    print(
        f"[{label}] {answered}/{n_answerable} correct, {traps}/{len(trap_ids)} traps,"
        f" tokens in={ptoks} out={ctoks}, est cost=${cost:.4f}"
    )

    summary = {
        "label": label,
        "model": os.environ.get("EVAL_MODEL", "deepseek-flash"),
        "questions": len(gold),
        "answerable_correct": answered,
        "answerable_total": n_answerable,
        "traps_refused": traps,
        "traps_total": len(trap_ids),
        "exec_rate": round(
            float(eval_df.loc[~eval_df["question_id"].isin(trap_ids), "executable"].mean()), 4
        ),
        "prompt_tokens": ptoks,
        "completion_tokens": ctoks,
        "price_in_per_M": price_in,
        "price_out_per_M": price_out,
        "est_cost_usd": round(cost, 4),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    bench = pd.DataFrame([summary])
    if BENCH.exists():
        prev = pd.read_csv(BENCH)
        prev = prev[prev["label"] != label]  # latest run per label wins
        bench = pd.concat([prev, bench], ignore_index=True)
    bench.to_csv(BENCH, index=False)


if __name__ == "__main__":
    main()
