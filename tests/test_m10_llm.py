"""M10 tests — gold eval set + eval artifacts + the read-only guardrail.

Skips automatically until the human builds db/gold_questions.csv and llm/qa.py.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
GOLD = REPO / "db" / "gold_questions.csv"
EVAL = REPO / "outputs" / "llm_eval.csv"

try:
    from cs2analytics.llm import qa  # noqa: F401

    _M10_READY = True
    _M10_REASON = ""
except ModuleNotFoundError as _e:
    _M10_READY = False
    _M10_REASON = f"M10 qa module not implemented yet ({_e.name}) — the human's build"

pytestmark = [pytest.mark.skipif(not _M10_READY, reason=_M10_REASON)]


def test_gold_set_shape():
    if not GOLD.exists():
        pytest.skip("db/gold_questions.csv not authored yet (M10 §1)")
    import pandas as pd

    df = pd.read_csv(GOLD)
    assert {"question_id", "question", "gold_sql"} <= set(df.columns)
    assert len(df) == 20
    assert df["question_id"].is_unique
    assert (df["question"].str.len() > 10).all()
    answerable = df["gold_sql"].notna().sum()
    traps = df["gold_sql"].isna().sum()
    assert answerable >= 16, f"need >=16 answerable questions, got {answerable}"
    assert traps >= 4, f"need >=4 trap questions (gold_sql NULL), got {traps}"


def test_eval_artifact():
    if not EVAL.exists():
        pytest.skip("outputs/llm_eval.csv not produced yet (M10 §3)")
    import pandas as pd

    df = pd.read_csv(EVAL)
    assert {"question_id", "executable", "result_match", "refused_ok", "error"} <= set(df.columns)
    assert len(df) == 20


@pytest.mark.skipif(
    not (REPO / "outputs" / "cs2.duckdb").exists(),
    reason="DuckDB not built yet (M5)",
)
def test_sql_guardrail_blocks_ddl():
    """Read-only rule: destructive statements must be refused before any model call."""
    from cs2analytics.llm.qa import ask

    res = ask("DROP TABLE matches", REPO / "outputs" / "cs2.duckdb")
    assert res["sql"] is None
    assert res["error"], "DDL must produce an explicit error, not a query"
