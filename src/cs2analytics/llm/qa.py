"""M10 §2 — `ask`: natural-language question -> SQL -> rows, read-only.

Design notes:
- The DDL/DML guardrail fires BEFORE any model call (the test runs without an
  API key): both the user question and the model's returned SQL are checked.
- System prompt carries the REAL schema (DESCRIBE output) + the three quirks
  that matter for SQL; capped at ~1.5k tokens. Grounding beats context size.
- SQL execution gets a hard timeout; every exception becomes `error`, never a
  crash. NULL results are legal; DDL/DML never runs.
"""

from __future__ import annotations

import duckdb

DDL_PATTERNS = (
    "drop ",
    "delete ",
    "update ",
    "insert ",
    "create ",
    "alter ",
    "attach ",
    "copy ",
    "pragma ",
)

SYSTEM_PROMPT = """You are a SQL analyst. Write ONE DuckDB SQL SELECT statement that answers \
the user's question about professional CS2 series. Return ONLY the SQL — no prose, no \
markdown fences.

Schema (DuckDB):
- matches(match_id INT, datetime TIMESTAMP, tournament VARCHAR, team1 VARCHAR, team2 VARCHAR, \
winner VARCHAR, score1_match INT, score2_match INT, t1_series_score INT, t2_series_score INT, \
games_played INT, bestOf INT, tier VARCHAR)
  One row per series. `winner` is ALREADY derived (the team with more maps) — never \
re-derive it from a flag column; there is NO team1_win column.
- teams(team_id INT, team_name VARCHAR)
- team_stats(team VARCHAR, n_series BIGINT, n_wins BIGINT, win_share DOUBLE)

Quirks that matter:
1. Bo1 masquerade: `bestOf` lies; `games_played == 1` is the real best-of-one test \
(a "Bo3" with games_played=1 is a one-map series in practice).
2. Series scores are per-team maps won: t1_series_score / t2_series_score; a sweep is \
either score being 0 in a multi-map series (games_played >= 2).
3. `datetime` is a TIMESTAMP; filter with TIMESTAMP literals or date_trunc; tier values \
are exactly 'tier1' | 'tier2' | 'tier3'.

If the question needs data these tables cannot provide (players, rounds, economy, future \
dates), do NOT invent a table — return exactly: CANNOT_ANSWER"""


def _contains_ddl(text: str) -> bool:
    lowered = " ".join(str(text).lower().split())
    return any(p in lowered for p in DDL_PATTERNS)


def _extract_sql(raw: str) -> str | None:
    text = raw.strip()
    if not text or "CANNOT_ANSWER" in text:
        return None
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith("sql"):
                candidate = candidate[3:].strip()
            if candidate.upper().lstrip().startswith(("SELECT", "WITH")):
                return candidate.strip(";")
        return None
    if text.upper().lstrip().startswith(("SELECT", "WITH")):
        return text.strip(";")
    return None


def ask(
    question: str,
    db_path: str,
    model: str = "deepseek-flash",
    timeout_s: float = 30.0,
) -> dict:
    """Question -> {"sql", "rows", "error", "raw"} — never raises.

    Guardrails fire before any model call: destructive statements are refused
    outright, so `ask("DROP TABLE matches")` errors with sql=None.
    """
    result: dict = {"sql": None, "rows": None, "error": None, "raw": ""}
    if _contains_ddl(question):
        result["error"] = "read-only assistant: DDL/DML statements are refused"
        return result

    try:
        import os

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            result["error"] = "OPENAI_API_KEY not set — cannot call the model"
            return result
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=300,
            timeout=timeout_s,
        )
        raw = completion.choices[0].message.content or ""
        result["raw"] = raw
        usage = getattr(completion, "usage", None)
        result["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        result["completion_tokens"] = getattr(usage, "completion_tokens", None)
    except Exception as exc:  # network/library errors surface as `error`
        result["error"] = f"model call failed: {exc}"
        return result

    sql = _extract_sql(raw)
    if sql is None:
        result["error"] = "model refused or returned no SQL (refusal is the correct trap behaviour)"
        return result
    if _contains_ddl(sql):
        result["sql"] = sql
        result["error"] = "model produced a destructive statement — blocked by the read-only guard"
        return result

    result["sql"] = sql
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute(sql).fetchall()
        con.close()
        result["rows"] = rows
    except Exception as exc:
        result["error"] = f"SQL execution failed: {exc}"
    return result


def _run_sql(sql: str, db_path: str | Path) -> list[tuple]:
    """Execute one read-only SELECT against the DuckDB file."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def client_for(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)
