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

from pathlib import Path

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

Schema (DuckDB). Column types are real — respect them:
- matches(match_id BIGINT, datetime TIMESTAMP WITH TIME ZONE, tournament VARCHAR, \
team1 VARCHAR, team2 VARCHAR, winner VARCHAR, score1_match BIGINT, score2_match BIGINT, \
t1_series_score BIGINT, t2_series_score BIGINT, games_played BIGINT, bestOf DOUBLE, \
tier VARCHAR, total_maps BIGINT, margin BIGINT, is_bo1 BOOLEAN, swept BOOLEAN, \
bo5_sweep BOOLEAN)
  One row per series (9,920 series). `winner` is ALREADY derived (the team that won
  more maps) — there is NO team1_win column, so never re-derive it from a flag.
- teams(team_id BIGINT, team_name VARCHAR)  — 19,846 roster rows; join on team_name.
- team_stats(team VARCHAR, n_series BIGINT, n_wins HUGEINT, win_share DOUBLE)

Columns that are ALREADY computed for you (prefer them; they cost nothing to use):
- is_bo1      : TRUE when games_played = 1 (exactly equivalent — verified on all rows).
- total_maps  : identical to games_played.
- margin      : map margin of victory, 1..3.
- swept       : the loser took zero maps (2-0 or 3-0).
- bo5_sweep   : a best-of-five that ended 3-0.

Quirks that matter:
1. Bo1 masquerade: `bestOf` LIES and is a DOUBLE (may print as 1.0). `is_bo1`
   (equivalently games_played = 1) is the real best-of-one test — a row claiming
   bestOf = 3 with is_bo1 TRUE was a one-map series in practice.
2. Series scores are per-team maps won: t1_series_score / t2_series_score; a sweep is
   `swept` (or either score being 0 in a multi-map series).
3. `datetime` is TIMESTAMP WITH TIME ZONE in Europe/Berlin; compare with TIMESTAMP
   literals (e.g. TIMESTAMP '2025-01-01') or date_trunc. Data spans 2023-01-10 ..
   2026-06-28. Use strftime(datetime, '%Y') or EXTRACT(YEAR FROM datetime) for years.
4. tier values are exactly 'tier1' | 'tier2' | 'tier3'.
5. A team can appear in team1 or team2 — to count ALL of a team's series, unpivot
   with UNION ALL (or GREATEST/LEAST for unordered pairs), never just one side.

If the question needs data these tables cannot provide (players, kills, rounds, \
economy, headshots, maps/vetoes, prize pools, rosters, coaches, rankings, viewers, \
or any FUTURE/predictive question), do NOT invent a table or a column — return exactly: \
CANNOT_ANSWER"""


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


def _model() -> str:
    import os

    return os.environ.get("EVAL_MODEL", "deepseek-flash")


def _max_tokens() -> int:
    """Reasoning models burn thinking tokens before any visible SQL; too-small
    budgets return empty content with finish_reason=length (measured: a trivial
    prompt needed 498 hidden reasoning tokens on a Hy4-class relay). 3000 keeps
    every model in the benchmark inside its visible-answer headroom."""
    import os

    return int(os.environ.get("EVAL_MAX_TOKENS", "3000"))


def ask(
    question: str,
    db_path: str,
    model: str | None = None,
    timeout_s: float = 60.0,
) -> dict:
    """Question -> {"sql", "rows", "error", "raw"} — never raises.

    Guardrails fire before any model call: destructive statements are refused
    outright, so `ask("DROP TABLE matches")` errors with sql=None.
    Model defaults to $EVAL_MODEL (was hardcoded); reasoning models need a
    large token budget ($EVAL_MAX_TOKENS, default 3000) or they return empty
    content with finish_reason=length.
    One self-repair retry: if the first SQL fails to execute, the error text is
    fed back so the model can fix its statement (off for trap questions — a
    refusal must stay a refusal).
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
        model = model or _model()
        max_tokens = _max_tokens()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout_s,
        )
        raw = completion.choices[0].message.content or ""
        finish = getattr(completion.choices[0], "finish_reason", None)
        usage = getattr(completion, "usage", None)
        result["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        result["completion_tokens"] = getattr(usage, "completion_tokens", None)
        result["finish_reason"] = finish
    except Exception as exc:  # network/library errors surface as `error`
        result["error"] = f"model call failed: {exc}"
        return result

    sql = _extract_sql(raw)
    if sql is None:
        if finish == "length" and not result["raw"]:
            result["error"] = (
                "empty content with finish_reason=length: reasoning consumed the whole "
                "token budget — raise EVAL_MAX_TOKENS"
            )
        else:
            result["error"] = (
                "model refused or returned no SQL (refusal is the correct trap behaviour)"
            )
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
        # Self-repair: feed the SQL error back once. Trap questions never get
        # here (their model output was None or DDL) so refusals stay refusals.
        fix = _ask_repair(client, model, max_tokens, question, sql, str(exc), timeout_s)
        if fix is not None:
            fixed_sql, fixed_raw, usage2 = fix
            result["completion_tokens"] = (result["completion_tokens"] or 0) + (
                usage2 or 0
            )
            result["repair_attempts"] = 1
            if fixed_sql is None:
                result["error"] = "self-repair returned no SQL"
                return result
            if _contains_ddl(fixed_sql):
                result["sql"] = fixed_sql
                result["error"] = "self-repair produced a destructive statement — blocked"
                return result
            result["sql"] = fixed_sql
            result["raw"] = (result["raw"] or "") + "\n--REPAIR--\n" + (fixed_raw or "")
            try:
                con = duckdb.connect(str(db_path), read_only=True)
                rows = con.execute(fixed_sql).fetchall()
                con.close()
                result["rows"] = rows
                result["error"] = None
                return result
            except Exception as exc2:
                result["error"] = f"SQL execution failed after repair: {exc2}"
                return result
        result["error"] = f"SQL execution failed: {exc}"
    return result


def _ask_repair(client, model, max_tokens, question, bad_sql, err, timeout_s):
    """One retry with the execution error fed back. Returns (sql|None, raw, ctokens)."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nYour SQL:\n{bad_sql}\n\n"
                        f"DuckDB error:\n{err}\n\nReturn ONE corrected DuckDB SELECT. "
                        "SQL only."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout_s,
        )
        raw = completion.choices[0].message.content or ""
        ctokens = getattr(getattr(completion, "usage", None), "completion_tokens", 0)
        return _extract_sql(raw), raw, ctokens
    except Exception:
        return None


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
