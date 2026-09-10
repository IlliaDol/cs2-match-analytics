# M10 BUILD SPEC — GenAI: the analyst chat over your own data  · `DS MLE DA`

**Install:** `pip install openai` (works with OpenRouter/Cerebras/Groq base URLs).
Key from env: `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for non-OpenAI providers). Never commit keys.

---

## §1 `db/gold_questions.csv` — the eval set (yours, authored by hand)

20 rows, columns `[question_id, question, gold_sql]`. Rules:
- Every `gold_sql` must run against `outputs/cs2.duckdb` (M5) and return ≥ 1 row.
- Cover the schema WITH its traps, minimum spread:
  - 4 straightforward aggregations (win counts, series counts by tier)
  - 4 with date filtering (2025 only, last 6 months, before a cutoff)
  - 4 join/window (longest streak, head-to-head, form)
  - 4 that require knowing the quirks (Bo1 vs Bo3 filtering; `games_played` NOT `bestOf`;
    the `winner` column being pre-derived; tier-tagged tables)
  - 4 that SHOULD FAIL gracefully (ask about a column that doesn't exist, a future year,
    a player-level question the series schema can't answer) — gold_sql = `NULL` and the
    expected behaviour is "refuse or ask for clarification", not a hallucinated query.

Write them as a real analyst would ask a colleague — no SQL words in the question text.
Contract (tested): 20 rows, unique ids, non-empty questions, `gold_sql` non-empty for the
16 answerable ones.

## §2 `src/cs2analytics/llm/qa.py` (yours)

```python
def ask(question: str, db_path, model: str = "gpt-4o-mini") -> dict
    # returns {"sql": str|None, "rows": list|None, "error": str|None, "raw": str}
```
- System prompt must include the REAL schema (paste the `DESCRIBE` output of each table)
  plus the three quirks from `DATA.md` that matter for SQL (Bo1 masquerade, winner column
  semantics, tier tags). Cap the prompt at ~1.5k tokens — this is a prompting exercise, not
  a context-stuffing exercise.
- Execute the returned SQL against DuckDB with a timeout; catch exceptions and return them
  in `error` (no crashing).
- Forbid DDL/DML (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `CREATE`) — reject with an error.
  (Read-only rule; tested.)

## §3 `scripts/eval_llm.py` (yours) → `outputs/llm_eval.csv`

Runs all 20 questions: columns `[question_id, executable, result_match, refused_ok, error]`.
- `executable`: SQL ran without error.
- `result_match`: model's result equals the gold result (compare as sets of tuples;
  for aggregations round floats to 4dp).
- `refused_ok`: for the 4 trap questions, TRUE if the model refused/clarified (no SQL or
  an explicit error) — a hallucinated table name counts as failure even if it "errored".
- Print the summary line: `X/16 answered correctly, 4/4 traps handled, Y/20 total`.

## §4 `notebooks/06_llm_analyst.ipynb` — the writeup
1. Eval table + the summary line.
2. **5 failure cases verbatim** with the model's SQL and what went wrong (schema blindness,
   quirk ignorance, wrong aggregation grain — name the failure mode for each).
3. Cost/latency: tokens per question, $ per 1000 questions at your provider's rate.
4. One paragraph: why retrieval/context engineering beats "just use a bigger context window"
   for this use case (data updates weekly; schema is small but quirks are non-obvious;
   grounding beats guessing).

## Tests (`tests/test_m10_llm.py`, skipif artifacts/key missing)
- gold CSV: 20 rows, unique ids, ≥ 16 with SQL, ≥ 4 with NULL gold (the traps)
- eval CSV exists with the 5 columns and 20 rows
- `qa.ask` refuses DDL: `ask("DROP TABLE matches")` → `error` non-None, `sql` None
  (this one runs WITHOUT an API key by asserting the guardrail fires before the model call —
  design `ask()` so the DDL check happens on the SYSTEM PROMPT-independent path if possible;
  if not, mark the test skipif no key)

## Checkpoint
1. Why is "execution accuracy" the right metric here instead of string-matching the SQL?
2. Your model gets 14/16 with the schema in the prompt. Name two things to try before
   changing the model.
3. Which of your 20 questions would a real analyst consider unfair, and why keep it?

## Appendix — hints
- Result comparison: `set(map(tuple, df.itertuples(index=False, name=None)))`; ROUND before
  comparing floats, or 2 `AVG`s will differ in the 12th decimal.
- Prompt skeleton: schema block → 3 few-shot examples (one per quirk) → "Return ONLY SQL."
- If OpenRouter: base_url `https://openrouter.ai/api/v1`, cheap model e.g. `openai/gpt-4o-mini`.
