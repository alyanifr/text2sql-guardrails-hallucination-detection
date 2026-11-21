"""Plain-Python text-to-SQL pipeline: no agent framework.

question -> LLM writes SQL (JSON) -> validate -> EXPLAIN (catches hallucinated
tables/columns) -> run read-only with row cap -> LLM writes the analysis.
"""
import json
import os
import re
from typing import Callable

import sqlparse
from sqlparse.tokens import DDL, DML
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

MODEL = "huggingface/Qwen/Qwen3-32B"   # your pinned provider
MAX_ROWS = 50
MAX_ATTEMPTS = 3

# ----- Define LLM ----- #
LLM = Callable[[list[dict], bool], str]

DEFINITIONS = (
    "'popular', 'desired', 'in demand', 'most common' = number of postings (COUNT(*)) per group. "
    "'well paid', 'highest paying' = AVG(Salary_USD). "
    "'growing' = Job_Growth_Projection = 'Growth'. "
    "'remote' = Remote_Friendly = 'Yes'."
)

def hf_llm(messages: list[dict], thinking: bool) -> str:
    import litellm  # imported lazily so tests don't need it

    resp = litellm.completion(
        model=MODEL,
        messages=messages,
        api_key=os.getenv("HF_TOKEN"),
        temperature=0.6 if thinking else 0.0,
    )
    return resp.choices[0].message.content

def strip_think(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

# ----- Get database schema ----- #
def get_schema(engine: Engine) -> str:
    inspector = inspect(engine)
    return "\n".join(
        f"Table '{t}': " + ", ".join(f"{c['name']} ({c['type']})" for c in inspector.get_columns(t))
        for t in inspector.get_table_names()
    )

# ----- Guardrails ----- #
def validate_sql(sql_query: str) -> tuple[bool, str]:
    """Validates that the SQL query is a single, well-formed, read-only SELECT statement.
       This function acts as the first layer of guiderails to prevent destructive queries."""
    if not sql_query or not sql_query.strip():
        return False, "Query is empty."

    # Parse the SQL string into individual statements
    parsed_statements = sqlparse.parse(sql_query.strip())

    # Block multiple or stacked queries. Prevents SQL injection chaining.
    if len(parsed_statements) != 1:
        return False, "Only a single SQL statement is allowed per execution."

    statement = parsed_statements[0]

    # Verify top-vele statement type is SELECT
    if statement.get_type() != "SELECT":
        return False, f"operation '{statement.get_type()}' is forbidden. Only SELECT queries are permitted."

    # Catch possible subqueries hiding write operations
    for token in statement.flatten():
        if token.ttype in (DDL, DML) and token.value.upper() != "SELECT":
            return False, f"Forbidden SQL keyword detected: '{token.value.upper()}'"

    return True, "Valid"


def check_against_schema(engine: Engine, sql: str) -> tuple[bool, str]:
    """EXPLAIN makes SQLite resolve every table/column without running the query."""
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(f"EXPLAIN {sql}")
        return True, "OK"
    except Exception as e:
        return False, str(getattr(e, "orig", e))


def run_query(engine: Engine, sql: str) -> tuple[list[dict], bool]:
    """Fetch at most MAX_ROWS rows; the extra row tells us if the result was cut off."""
    with engine.connect() as conn:
        result = conn.exec_driver_sql(f"SELECT * FROM ({sql}) LIMIT {MAX_ROWS + 1}")
        cols = list(result.keys())
        rows = result.fetchall()
    truncated = len(rows) > MAX_ROWS
    return [dict(zip(cols, r)) for r in rows[:MAX_ROWS]], truncated


# ----- LLM steps ----- #
def parse_sql_json(raw: str) -> dict:
    """Pull {"sql": "..."} out of the model's reply, tolerating think blocks and fences."""
    cleaned = strip_think(raw)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in reply.")
    return json.loads(cleaned[start : end + 1])["sql"].strip().rstrip(";")


def generate_sql(llm: LLM, engine: Engine, question: str) -> dict:
    schema = get_schema(engine)
    messages = [
        {"role": "system", "content": (
            "You write read-only SQLite queries. Use ONLY the tables and columns below.\n"
            f"{schema}\n"
            f"Definitions of common words: {DEFINITIONS}\n"
            "If a vague term is not defined above and could map to more than one column, "
            "do NOT guess. Reply {\"clarify\": \"<one short question>\"}.\n"
            "Otherwise reply {\"sql\": \"...\", \"assumption\": \"<how you interpreted any "
            "vague terms, or 'none'>\"}"
            'Reply with JSON only: {"sql": "<one SELECT statement>"}')},
        {"role": "user", "content": f"{question} /no_think"},
    ]

    errors: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        raw = llm(messages, False)
        assumption = ""
        try:
            reply = parse_sql_json(raw)
            clarify = reply.get("clarify")
            if isinstance(clarify, str) and clarify.strip():      # ask, don't guess
                return {"sql": "", "assumption": "", "clarify": clarify.strip(), "errors": errors}
            
            sql = reply["sql"].strip().rstrip(";")
            assumption = str(reply.get("assumption") or "").strip()

            if assumption.lower() in ("none", "n/a"):
                assumption = ""
            ok, msg = validate_sql(sql)

            if ok:
                ok, msg = check_against_schema(engine, sql)

        except (ValueError, KeyError, TypeError, AttributeError) as e:   # JSONDecodeError is a ValueError
            sql, ok, msg = "", False, f"Bad reply format: {e!r}"

        if ok:
            return {"sql": sql, "assumption": assumption, "clarify": None, "errors": errors}
        errors.append(msg)
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"That failed: {msg}\nFix it and reply with JSON only. /no_think"},
        ]

    raise RuntimeError(f"Could not produce a valid query after {MAX_ATTEMPTS} attempts: {errors}")


def analyze(llm: LLM, question: str, sql: str, rows: list[dict], truncated: bool) -> str:
    note = (f"The result was cut off at {MAX_ROWS} rows, so it does NOT cover the whole table."
            if truncated else "The result is complete.")
    messages = [
        {"role": "system", "content": "You are a data analyst. Base your answer strictly on the data given."},
        {"role": "user", "content": (
            f"Question: {question}\nSQL used: {sql}\n{note}\n"
            f"Rows ({len(rows)}):\n{json.dumps(rows, default=str)}")},
    ]
    return strip_think(llm(messages, True))


# ----- Entry point ----- #
def answer(engine: Engine, question: str, llm: LLM = hf_llm) -> dict:
    gen = generate_sql(llm, engine, question)
    base = {"clarify": gen["clarify"], "assumption": gen["assumption"], "sql": gen["sql"],
            "retry_errors": gen["errors"], "rows_returned": 0, "truncated": False, "report": ""}
    if gen["clarify"]:                      # ambiguous: ask the user, run nothing
        return base

    rows, truncated = run_query(engine, gen["sql"])
    report = analyze(llm, question, gen["sql"], rows, truncated)
    if gen["assumption"]:                   # always show how a vague word was read
        report = f"*Interpreted as: {gen['assumption']}*\n\n{report}"
    return {**base, "rows_returned": len(rows), "truncated": truncated, "report": report}


def main():
    from setup_database import setup_database

    eng = setup_database()
    q = input("Ask a question: ").strip()
    if q:
        out = answer(eng, q)
        if out["clarify"]:
            print("\nI need a bit more detail:", out["clarify"])
        else:
            print("\nSQL:", out["sql"])
            print("\n" + out["report"])

if __name__ == "__main__":
    main()