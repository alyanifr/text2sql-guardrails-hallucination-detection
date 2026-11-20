import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any
from textwrap import dedent

import pandas as pd
import sqlparse 
from sqlparse.tokens import Keyword
from sqlalchemy import create_engine, text, inspect
from crewai import Agent, LLM, Task, Crew, Process
from crewai.tools import tool
from setup_database import setup_database

# ----- Configure logger ----- #
AUDIT_LOG_FILE = "audit.log"

audit_logger = logging.getLogger("audit_trail")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False

if not audit_logger.handlers:           # Logs every attempts
    file_handler = logging.FileHandler(AUDIT_LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    audit_logger.addHandler(file_handler)

# JSON audit entry
def log_audit_event(event_type: str, payload: dict[str, Any]) -> None:
    """Writes a structured JSON event (SQL execution or Crew step/task) entry to the audit log."""
    entry = {
        "event_type": event_type,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload
    }
    audit_logger.info(json.dumps(entry, default=str))

# CrewAI callbacks for unified audit logging
def crew_step_callback(step_output: Any) -> None:
    """Logs every intermediate agent thought/tool call into audit.log"""
    log_audit_event("CREW_AGENT_STEP", {"step_detail": str(step_output)})

def crew_task_callback(task_output: Any) -> None:
    """Logs completed task outputs into audit.log"""
    log_audit_event("CREW_TASK_COMPLETED", {
        "task_description": getattr(task_output, "description", "N/A"),
        "agent": getattr(task_output, "agent", "N/A"),
        "raw_output": getattr(task_output, "raw", str(task_output)),
        },
    )

# ----- Define LLM ----- #
hf_llm = LLM(
    model="huggingface/Qwen/Qwen3-32B",
    temperature=0.0,     # deterministic SQL generation
    hf_token=os.getenv('HF_TOKEN')
)

# ----- Initialized guardrails keywords ----- #
MAX_ROW_LIMITS = 50     # cut tokens usage
FORBIDDEN_KEYWORDS = {
    "CREATE",
    "ALTER",
    "DROP",
    "INSERT",
    "UPDATE",
    "DELETE",
    "REPLACE",
    "TRUNCATE",
    "EXECUTE",
    "EXEC",
    "GRANT",
    "REVOKE"
}

# ----- SQL syntax validation with sqlparse ----- #
# Validation function
def validate_sql_syntax(sql_query: str) -> tuple[bool, str]:
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
        if token.ttype in Keyword or token.ttype in Keyword.DML:
            if token.value.upper() in FORBIDDEN_KEYWORDS:
                return False, f"Forbidden SQL keyword detected: '{token.value.upper()}'"

    return True, "Valid"

# ----- Schema inspection ----- #
@tool("get database schema")
def get_database_schema() -> str:
    """Returns a comma-separated list of all table names and columns in the database."""
    inspector = inspect(engine)
    schema_info = []
    for table_name in inspector.get_table_names():
        columns = [
            f"{col['name']} ({col['type']})"
            for col in inspector.get_columns(table_name)
        ]
        schema_info.append(f"Table '{table_name}': {', '.join(columns)}")
    
    schema_str = "\n".join(schema_info) if schema_info else "No tables found."
    log_audit_event("SCHEMA_INSPECTION", {"tables": inspector.get_table_names()})
    return schema_str

# ----- Execute Query ----- #
# Execute SQL query
@tool("execute_query")
def execute_query(sql_query: str) -> str:
    """Validates SQL syntax and executes read-only SELECT query against the database."""
    #timestamp = datetime.now(timezone.utc).isoformat()
    cleaned_query = sqlparse.format(sql_query, strip_comments=True, keyword_case="upper").strip()

    audit_record: dict[str, Any] = {
        "raw_query": sql_query,
        "cleaned_query": cleaned_query,
        "status": "PENDING",
        "execution_time_ms": 0.0,
        "total_rows_returned": 0,
        "rows_sent_to_agent": 0,
        "error": None,
    }

    # Run syntax validation before executing against the database
    is_valid, message = validate_sql_syntax(cleaned_query)
    if not is_valid:
        audit_record["status"] = "VALIDATION_FAILED"
        audit_record["error"] = message
        log_audit_event("SQL_EXECUTION", audit_record)
        return json.dumps({
            "status": "error",
            "error_type": "validation_error",
            "message": message,
        })

    try:
        with engine.connect() as conn:

            # Measure query execution & dataframe load time
            start_time = time.perf_counter()
            df = pd.read_sql_query(text(f"SELECT * FROM ({cleaned_query.rstrip(';')}) LIMIT {MAX_ROW_LIMITS + 1}"), conn)       # Max row limits + truncation flag
            #df = pd.read_sql_query(text(cleaned_query), conn)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        total_rows = len(df)
        capped_df = df.head(MAX_ROW_LIMITS)

        # Convert dataframe records to json
        raw_results = json.loads(
            capped_df.to_json(orient="records", date_format="iso")
        )

        # Update audit log
        audit_record.update({
            "status": "SUCCESS",
            "execution_time_ms": elapsed_ms,
            "total_rows_returned": total_rows,
            "rows_sent_to_agent": len(raw_results),
            "truncated": total_rows > MAX_ROW_LIMITS,
        })

        log_audit_event("SQL_EXECUTION", audit_record)

        # Package response for agent
        response_payload = {
            "status": "success",
            "metadata": {
                "execution_time_ms": elapsed_ms,
                "rows_returned": total_rows,
                "rows_displayed": len(raw_results),
                "truncated": total_rows > MAX_ROW_LIMITS,
            },
            "results": raw_results
        }

        return json.dumps(response_payload, indent=2)

    except Exception as e:
        audit_record["status"] = "EXECUTION_ERROR"
        audit_record["error"] = str(e)
        log_audit_event("SQL_EXECUTION", audit_record)

        return json.dumps({
            "status": "error",
            "error_type": "database_execution_error",
            "message": str(e),
        })

# ----- Defining agents ----- #
# Agent 1: SQL expert 
sql_dev = Agent(
    role="Senior SQL Database Developer",
    goal=dedent(
        """
        Inspect the database schema, write optimized read-only SQL queries,
        and extract accurate datasets to answer: {user_question}
        """
    ),
    backstory=dedent(
        """
        You are a database engineer with more than 10 years of professional experience,
        a master at creating efficient SQL queries for complex request.
        You have a deep understanding of how different databases work and how to optimize quaries.
        You have a habit of always inspecting the database schema by using 'get_database_schema',
        to be sure the tables and columns actually exist. You would also validate the SQL queries,
        before executing them against the database using 'execute_query'.
        """
    ),
    llm=hf_llm,
    tools=[get_database_schema, execute_query],
    verbose=True,
    allow_delegation=False,
)

# Agent 2: Data analyst
data_analyst = Agent(
    role="Senior Data Analyst",
    goal=dedent(
        """
        Analyze the extracted database results and query metrics to deliver
        clear, actionable insights for: {user_question}.
        """
    ),
    backstory=dedent(
        """
        You are a data analyst with more than 10 years of professional experience.
        You are an expert at interpreting structured JSON datasets, spotting trends, summarizing key metrics, 
        and noting any data truncation or query performance observations.
        You are working closely with the Senior SQL Database Developer to obtain the required data.
        Your analysis is strictly based on the provided data.
        """
    ),
    llm=hf_llm,
    verbose=True,
    allow_delegation=False,
)

# ----- Defining tasks and crew ----- #
# Tasks
extract_data = Task(
    description=dedent(
        """
        1. Use 'get_database_schema' to examine the available tables and columns.
        2. Write clean, read-only SELECT query to answer: '{user_question}'.
        3. Execute the query using 'execute_query' and return the complete JSON response
        containing the results, row counts, and execution time.
        """
    ),
    expected_output=dedent(
        """
        The full structured JSON payload returned by 'execute_query',
        preserving all metadata and result rows.
        """
    ),
    agent=sql_dev,
)

analyze_data = Task(
    description=dedent(
        """
        1. Review the extracted dataset and metadata provided by the Senior SQL Database Developer
        for the question: '{user_question}'.
        2. provide a clear summary of the findings, highlight key trends or anomalies,
        and include a brief technical note on query execution time,
        whether rows were truncated.
        """
    ),
    expected_output=dedent(
        """
        A well-structured analytical report answering '{user_question}',
        complete with key data takeaways and a brief query performance summary.
        """
    ),
    agent=data_analyst,
    context=[extract_data],
)

# Crew
crew = Crew(
    agents=[sql_dev, data_analyst],
    tasks=[extract_data, analyze_data],
    process=Process.sequential,
    verbose=True,
    memory=False,
    step_callback=crew_step_callback,
    task_callback=crew_task_callback,
)

# ----- Execution entry point ----- #
if __name__ == "__main__":
    engine = setup_database()
    user_query = input()
    log_audit_event("CREW_RUN_STARTED", {
        "user_question": user_query,
        "agents": [a.role for a in crew.agents],
        },
    )   

    try:
        final_report = crew.kickoff(inputs={"user_question": user_query})

        log_audit_event("CREW_RUN_COMPLETED", {
            "status": "SUCCESS",
            "token_usage": getattr(final_report, "token_usage", "N/A"),
            },
        )

        print("\n==================== FINAL REPORT ====================\n")
        print(final_report)

    except Exception as e:
        log_audit_event("CREW_RUN_FAILED", {"status": "ERROR", "error": str(e)})
        raise