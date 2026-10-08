from __future__ import annotations

from .types import ExecutionResult, SqlTask

SYSTEM_PROMPT = """You are a Text-to-SQL policy operating in a read-only SQLite environment.
Your goal is to answer the user's question by producing exactly one safe SELECT/WITH query.
Use only tables and columns present in the supplied schema.
Never use INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, PRAGMA, ATTACH, or multiple statements.
Return only JSON in this form: {\"sql\": \"SELECT ...\"}.
Do not include hidden reasoning, Markdown, or prose outside the JSON object."""


def build_messages(
    task: SqlTask,
    schema: str,
    turn: int,
    previous_sql: str | None = None,
    previous_execution: ExecutionResult | None = None,
    feedback: str = "",
) -> list[dict[str, str]]:
    context = [f"Question:\n{task.question}", f"Database schema:\n{schema}"]
    if task.evidence:
        context.append(f"Additional evidence:\n{task.evidence}")
    if turn > 1:
        context.append(f"Previous SQL:\n{previous_sql or '[none]'}")
        if previous_execution is not None:
            if previous_execution.executed:
                context.append(
                    "Previous execution feedback:\n"
                    f"columns={previous_execution.columns}\n"
                    f"row_count={previous_execution.row_count}\n"
                    f"result_preview={previous_execution.rows[:10]}\n"
                    "The result did not match the expected answer."
                )
            else:
                context.append(
                    "Previous execution feedback:\n"
                    f"error_type={previous_execution.error_type}\n"
                    f"error={previous_execution.error_message}"
                )
        if feedback:
            context.append(f"Correction guidance:\n{feedback}")
        context.append("Rewrite the query to fix the failure. Return one JSON object only.")
    else:
        context.append("Generate the query. Return one JSON object only.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(context)}]


CHECK_SYSTEM_PROMPT = """You are a SQL verifier. Inspect the question, schema, SQL and execution result.
Identify the most likely semantic mistake without revealing or inventing the gold SQL.
Return concise JSON: {\"feedback\": \"...\"}. Do not return a corrected query."""


def build_check_messages(task: SqlTask, schema: str, sql: str, execution: ExecutionResult) -> list[dict[str, str]]:
    preview = execution.rows[:10] if execution.executed else []
    user = (
        f"Question:\n{task.question}\n\nSchema:\n{schema}\n\nCandidate SQL:\n{sql}\n\n"
        f"Execution status: executed={execution.executed}, error={execution.error_message}\n"
        f"Columns: {execution.columns}\nRows preview: {preview}\n"
        "The result is known to be incorrect. Diagnose the likely mistake."
    )
    return [{"role": "system", "content": CHECK_SYSTEM_PROMPT}, {"role": "user", "content": user}]
