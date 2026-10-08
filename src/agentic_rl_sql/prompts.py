from __future__ import annotations

from .types import ExecutionResult, PolicyTask

SYSTEM_PROMPT = """You are a Text-to-SQL agent in a read-only SQLite environment.
Generate exactly one safe SELECT/WITH query using the provided schema.
You do NOT know the correct answer. You only see actual execution results
and database errors; execution success does not prove answer correctness.
Return JSON: {"sql":"SELECT ...", "decision":"final"}.
If you want to inspect the real result and revise, choose "decision":"inspect".
Inspection costs a further call and is limited by the turn budget.
The environment automatically retries invalid or failed SQL when budget remains.
Never use INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, PRAGMA, or ATTACH.
Do not output reasoning or prose outside the JSON object."""


def build_messages(
    task: PolicyTask, schema: str, turn: int,
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
                    "Previous execution result (correctness UNKNOWN):\n"
                    f"columns={previous_execution.columns}\n"
                    f"row_count={previous_execution.row_count}\n"
                    f"result_preview={previous_execution.rows[:10]}\n"
                    f"truncated={previous_execution.truncated}"
                )
            else:
                context.append(
                    "Previous execution feedback:\n"
                    f"error_type={previous_execution.error_type}\n"
                    f"error={previous_execution.error_message}"
                )
        if feedback:
            context.append(f"Available feedback:\n{feedback}")
        context.append("Independently decide whether to revise. Return JSON only.")
    else:
        context.append("Generate SQL and choose final or inspect. Return JSON only.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(context)}]


CHECK_SYSTEM_PROMPT = """You are a SQL self-checker. Inspect ONLY the question,
schema, SQL and actual execution result. Correctness is UNKNOWN.
Never claim access to any expected/gold answer. Return concise JSON:
{"feedback":"..."}. Describe potential errors, not corrected SQL."""


def build_check_messages(
    task: PolicyTask, schema: str, sql: str, execution: ExecutionResult,
) -> list[dict[str, str]]:
    preview = execution.rows[:10] if execution.executed else []
    user = (
        f"Question:\n{task.question}\n\nSchema:\n{schema}\n\nCandidate SQL:\n{sql}\n\n"
        f"Execution status: executed={execution.executed}, error={execution.error_message}\n"
        f"Columns: {execution.columns}\nRows preview: {preview}\n"
        "Correctness is unknown. Identify possible semantic issues, if any."
    )
    return [{"role": "system", "content": CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": user}]
