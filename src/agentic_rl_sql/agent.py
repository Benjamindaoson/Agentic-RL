from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .context import MessageCounter, fit_schema_to_budget
from .db import render_schema
from .execution import execute_sql
from .prompts import build_check_messages, build_messages
from .types import ExecutionResult, PolicyTask, Trajectory, TrajectoryStep

SQL_FENCE_RE = re.compile(r"\x60\x60\x60(?:sql)?\s*(.*?)\s*\x60\x60\x60", re.I | re.S)
JSON_RE = re.compile(r"\{.*\}", re.S)


class AsyncChatClient(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(slots=True)
class OpenAICompatibleClient:
    base_url: str
    api_key: str
    model: str = "auto"
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = None
    _client: Any = field(init=False, default=None, repr=False)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        from openai import AsyncOpenAI

        if self._client is None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=3)
        kwargs: dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def _json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except (ValueError, TypeError):
        pass
    matched = JSON_RE.search(raw)
    if matched:
        try:
            value = json.loads(matched.group(0))
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            pass
    return None


def parse_sql_output(text: str) -> str:
    raw = (text or "").strip()
    obj = _json_object(raw)
    if obj is not None:
        sql = obj.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql.strip()
    fence = SQL_FENCE_RE.search(raw)
    if fence:
        return fence.group(1).strip()
    candidate = raw.strip(chr(96)).strip()
    if candidate.upper().startswith(("SELECT", "WITH")):
        return candidate
    raise ValueError("model output did not contain a SQL query")


def parse_decision(text: str) -> str:
    obj = _json_object(text)
    value = obj.get("decision", "final") if obj else "final"
    if value not in ("final", "inspect"):
        raise ValueError("decision must be final or inspect")
    return value


def parse_checker_feedback(text: str) -> str:
    obj = _json_object(text)
    value = obj.get("feedback") if obj else None
    if isinstance(value, str) and value.strip():
        return value[:1000]
    return (text or "")[:1000] or "Independently re-check the SQL semantics."


def correction_feedback(execution: ExecutionResult) -> str:
    """Environment-only feedback: NEVER based on gold SQL."""
    if not execution.valid:
        return "SQL was not parseable. Return one valid SELECT/WITH statement."
    if not execution.safe:
        return "SQL was blocked by the read-only environment."
    if not execution.executed:
        return f"Query failed: {execution.error_type}: {execution.error_message}."
    if execution.truncated:
        return "Result truncated; use filtering or aggregation to narrow the query."
    return "SQL executed. Correctness is unknown; independently verify the question."


class SqlAgentRunner:
    """Blind execution: no gold SQL, reward or expected-answer feedback."""

    def __init__(
        self, client: AsyncChatClient, *, max_schema_chars: int = 12000,
        timeout_seconds: float = 8.0, max_rows: int = 100000,
        token_counter: MessageCounter | None = None,
    ) -> None:
        self.client = client
        self.max_schema_chars = max_schema_chars
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows
        self.token_counter = token_counter

    async def run(self, task: PolicyTask) -> Trajectory:
        if not isinstance(task, PolicyTask):
            raise TypeError("SqlAgentRunner accepts PolicyTask only; call SqlTask.policy_view()")
        started = time.perf_counter()
        schema = render_schema(task.db_path, max_chars=self.max_schema_chars)
        steps: list[TrajectoryStep] = []
        previous_sql: str | None = None
        previous_execution: ExecutionResult | None = None
        feedback = ""
        stop_reason = "max_turns"
        final_sql: str | None = None
        if task.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        for turn in range(1, task.max_turns + 1):
            make_prompt = lambda s: build_messages(task, s, turn, previous_sql, previous_execution, feedback)
            messages, prompt_tokens, truncated = fit_schema_to_budget(
                make_prompt, schema, task.context_limit, self.token_counter,
            )
            raw = await self.client.complete(messages)
            decision = "final"
            try:
                sql = parse_sql_output(raw)
                decision = parse_decision(raw)
                execution = execute_sql(task.db_path, sql, self.timeout_seconds, self.max_rows)
            except ValueError as exc:
                sql = None
                execution = ExecutionResult(
                    sql="", valid=False, safe=False, executed=False,
                    error_type="output_parse_error", error_message=str(exc),
                )
            feedback = correction_feedback(execution)
            checker_output = None
            should_continue = (
                (not execution.executed or execution.truncated or decision == "inspect")
                and turn < max(task.max_turns, 1)
            )
            if should_continue and sql and task.metadata.get("explicit_check", False) and execution.safe:
                check_messages, _, _ = fit_schema_to_budget(
                    lambda s: build_check_messages(task, s, sql, execution),
                    schema, task.context_limit, self.token_counter,
                )
                checker_output = await self.client.complete(check_messages)
                feedback = parse_checker_feedback(checker_output)
            steps.append(TrajectoryStep(
                turn=turn, prompt_kind="initial" if turn == 1 else "rewrite",
                raw_model_output=raw, sql=sql, execution=execution,
                feedback=feedback, decision=decision, checker_output=checker_output,
                prompt_tokens=prompt_tokens, schema_truncated=truncated,
            ))
            previous_sql, previous_execution, final_sql = sql, execution, sql
            if not should_continue:
                stop_reason = "policy_final" if execution.executed and decision == "final" else "budget_exhausted"
                break
        return Trajectory(
            task=task, steps=steps, final_sql=final_sql,
            total_elapsed_ms=(time.perf_counter() - started) * 1000,
            stop_reason=stop_reason,
        )
