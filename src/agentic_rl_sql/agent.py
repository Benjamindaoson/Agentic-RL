from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .db import render_schema
from .execution import execution_match
from .prompts import build_check_messages, build_messages
from .reward import compute_reward
from .types import ExecutionResult, SqlTask, Trajectory, TrajectoryStep

SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.I | re.S)
JSON_RE = re.compile(r"\{.*\}", re.S)


class AsyncChatClient(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(slots=True)
class OpenAICompatibleClient:
    base_url: str
    api_key: str
    model: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 2048
    _client: Any = field(init=False, default=None, repr=False)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        from openai import AsyncOpenAI

        if self._client is None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=3)
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


def parse_sql_output(text: str) -> str:
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        sql = obj.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql.strip()
    except Exception:
        match = JSON_RE.search(raw)
        if match:
            try:
                obj = json.loads(match.group(0))
                sql = obj.get("sql")
                if isinstance(sql, str) and sql.strip():
                    return sql.strip()
            except Exception:
                pass
    fence = SQL_FENCE_RE.search(raw)
    if fence:
        return fence.group(1).strip()
    candidate = raw.strip("`").strip()
    if candidate.upper().startswith(("SELECT", "WITH")):
        return candidate
    raise ValueError("model output did not contain a SQL query")


def parse_checker_feedback(text: str) -> str:
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        value = obj.get("feedback")
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        match = JSON_RE.search(raw)
        if match:
            try:
                value = json.loads(match.group(0)).get("feedback")
                if isinstance(value, str) and value.strip():
                    return value.strip()
            except Exception:
                pass
    return raw[:1000] if raw else "Re-check the query semantics."


def correction_feedback(execution: ExecutionResult) -> str:
    if not execution.valid:
        return "The previous output was not valid SQL. Return one parseable SELECT/WITH statement."
    if not execution.safe:
        return "The previous query violated the read-only policy. Use only SELECT/WITH."
    if not execution.executed:
        return f"The previous query failed with {execution.error_type}: {execution.error_message}."
    return "The query executed, but its result did not answer the question correctly. Re-check joins, filters, aggregation, grouping, and ordering."


class SqlAgentRunner:
    def __init__(self, client: AsyncChatClient, *, max_schema_chars: int = 12000, timeout_seconds: float = 8.0, max_rows: int = 5000) -> None:
        self.client = client
        self.max_schema_chars = max_schema_chars
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    async def run(self, task: SqlTask) -> Trajectory:
        started = time.perf_counter()
        schema = render_schema(task.db_path, max_chars=self.max_schema_chars)
        steps: list[TrajectoryStep] = []
        previous_sql: str | None = None
        previous_execution: ExecutionResult | None = None
        feedback = ""
        success = False
        final_sql: str | None = None
        for turn in range(1, max(task.max_turns, 1) + 1):
            messages = build_messages(task, schema, turn, previous_sql, previous_execution, feedback)
            raw = await self.client.complete(messages)
            try:
                sql = parse_sql_output(raw)
                matched, execution, _ = execution_match(task.db_path, sql, task.gold_sql, self.timeout_seconds, self.max_rows)
            except ValueError as exc:
                sql = None
                execution = ExecutionResult(sql="", valid=False, safe=False, executed=False, error_type="output_parse_error", error_message=str(exc))
                matched = False
            feedback = correction_feedback(execution)
            checker_output = None
            if not matched and sql and bool(task.metadata.get("explicit_check", False)) and execution.safe:
                checker_output = await self.client.complete(build_check_messages(task, schema, sql, execution))
                feedback = parse_checker_feedback(checker_output)
            steps.append(TrajectoryStep(turn, "initial" if turn == 1 else "rewrite", raw, sql, execution, matched, feedback, checker_output))
            previous_sql, previous_execution, final_sql = sql, execution, sql
            if matched:
                success = True
                break
        reward = compute_reward(steps[-1].execution, success, len(steps))
        return Trajectory(task, steps, reward, success, final_sql, (time.perf_counter() - started) * 1000)
