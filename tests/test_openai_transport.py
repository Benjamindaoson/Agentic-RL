"""Real OpenAI SDK request/response path, mocked HTTP transport, real SQLite.

Exercises the actual JSON wire protocol used by vLLM and Agent Lightning,
without needing a model or GPU.
"""
from __future__ import annotations

import json
import sqlite3

import httpx
import pytest
from openai import AsyncOpenAI

from agentic_rl_sql.agent import OpenAICompatibleClient, SqlAgentRunner
from agentic_rl_sql.evaluator import SqlEvaluator
from agentic_rl_sql.types import SqlTask


@pytest.mark.asyncio
async def test_openai_wire_format_and_blind_execution_recovery(tmp_path):
    db = tmp_path / "items.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE items(id INTEGER, value TEXT);"
                           "INSERT INTO items VALUES(1,'a'),(2,'b');")
    task = SqlTask(
        "wire", "toy", "test", "items", str(db), "Find the value with id 2",
        "SELECT value FROM items WHERE id=2", max_turns=2, context_limit=2048,
    )
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content)
        assert payload["model"] == "sql-policy"
        assert payload["temperature"] == 0.0
        prompt = json.dumps(payload["messages"])
        assert "gold_sql" not in prompt
        assert "expected answer did not match" not in prompt.lower()
        seen.append(payload)
        # Controlled *environmental* failure followed by a successful rewrite.
        sql = "SELECT nonexistent FROM items" if len(seen) == 1 else "SELECT value FROM items WHERE id=2"
        return httpx.Response(200, json={
            "id": f"mock-{len(seen)}", "object": "chat.completion",
            "created": 1234567890, "model": "sql-policy",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps({
                             "sql": sql, "decision": "final",
                         })}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async_llm = AsyncOpenAI(
            api_key="offline-testing-only", base_url="http://vllm.invalid/v1",
            http_client=http, max_retries=0,
        )
        client = OpenAICompatibleClient(
            base_url="http://vllm.invalid/v1", api_key="offline-testing-only",
            model="sql-policy", temperature=0.0,
        )
        client._client = async_llm
        runner = SqlAgentRunner(client, token_counter=lambda _: 100)
        trace = await runner.run(task.policy_view())
        scored = SqlEvaluator().evaluate(task, trace)
    assert len(seen) == len(trace.steps) == 2
    assert trace.stop_reason == "policy_final"
    assert "execution_error" in seen[1]["messages"][-1]["content"]
    assert scored["success"] and not scored["first_turn_success"]
    assert trace.to_dict()["task"].get("gold_sql") is None
