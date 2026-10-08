"""Real SQLite + actual Agent wrapper + fake HTTP gateway.

A stub policy is used ONLY to verify observable inputs, AGL reward-event
wire format, and post-hoc correctness isolation. This is not a model benchmark.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from agent import sql_agent_entrypoint as entrypoint


@pytest.mark.asyncio
async def test_agent_lightning_reward_event_contract_and_no_gold_leak(tmp_path, monkeypatch):
    db = tmp_path / "items.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE items(id INTEGER, value TEXT);"
            "INSERT INTO items VALUES (1,'a'), (2,'b');"
        )
    seen = []
    events = []

    class DummyPolicy:
        def __init__(self, **kwargs):
            pass
        async def complete(self, messages):
            seen.append(json.dumps(messages))
            return '{"sql":"SELECT value FROM items WHERE id=2","decision":"final"}'

    class DummyResponse:
        def raise_for_status(self):
            return None

    class FakeHttp:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        async def post(self, url, *, json, headers):
            events.append((url, json, headers))
            return DummyResponse()

    monkeypatch.setattr(entrypoint, "OpenAICompatibleClient", DummyPolicy)
    monkeypatch.setattr(entrypoint, "hf_message_counter", lambda _: lambda msgs: 128)
    monkeypatch.setattr(entrypoint.httpx, "AsyncClient", FakeHttp)
    monkeypatch.setenv("AGL_OPENAI_BASE_URL", "http://localhost:8181/proxy/v1")
    monkeypatch.setenv("AGL_KEY", "testing-local-only")
    monkeypatch.setenv("AGL_EVENT_URL", "http://localhost:8181/api/events")

    task = {
        "task_id": "p0", "dataset": "toy", "split": "test",
        "db_id": "items", "db_path": str(db), "question": "Find id 2",
        "gold_sql": "SELECT value FROM items WHERE id=2", "max_turns": 2,
    }
    async def run(gold):
        task["gold_sql"] = gold
        monkeypatch.setenv("TASK_JSON", json.dumps(task))
        await entrypoint.Agent().run()

    # A change in the Gold SQL changes rewards, but not policy prompts.
    await run("SELECT value FROM items WHERE id=2")
    await run("SELECT value FROM items WHERE id=1")
    assert len(seen) == len(events) == 2
    assert seen[0] == seen[1]
    assert "gold_sql" not in seen[0]
    assert events[0][1]["data"]["value"] == 1.0
    assert events[1][1]["data"]["value"] <= 0.5
    assert events[0][1]["event_type"] == "reward"
    assert events[0][1]["data"]["metadata"]["evaluation_protocol"] == "blind-final-v1"
    assert all(ev[2]["Authorization"] == "Bearer testing-local-only" for ev in events)
