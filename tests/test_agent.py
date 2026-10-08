import sqlite3

import pytest

from agentic_rl_sql.agent import SqlAgentRunner, parse_sql_output
from agentic_rl_sql.types import SqlTask


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    async def complete(self, messages):
        return next(self.outputs)


def make_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE items(id INTEGER, value TEXT); INSERT INTO items VALUES (1,'a'),(2,'b');")
    conn.commit()
    conn.close()


def test_parse_sql_output_variants():
    assert parse_sql_output('{"sql":"SELECT 1"}') == "SELECT 1"
    assert parse_sql_output("```sql\nSELECT 2\n```") == "SELECT 2"


@pytest.mark.asyncio
async def test_agent_uses_execution_feedback_to_recover(tmp_path):
    db = tmp_path / "toy.sqlite"
    make_db(db)
    task = SqlTask(
        task_id="toy",
        dataset="toy",
        split="test",
        db_id="toy",
        db_path=str(db),
        question="What value has id 2?",
        gold_sql="SELECT value FROM items WHERE id=2",
        max_turns=2,
    )
    client = FakeClient([
        '{"sql":"SELECT value FROM items WHERE id=1"}',
        '{"sql":"SELECT value FROM items WHERE id=2"}',
    ])
    trajectory = await SqlAgentRunner(client).run(task)
    assert trajectory.success
    assert len(trajectory.steps) == 2
    assert trajectory.reward.total > 0.9


@pytest.mark.asyncio
async def test_explicit_checker_adds_feedback_call(tmp_path):
    db = tmp_path / "toy.sqlite"
    make_db(db)
    task = SqlTask(
        task_id="toy-check",
        dataset="toy",
        split="test",
        db_id="toy",
        db_path=str(db),
        question="What value has id 2?",
        gold_sql="SELECT value FROM items WHERE id=2",
        max_turns=2,
        metadata={"explicit_check": True},
    )
    client = FakeClient([
        '{"sql":"SELECT value FROM items WHERE id=1"}',
        '{"feedback":"The filter uses the wrong id."}',
        '{"sql":"SELECT value FROM items WHERE id=2"}',
    ])
    trajectory = await SqlAgentRunner(client).run(task)
    assert trajectory.success
    assert trajectory.steps[0].checker_output is not None
