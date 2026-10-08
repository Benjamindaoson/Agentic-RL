import sqlite3

import pytest

from agentic_rl_sql.agent import SqlAgentRunner, parse_sql_output
from agentic_rl_sql.evaluator import SqlEvaluator
from agentic_rl_sql.types import SqlTask


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    async def complete(self, messages):
        self.messages.append(messages)
        return next(self.outputs)


def make_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE items(id INTEGER, value TEXT); INSERT INTO items VALUES (1,'a'),(2,'b');")
    conn.commit()
    conn.close()


def task_for(db, *, gold="SELECT value FROM items WHERE id=2", max_turns=2, check=False):
    return SqlTask(
        task_id="toy", dataset="toy", split="test", db_id="toy",
        db_path=str(db), question="What value has id 2?", gold_sql=gold,
        max_turns=max_turns, metadata={"explicit_check": check},
    )


def test_parse_sql_output_variants():
    assert parse_sql_output('{"sql":"SELECT 1"}') == "SELECT 1"
    assert parse_sql_output(chr(96) * 3 + "sql\nSELECT 2\n" + chr(96) * 3) == "SELECT 2"


@pytest.mark.asyncio
async def test_agent_uses_visible_execution_error_to_recover(tmp_path):
    db = tmp_path / "toy.sqlite"
    make_db(db)
    task = task_for(db)
    client = FakeClient([
        '{"sql":"SELECT nonexistent FROM items","decision":"final"}',
        '{"sql":"SELECT value FROM items WHERE id=2","decision":"final"}',
    ])
    trajectory = await SqlAgentRunner(client).run(task.policy_view())
    assert len(trajectory.steps) == 2
    assert trajectory.stop_reason == "policy_final"
    assert "execution_error" in client.messages[1][1]["content"]
    assert "success" not in trajectory.to_dict()
    assert SqlEvaluator().evaluate(task, trajectory)["success"]


@pytest.mark.asyncio
async def test_explicit_checker_only_on_policy_inspection(tmp_path):
    db = tmp_path / "toy.sqlite"
    make_db(db)
    task = task_for(db, check=True)
    client = FakeClient([
        '{"sql":"SELECT value FROM items WHERE id=1","decision":"inspect"}',
        '{"feedback":"Double-check the requested id."}',
        '{"sql":"SELECT value FROM items WHERE id=2","decision":"final"}',
    ])
    trajectory = await SqlAgentRunner(client).run(task.policy_view())
    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].checker_output is not None
    assert "known to be incorrect" not in str(client.messages).lower()
    assert SqlEvaluator().evaluate(task, trajectory)["success"]


@pytest.mark.asyncio
async def test_changing_gold_cannot_change_policy_or_termination(tmp_path):
    db = tmp_path / "toy.sqlite"
    make_db(db)
    original = task_for(db, gold="SELECT value FROM items WHERE id=2")
    changed = task_for(db, gold="SELECT value FROM items WHERE id=1")
    output = ['{"sql":"SELECT value FROM items WHERE id=1","decision":"final"}']
    a, b = FakeClient(output), FakeClient(output)
    ta = await SqlAgentRunner(a).run(original.policy_view())
    tb = await SqlAgentRunner(b).run(changed.policy_view())
    assert a.messages == b.messages
    assert ta.final_sql == tb.final_sql and ta.stop_reason == tb.stop_reason
    assert len(ta.steps) == len(tb.steps) == 1
    assert ta.task.to_dict() == tb.task.to_dict()
    assert SqlEvaluator().evaluate(original, ta)["success"] is False
    assert SqlEvaluator().evaluate(changed, tb)["success"] is True


@pytest.mark.asyncio
async def test_runner_rejects_private_task(tmp_path):
    db = tmp_path / "toy.sqlite"
    make_db(db)
    with pytest.raises(TypeError, match="PolicyTask only"):
        await SqlAgentRunner(FakeClient([])).run(task_for(db))
