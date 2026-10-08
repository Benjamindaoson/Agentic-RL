import sqlite3

import pytest

from agentic_rl_sql.agent import SqlAgentRunner
from agentic_rl_sql.evaluator import InvalidGoldSQL, SqlEvaluator
from agentic_rl_sql.types import SqlTask


class Stub:
    async def complete(self, messages):
        return '{"sql":"SELECT value FROM items WHERE id=2","decision":"final"}'


@pytest.fixture
def sql_task(tmp_path):
    db = tmp_path / "x.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE items(id INTEGER, value TEXT); INSERT INTO items VALUES (1,'a'), (2,'b');")
    return SqlTask("one", "toy", "test", "items", str(db), "Value with id 2?",
                   "SELECT value FROM items WHERE id=2")


@pytest.mark.asyncio
async def test_evaluator_only_after_frozen_rollout(sql_task):
    trajectory = await SqlAgentRunner(Stub()).run(sql_task.policy_view())
    blind = trajectory.to_dict()
    assert "gold_sql" not in str(blind)
    assert "reward" not in blind and "success" not in blind
    result = SqlEvaluator().evaluate(sql_task, trajectory)
    assert result["success"] and result["first_turn_success"]
    assert result["reward"]["total"] == 1.0
    assert result["evaluation"]["gold_access"] == "post_rollout_only"
    assert "gold_sql" not in str(result)


@pytest.mark.asyncio
async def test_invalid_gold_rejected(sql_task):
    trajectory = await SqlAgentRunner(Stub()).run(sql_task.policy_view())
    sql_task.gold_sql = "SELECT missing FROM items"
    with pytest.raises(InvalidGoldSQL):
        SqlEvaluator().evaluate(sql_task, trajectory)


@pytest.mark.asyncio
async def test_correct_first_turn_does_not_force_stop(sql_task):
    class Inspect:
        def __init__(self):
            self.calls = 0
            self.prompts = []
        async def complete(self, messages):
            self.prompts.append(messages)
            self.calls += 1
            return ('{"sql":"SELECT value FROM items WHERE id=2","decision":"inspect"}'
                    if self.calls == 1 else
                    '{"sql":"SELECT value FROM items WHERE id=2","decision":"final"}')
    sql_task.max_turns = 2
    client = Inspect()
    trajectory = await SqlAgentRunner(client).run(sql_task.policy_view())
    assert len(trajectory.steps) == 2
    assert "correctness unknown" in client.prompts[1][1]["content"].lower()
    result = SqlEvaluator().evaluate(sql_task, trajectory)
    assert result["evaluation"]["per_turn_execution_match_posthoc"] == [True, True]
