import json
import sqlite3

import pytest

from agentic_rl_sql.agent import SqlAgentRunner
from agentic_rl_sql.types import SqlTask
from scripts.compare_experiments import load_run
from scripts.run_rollouts import fingerprint, sha256_file


@pytest.mark.asyncio
async def test_zero_turn_budget_is_rejected(tmp_path):
    db = tmp_path / "db.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (id INTEGER)")
    task = SqlTask("t", "toy", "test", "db", str(db), "q", "SELECT 1", max_turns=0)
    class Client:
        async def complete(self, messages):
            raise AssertionError("should not call model")
    with pytest.raises(ValueError, match="max_turns"):
        await SqlAgentRunner(Client()).run(task.policy_view())


def test_comparison_rejects_forged_task_set(tmp_path):
    raw = tmp_path / "eval.jsonl"
    row = {"task": {"task_id": "changed"}, "evaluation": {"protocol": "blind-final-v1"}, "success": False}
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    protocol = {
        "protocol": "blind-final-v1", "oracle_access": "post_rollout_only",
        "task_count": 1, "task_ids_sha256": fingerprint(["original"]),
        "fingerprint": "same",
    }
    raw.with_name("eval_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    raw.with_name("eval_metrics.json").write_text(
        json.dumps({"samples": 1, "trajectories_sha256": sha256_file(raw),
                    "protocol_fingerprint": "same"}), encoding="utf-8")
    with pytest.raises(ValueError, match="task IDs"):
        load_run(raw)
