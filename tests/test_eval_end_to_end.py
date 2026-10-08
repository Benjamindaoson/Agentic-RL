"""CPU end-to-end: actual SQLite -> blind rollout -> post-hoc scoring -> paired stats.

Stub policy is deliberately NOT a pretrained model. These test scores are never
reported as empirical LLM benchmark results.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rl_sql.dataset import write_parquet
from agentic_rl_sql.types import SqlTask
from scripts import compare_experiments, run_rollouts


class ReplayPolicy:
    def __init__(self, sql):
        self.sql = sql
        self.messages = []

    async def complete(self, messages):
        self.messages.append(messages)
        return json.dumps({"sql": self.sql, "decision": "final"})


def args_for(parquet, output, model_revision="first"):
    return SimpleNamespace(
        dataset=str(parquet), output=str(output), base_url="http://unused/v1",
        api_key="NO_KEY", model="sql-policy", tokenizer="offline-test-tokenizer",
        policy_checkpoint=model_revision, policy_manifest="", seed=42,
        temperature=0.0, max_tokens=128, context_limit=2048, max_turns=1,
        explicit_check=False, max_schema_chars=12000, sql_timeout=5.0,
        max_rows=100, concurrency=2, limit=0, resume=False, overwrite=False,
    )


@pytest.mark.asyncio
async def test_offline_sql_evaluation_and_paired_report(tmp_path, monkeypatch):
    db = tmp_path / "tiny.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE items(id INTEGER, value TEXT); "
            "INSERT INTO items VALUES (1,'a'),(2,'b');"
        )
    tasks = [
        SqlTask(f"item-{i}", "toy", "test", "items", str(db),
                f"Get value for id {i}", f"SELECT value FROM items WHERE id={i}",
                max_turns=1, context_limit=2048)
        for i in (1, 2)
    ]
    parquet = tmp_path / "frozen_test.parquet"
    assert write_parquet(tasks, parquet) == 2

    monkeypatch.setattr(run_rollouts, "hf_message_counter", lambda *_: lambda messages: 50)
    policy_a = ReplayPolicy("SELECT value FROM items WHERE id=3")
    policy_b = ReplayPolicy("SELECT value FROM items WHERE id=1")

    base_path = tmp_path / "base_trajectories.jsonl"
    grpo_path = tmp_path / "grpo_trajectories.jsonl"
    monkeypatch.setattr(run_rollouts, "OpenAICompatibleClient", lambda **kwargs: policy_a)
    await run_rollouts.main_async(args_for(parquet, base_path, "base-revision"))
    monkeypatch.setattr(run_rollouts, "OpenAICompatibleClient", lambda **kwargs: policy_b)
    await run_rollouts.main_async(args_for(parquet, grpo_path, "synthetic-candidate"))

    base = compare_experiments.load_run(base_path)
    candidate = compare_experiments.load_run(grpo_path)
    compare_experiments.validate_paired(base, candidate)
    result = compare_experiments.build_comparison(
        {"base": base, "grpo": candidate}, seed=42, n_boot=100
    )
    assert result["runs"]["base"]["task_accuracy"] == 0
    assert result["runs"]["grpo"]["task_accuracy"] == 0.5
    assert result["paired_vs_base"]["grpo"]["gain_pp"] == 50
    assert all("gold_sql" not in row["task"] for row in candidate[2].values())
    assert all(row["evaluation"]["gold_access"] == "post_rollout_only" for row in candidate[2].values())

    # A different sampling budget cannot resume or be mixed with the frozen run.
    rerun = args_for(parquet, grpo_path, "synthetic-candidate")
    rerun.max_turns = 3
    rerun.resume = True
    with pytest.raises(RuntimeError, match="mismatch"):
        await run_rollouts.main_async(rerun)
