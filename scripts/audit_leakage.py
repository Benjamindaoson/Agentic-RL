#!/usr/bin/env python3
"""Metamorphic audit: gold-label changes must not change policy behavior."""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rl_sql.agent import SqlAgentRunner
from agentic_rl_sql.evaluator import SqlEvaluator
from agentic_rl_sql.types import SqlTask


class ReplayPolicy:
    def __init__(self):
        self.prompts = []
        self.outputs = iter([
            '{"sql":"SELECT value FROM items WHERE id=1","decision":"inspect"}',
            '{"sql":"SELECT value FROM items WHERE id=2","decision":"final"}',
        ])

    async def complete(self, messages):
        self.prompts.append(messages)
        return next(self.outputs)


async def audit():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "items.sqlite"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                "CREATE TABLE items(id INTEGER,value TEXT); "
                "INSERT INTO items VALUES(1,'a'),(2,'b');"
            )
        def task(gold):
            return SqlTask(
                task_id="gold-invariance", dataset="synthetic", split="audit",
                db_id="items", db_path=str(db), question="Find id 2",
                gold_sql=gold, max_turns=2,
            )
        a = task("SELECT value FROM items WHERE id=2")
        b = task("SELECT value FROM items WHERE id=1")
        client_a, client_b = ReplayPolicy(), ReplayPolicy()
        result_a = await SqlAgentRunner(client_a).run(a.policy_view())
        result_b = await SqlAgentRunner(client_b).run(b.policy_view())
        policy_invariant = (
            client_a.prompts == client_b.prompts
            and [s.sql for s in result_a.steps] == [s.sql for s in result_b.steps]
            and result_a.stop_reason == result_b.stop_reason
            and len(result_a.steps) == len(result_b.steps)
        )
        public_safe = "gold_sql" not in json.dumps(result_a.to_dict())
        grader_separated = (
            SqlEvaluator().evaluate(a, result_a)["success"] is True
            and SqlEvaluator().evaluate(b, result_b)["success"] is False
        )
        no_oracle_prompt = not any(
            "result did not match" in str(messages).lower()
            or "known to be incorrect" in str(messages).lower()
            for messages in client_a.prompts + client_b.prompts
        )
        checks = {
            "gold_mutation_does_not_change_policy": policy_invariant,
            "private_gold_not_serialized_into_rollout": public_safe,
            "posthoc_grade_does_change": grader_separated,
            "no_oracle_correctness_prompts": no_oracle_prompt,
        }
        return {
            "protocol": "blind-final-v1",
            "audit_type": "deterministic_metamorphic",
            "checks": checks, "passed": all(checks.values()),
            "note": "Synthetic deterministic audit. Does not prove GPU training or model accuracy.",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = asyncio.run(audit())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
