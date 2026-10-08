from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx

from agentic_rl_sql.agent import OpenAICompatibleClient, SqlAgentRunner
from agentic_rl_sql.context import hf_message_counter
from agentic_rl_sql.evaluator import SqlEvaluator
from agentic_rl_sql.types import SqlTask


class Agent:
    """Agent Lightning entrypoint; policy receives only a blind PolicyTask."""

    async def run(self) -> None:
        task_payload = json.loads(os.environ["TASK_JSON"])
        task_payload["db_path"] = os.environ.get("DB_PATH", task_payload.get("db_path"))
        task_payload["max_turns"] = int(os.environ.get("MAX_TURNS", task_payload.get("max_turns", 3)))
        if os.environ.get("POLICY_PROMPT_TOKEN_LIMIT"):
            task_payload["context_limit"] = int(os.environ["POLICY_PROMPT_TOKEN_LIMIT"])
        task = SqlTask.from_dict(task_payload)
        if not Path(task.db_path).exists():
            raise FileNotFoundError(task.db_path)
        client = OpenAICompatibleClient(
            base_url=os.environ["AGL_OPENAI_BASE_URL"], api_key=os.environ["AGL_KEY"],
            model="auto", temperature=float(os.environ.get("ROLLOUT_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("ROLLOUT_MAX_TOKENS", "1024")),
        )
        runner = SqlAgentRunner(
            client, max_schema_chars=int(os.environ.get("MAX_SCHEMA_CHARS", "12000")),
            timeout_seconds=float(os.environ.get("SQL_TIMEOUT_SECONDS", "8")),
            max_rows=int(os.environ.get("SQL_MAX_ROWS", "5000")),
            token_counter=hf_message_counter(
                os.environ.get("POLICY_TOKENIZER_PATH", "Qwen/Qwen2.5-Coder-3B-Instruct")
            ),
        )
        trajectory = await runner.run(task.policy_view())
        graded = SqlEvaluator(
            timeout_seconds=runner.timeout_seconds, max_rows=runner.max_rows,
        ).evaluate(task, trajectory)
        event = {
            "event_type": "reward",
            "data": {
                "value": graded["reward"]["total"],
                "metadata": {
                    "task_id": task.task_id, "dataset": task.dataset,
                    "success": graded["success"],
                    "first_turn_success": graded["first_turn_success"],
                    "turns": len(trajectory.steps), "final_sql": trajectory.final_sql,
                    "stop_reason": trajectory.stop_reason,
                    "reward": graded["reward"], "steps": graded["steps"],
                    "evaluation_protocol": "blind-final-v1",
                },
            },
        }
        async with httpx.AsyncClient(timeout=20.0) as http:
            response = await http.post(
                os.environ["AGL_EVENT_URL"], json=event,
                headers={"Authorization": f"Bearer {os.environ['AGL_KEY']}"},
            )
            response.raise_for_status()


if __name__ == "__main__":
    asyncio.run(Agent().run())
