from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx

from agentic_rl_sql.agent import OpenAICompatibleClient, SqlAgentRunner
from agentic_rl_sql.types import SqlTask


class Agent:
    """Agent Lightning local/Kubernetes agent entrypoint."""

    async def run(self) -> None:
        task_payload = json.loads(os.environ["TASK_JSON"])
        task_payload["db_path"] = os.environ.get("DB_PATH", task_payload.get("db_path"))
        task_payload["max_turns"] = int(os.environ.get("MAX_TURNS", task_payload.get("max_turns", 3)))
        task = SqlTask.from_dict(task_payload)
        if not Path(task.db_path).exists():
            raise FileNotFoundError(task.db_path)

        client = OpenAICompatibleClient(
            base_url=os.environ["AGL_OPENAI_BASE_URL"],
            api_key=os.environ["AGL_KEY"],
            model="auto",
            temperature=float(os.environ.get("ROLLOUT_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("ROLLOUT_MAX_TOKENS", "2048")),
        )
        runner = SqlAgentRunner(
            client,
            max_schema_chars=int(os.environ.get("MAX_SCHEMA_CHARS", "12000")),
            timeout_seconds=float(os.environ.get("SQL_TIMEOUT_SECONDS", "8")),
            max_rows=int(os.environ.get("SQL_MAX_ROWS", "5000")),
        )
        trajectory = await runner.run(task)
        event = {
            "event_type": "reward",
            "data": {
                "value": trajectory.reward.total,
                "metadata": {
                    "task_id": task.task_id,
                    "dataset": task.dataset,
                    "success": trajectory.success,
                    "turns": len(trajectory.steps),
                    "final_sql": trajectory.final_sql,
                    "reward": trajectory.reward.to_dict(),
                    "steps": [step.to_dict() for step in trajectory.steps],
                },
            },
        }
        async with httpx.AsyncClient(timeout=20.0) as http:
            response = await http.post(
                os.environ["AGL_EVENT_URL"],
                json=event,
                headers={"Authorization": f"Bearer {os.environ['AGL_KEY']}"},
            )
            response.raise_for_status()


if __name__ == "__main__":
    asyncio.run(Agent().run())
