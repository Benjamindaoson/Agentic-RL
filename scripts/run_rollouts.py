#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rl_sql.agent import OpenAICompatibleClient, SqlAgentRunner
from agentic_rl_sql.dataset import read_training_parquet, unpack_task
from agentic_rl_sql.metrics import summarize_trajectories


async def main_async(args):
    records = read_training_parquet(args.dataset)
    if args.limit:
        records = records[: args.limit]
    tasks = [unpack_task(record) for record in records]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    if args.resume and output.exists():
        with output.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    completed.add(str(row.get("task", {}).get("task_id")))
    tasks = [task for task in tasks if task.task_id not in completed]

    client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    runner = SqlAgentRunner(
        client,
        max_schema_chars=args.max_schema_chars,
        timeout_seconds=args.sql_timeout,
        max_rows=args.max_rows,
    )
    sem = asyncio.Semaphore(args.concurrency)

    async def one(task):
        async with sem:
            try:
                return await runner.run(task)
            except Exception as exc:
                return exc

    all_rows = []
    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as sink:
        for start in range(0, len(tasks), args.concurrency * 4):
            batch = tasks[start : start + args.concurrency * 4]
            results = await asyncio.gather(*(one(task) for task in batch))
            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    row = {
                        "task": task.to_dict(),
                        "steps": [],
                        "reward": {"total": -1.0},
                        "success": False,
                        "final_sql": None,
                        "total_elapsed_ms": 0.0,
                        "runner_error": repr(result),
                    }
                else:
                    row = result.to_dict()
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                all_rows.append(row)

    if args.resume and output.exists():
        all_rows = []
        with output.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_rows.append(json.loads(line))
    metrics = summarize_trajectories(all_rows)
    metrics.update({
        "dataset": str(Path(args.dataset).resolve()),
        "model": args.model,
        "base_url": args.base_url,
    })
    metrics_path = output.with_name(output.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Run direct SQL-agent rollouts against an OpenAI-compatible policy endpoint.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-schema-chars", type=int, default=12000)
    ap.add_argument("--sql-timeout", type=float, default=8.0)
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
