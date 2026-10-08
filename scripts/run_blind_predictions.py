#!/usr/bin/env python3
"""Full-coverage BIRD official scorer preparation: blind policy ONLY, no Gold.

The internal SQLite Gold eligibility filter may discard difficult queries that
the official BIRD scorer must receive. This CLI instead asks the policy about
ALL 500 official Mini-Dev SELECT-only tasks and stores 500 predictions without
ever evaluating Gold SQL. The resulting JSONL can be passed to
export_bird_official.py, which enforces source ordering and calls official EX.

NO GPU work is simulated. It requires a REAL running policy endpoint for
model inference; the CPU tests use a deterministic stub.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rl_sql.agent import OpenAICompatibleClient, SqlAgentRunner
from agentic_rl_sql.context import hf_message_counter
from agentic_rl_sql.dataset import read_training_parquet, unpack_task


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


async def run_blind(args) -> dict:
    dataset = Path(args.dataset).resolve()
    tasks = [unpack_task(r) for r in read_training_parquet(dataset)]
    if not tasks:
        raise ValueError("blind inference dataset is empty")
    task_ids = [t.task_id for t in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate blind-inference task IDs")
    for task in tasks:
        if task.context_limit != args.context_limit or task.max_turns != args.max_turns:
            raise ValueError(f"task budgets do not match CLI: {task.task_id}")
        if not Path(task.db_path).is_file():
            raise FileNotFoundError(f"missing SQLite database for {task.task_id}")
    if args.require_official_mini_dev:
        if len(tasks) != 500:
            raise ValueError(f"official BIRD Mini-Dev requires 500 tasks, found {len(tasks)}")
        if [t.task_id for t in tasks] != [f"bird-eval-{i:05d}" for i in range(500)]:
            raise ValueError("official BIRD Mini-Dev requires canonical original task IDs in order")
        if not all(t.dataset == "bird-sql" and t.split == "eval" for t in tasks):
            raise ValueError("not the official BIRD Mini-Dev prepared task split")
    policy_manifest = Path(args.policy_manifest).resolve()
    metadata = json.loads(policy_manifest.read_text(encoding="utf-8"))
    if not metadata.get("weights_fingerprint_sha256") and not metadata.get("file_manifest", {}).get("weight_file_count"):
        raise ValueError("policy manifest does not identify genuine model weights")
    protocol = {
        "protocol": "blind-ungraded-v1",
        "gold_access": "never",
        "purpose": "official BIRD scorer predictions: no internal Gold eligibility exclusions",
        "dataset_sha256": hash_file(dataset),
        "task_ids_sha256": fingerprint(sorted(task_ids)),
        "task_count": len(tasks),
        "checkpoint": args.policy_checkpoint,
        "policy_manifest_sha256": hash_file(policy_manifest),
        "tokenizer": args.tokenizer,
        "model": args.model,
        "seed": args.seed,
        "temperature": args.temperature,
        "context_limit": args.context_limit,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "sql_max_rows": args.max_rows,
        "sql_timeout_seconds": args.sql_timeout,
    }
    protocol["fingerprint"] = fingerprint(protocol)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = out.with_name(out.stem + "_protocol.json")
    if out.exists():
        if not args.resume:
            raise FileExistsError("output exists; use --resume with exact same frozen protocol")
        if not protocol_path.is_file():
            raise ValueError("cannot resume without previous protocol")
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol["fingerprint"] != existing.get("fingerprint"):
            raise ValueError("resume policy/data/budget fingerprint differs from existing run")
    else:
        if args.resume:
            raise FileNotFoundError("resume requested without existing output file")
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    finished = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ident = row["task"]["task_id"]
            if ident in finished or ident not in task_ids:
                raise ValueError(f"duplicate or foreign task in resumed output: {ident}")
            finished[ident] = row
    client = OpenAICompatibleClient(
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
    )
    runner = SqlAgentRunner(
        client, token_counter=hf_message_counter(args.tokenizer),
        timeout_seconds=args.sql_timeout, max_rows=args.max_rows,
    )
    lock = asyncio.Semaphore(args.concurrency)

    async def one(task):
        async with lock:
            raw = await runner.run(task.policy_view())
            row = raw.to_dict()
            # "blind-ungraded-v1" must not imply success or evaluator usage.
            row["evaluation"] = {"protocol": "blind-ungraded-v1", "gold_access": "none"}
            if "gold_sql" in row["task"] or "reward" in row or "success" in row:
                raise AssertionError("raw blind prediction leaked private grade")
            return row

    pending = [t for t in tasks if t.task_id not in finished]
    with out.open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending), args.concurrency):
            rows = await asyncio.gather(*(one(t) for t in pending[start:start + args.concurrency]))
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
    actual = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            task_id = row["task"]["task_id"]
            if task_id in actual:
                raise ValueError(f"duplicate task after finalization: {task_id}")
            actual[task_id] = row
    if set(actual) != set(task_ids):
        raise RuntimeError("blind inference did not cover every requested task")
    result = {
        **protocol,
        "completed": len(actual),
        "trajectories_sha256": hash_file(out),
        "official_ex_scored": False,
        "gpu_training_performed": False,
    }
    manifest_path = out.with_name(out.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="sql-policy")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--policy-manifest", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--require-official-mini-dev", action="store_true")
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sql-timeout", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_turns < 1 or args.concurrency < 1 or args.context_limit < 1 or args.max_rows < 1 or args.sql_timeout <= 0:
        parser.error("all execution budgets must be positive")
    result = asyncio.run(run_blind(args))
    print(json.dumps({"completed": result["completed"],
                      "official_ex_scored": False, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
