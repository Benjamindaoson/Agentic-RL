#!/usr/bin/env python3
"""Blind evaluation with dataset hashes, resume validation and row-level evidence."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rl_sql.agent import OpenAICompatibleClient, SqlAgentRunner
from agentic_rl_sql.context import hf_message_counter
from agentic_rl_sql.dataset import read_training_parquet, unpack_task
from agentic_rl_sql.evaluator import InvalidGoldSQL, SqlEvaluator
from agentic_rl_sql.metrics import summarize_trajectories


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def main_async(args):
    data_file = Path(args.dataset).resolve()
    records = read_training_parquet(data_file)
    if args.limit:
        records = records[:args.limit]
    tasks = [unpack_task(row) for row in records]
    if not tasks:
        raise ValueError("evaluation dataset is empty")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs invalidate paired evaluation")
    for task in tasks:
        if args.context_limit:
            task.context_limit = args.context_limit
        if args.max_turns:
            task.max_turns = args.max_turns
        if args.explicit_check is not None:
            task.metadata["explicit_check"] = args.explicit_check
    if len({(t.context_limit, t.max_turns, bool(t.metadata.get("explicit_check"))) for t in tasks}) != 1:
        raise ValueError("evaluation tasks must have identical context/turn/check budgets")
    budget = {
        "context_limit": tasks[0].context_limit,
        "max_turns": tasks[0].max_turns,
        "explicit_check": bool(tasks[0].metadata.get("explicit_check")),
        "max_model_output_tokens": args.max_tokens,
        "sql_timeout_seconds": args.sql_timeout,
        "sql_max_rows": args.max_rows,
        "max_schema_chars": args.max_schema_chars,
    }
    protocol = {
        "protocol": "blind-final-v1",
        "oracle_access": "post_rollout_only",
        "dataset_sha256": sha256_file(data_file),
        "task_ids_sha256": fingerprint(sorted(ids)),
        "task_count": len(tasks),
        "model": args.model, "tokenizer": args.tokenizer,
        "policy_checkpoint": args.policy_checkpoint,
        "policy_identity_sha256": sha256_file(Path(args.policy_manifest)) if args.policy_manifest else None,
        "seed": args.seed, "temperature": args.temperature,
        "budget": budget,
    }
    protocol["fingerprint"] = fingerprint(protocol)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = output.with_name(output.stem + "_protocol.json")
    if args.resume and output.exists():
        if not protocol_path.exists():
            raise RuntimeError("resume requires existing protocol manifest; refusing mixed results")
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != protocol["fingerprint"]:
            raise RuntimeError("resume config/dataset/model mismatch; refusing mixed trajectories")
    else:
        if args.resume and not output.exists():
            raise FileNotFoundError("resume requested but no prior rollout file exists")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} already exists; choose --overwrite or --resume")
        output.write_text("", encoding="utf-8")
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    done = set()
    if args.resume:
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                key = str(json.loads(line)["task"]["task_id"])
                if key in done:
                    raise ValueError(f"duplicate resume task: {key}")
                done.add(key)
        if not done.issubset(set(ids)):
            raise ValueError("resume contains task IDs not in current evaluation dataset")
    pending = [task for task in tasks if task.task_id not in done]
    token_counter = hf_message_counter(args.tokenizer)
    client = OpenAICompatibleClient(
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        temperature=args.temperature, max_tokens=args.max_tokens,
        seed=args.seed,
    )
    runner = SqlAgentRunner(
        client, max_schema_chars=args.max_schema_chars,
        timeout_seconds=args.sql_timeout, max_rows=args.max_rows,
        token_counter=token_counter,
    )
    evaluator = SqlEvaluator(timeout_seconds=args.sql_timeout, max_rows=args.max_rows)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(task):
        async with semaphore:
            try:
                raw = await runner.run(task.policy_view())
                return evaluator.evaluate(task, raw)
            except InvalidGoldSQL:
                raise
            except Exception as exc:
                return {
                    "task": task.policy_view().to_dict(), "steps": [],
                    "reward": {"total": -1.0}, "success": False,
                    "first_turn_success": False, "final_sql": None,
                    "total_elapsed_ms": 0.0, "stop_reason": "runner_error",
                    "runner_error": f"{type(exc).__name__}: {exc}",
                }

    with output.open("a", encoding="utf-8") as sink:
        for start in range(0, len(pending), args.concurrency * 4):
            results = await asyncio.gather(
                *(one(task) for task in pending[start:start + args.concurrency * 4]),
            )
            for result in results:
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()

    all_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(all_rows) != len(tasks):
        raise RuntimeError(f"expected {len(tasks)} evaluated tasks but found {len(all_rows)}")
    if sorted(r["task"]["task_id"] for r in all_rows) != sorted(ids):
        raise RuntimeError("output tasks no longer match protocol task set")
    metrics = summarize_trajectories(all_rows)
    metrics.update({
        "protocol": "blind-final-v1",
        "protocol_fingerprint": protocol["fingerprint"],
        "dataset_sha256": protocol["dataset_sha256"],
        "task_ids_sha256": protocol["task_ids_sha256"],
        "model": args.model, "policy_checkpoint": args.policy_checkpoint,
        "policy_identity_sha256": protocol["policy_identity_sha256"],
        "tokenizer": args.tokenizer, "temperature": args.temperature,
        "seed": args.seed, "budget": budget, "trajectories_sha256": sha256_file(output),
    })
    metrics_path = output.with_name(output.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if metrics["runner_error_count"]:
        raise SystemExit(f"evaluation had {metrics['runner_error_count']} runner errors; not valid for publication")


def main():
    ap = argparse.ArgumentParser(description="Evaluate policy with NO answer-driven continuation.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="sql-policy")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--policy-checkpoint", required=True, help="Immutable checkpoint identifier or base revision")
    ap.add_argument("--policy-manifest", default="", help="Local base_model_identity.json or exported HF export_manifest.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--context-limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--explicit-check", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--max-schema-chars", type=int, default=12000)
    ap.add_argument("--sql-timeout", type=float, default=8.0)
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.policy_manifest:
        payload = json.loads(Path(args.policy_manifest).read_text(encoding="utf-8"))
        if not (payload.get("weights_fingerprint_sha256") or payload.get("file_manifest", {}).get("weight_file_count")):
            ap.error("--policy-manifest does not describe hashed weight files")
    if args.policy_checkpoint.strip() == "":
        ap.error("checkpoint identifier must not be blank")
    if args.limit < 0:
        ap.error("--limit must not be negative")
    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    if args.resume and args.overwrite:
        ap.error("choose --resume or --overwrite, not both")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
