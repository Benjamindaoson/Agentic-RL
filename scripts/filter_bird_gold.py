#!/usr/bin/env python3
"""Preclassify BIRD SQLite Gold SQL with explicit per-task exclusions.

Never silently remove hard BIRD questions or call a subset the full official
Mini-Dev benchmark. Gold validation is CPU-only and separate from policy calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl_sql.dataset import read_training_parquet, unpack_task, write_parquet
from agentic_rl_sql.execution import execute_sql


def filter_gold(
    source: str | Path, target: str | Path, *, timeout: float = 30.0,
    max_rows: int = 100000,
) -> dict:
    source = Path(source).resolve()
    target = Path(target).resolve()
    if source == target:
        raise ValueError("filter output must not overwrite its source Parquet")
    tasks = [unpack_task(row) for row in read_training_parquet(source)]
    if not tasks or timeout <= 0 or max_rows < 1:
        raise ValueError("nonempty input and positive execution limits required")
    eligible, rejected = [], []
    for task in tasks:
        result = execute_sql(task.db_path, task.gold_sql,
                             timeout_seconds=timeout, max_rows=max_rows)
        if result.executed and not result.truncated:
            eligible.append(task)
        else:
            rejected.append({
                "task_id": task.task_id, "db_id": task.db_id,
                "reason": result.error_type or ("truncated" if result.truncated else "unknown"),
                "message": (result.error_message or "")[:250],
            })
    if not eligible:
        raise ValueError("all BIRD Gold SQL queries were unscorable")
    n = write_parquet(eligible, target)
    source_meta = json.loads(source.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    new_meta = {
        **source_meta,
        "output": str(target), "samples": n,
        "gold_eligible": True, "gold_timeout_seconds": timeout,
        "gold_max_rows": max_rows,
        "gold_excluded_count": len(rejected),
    }
    target.with_suffix(".manifest.json").write_text(
        json.dumps(new_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    audit = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(target), "original_samples": len(tasks), "eligible_samples": n,
        "excluded_gold": rejected, "excluded_count": len(rejected),
        "coverage_ratio": n / len(tasks),
        "gold_timeout_seconds": timeout, "gold_max_rows": max_rows,
        "source_is_filtered_subset": bool(rejected or source_meta.get("skipped_non_select", 0)),
        "comparable_to_full_official_mini_dev": not rejected and not source_meta.get("skipped_non_select", 0),
        "note": "Gold exclusions must be disclosed; evaluated accuracy is on the retained subset.",
    }
    target.with_suffix(".gold_eligibility.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sql-timeout", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=100000)
    args = parser.parse_args()
    report = filter_gold(args.input, args.output,
                         timeout=args.sql_timeout, max_rows=args.max_rows)
    print(json.dumps({"eligible_samples": report["eligible_samples"],
                      "excluded_count": report["excluded_count"],
                      "coverage_ratio": report["coverage_ratio"]}, indent=2))


if __name__ == "__main__":
    main()
