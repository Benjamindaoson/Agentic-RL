#!/usr/bin/env python3
"""Build an explicit Gold-eligible Spider subset without hiding exclusions.

Some official Spider 1.0 annotations are not executable in the actual SQLite
release; many other queries exceed a 5,000-row arbitrary cap. This CPU pass
classifies all unique Gold SQL tasks ONCE, filters consistently across EVERY
ablation, and records exact exclusions and final-test coverage. It must never
silently relabel a filtered result as official full-Dev accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl_sql.dataset import read_training_parquet, unpack_task, write_parquet
from agentic_rl_sql.execution import execute_sql
from scripts.validate_datasets import validate_spider


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepared_tasks(path: str | Path):
    return [unpack_task(item) for item in read_training_parquet(path)]


def filter_manifest(
    manifest_path: str | Path, output_dir: str | Path, *,
    max_rows: int = 5000, timeout: float = 8.0,
) -> dict:
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == manifest_path.parent:
        raise ValueError("output directory must differ from source directory")
    if max_rows < 1 or timeout <= 0:
        raise ValueError("invalid Gold SQL execution budgets")
    # Exhaustively check all task/split/ablation IDs before writing any results.
    validate_spider(manifest_path, verify_gold=False)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = source["variants"]
    first = variants[0]
    decisions = {}
    exclusion_details = []
    total = {}
    for split in ("train", "val", "test"):
        tasks = prepared_tasks(first["files"][split])
        total[split] = len(tasks)
        for task in tasks:
            key = (split, task.task_id)
            if key in decisions:
                raise ValueError("duplicate Gold task in initial variant")
            execution = execute_sql(task.db_path, task.gold_sql,
                                    timeout_seconds=timeout, max_rows=max_rows)
            ok = bool(execution.executed and not execution.truncated)
            decisions[key] = ok
            if not ok:
                exclusion_details.append({
                    "split": split, "task_id": task.task_id, "db_id": task.db_id,
                    "error_type": execution.error_type or (
                        "truncated" if execution.truncated else "unknown"
                    ),
                    "error_message": (execution.error_message or "")[:300],
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    changed = deepcopy(source)
    new_variants = []
    for item in variants:
        copied = deepcopy(item)
        copied["excluded_gold_tasks"] = {}
        for split in ("train", "val", "test"):
            tasks = prepared_tasks(item["files"][split])
            kept = [task for task in tasks if decisions[(split, task.task_id)]]
            if not kept:
                raise ValueError(f"Gold filtering left {item['name']}/{split} empty")
            if len(kept) > len(tasks):
                raise AssertionError("Gold filtering unexpectedly increased sample count")
            target = output_dir / Path(item["files"][split]).name
            n = write_parquet(kept, target)
            schemas = sorted({task.db_id for task in kept})
            copied["files"][split] = str(target)
            copied["samples"][split] = n
            copied["database_counts"][split] = len(schemas)
            copied["database_ids_sha256"][split] = hashlib.sha256(
                json.dumps(schemas).encode("utf-8")
            ).hexdigest()
            copied["excluded_gold_tasks"][split] = len(tasks) - n
        new_variants.append(copied)
    changed["variants"] = new_variants
    changed["source_manifest_sha256"] = sha256(manifest_path)
    changed["gold_eligible_subset"] = True
    changed["gold_filter_max_rows"] = max_rows
    changed["gold_filter_timeout_seconds"] = timeout
    changed["test_source"] = "official Spider 1.0 dev.json (explicit Gold-eligible subset)"
    changed["test_is_full_official_dev"] = not any(item["split"] == "test" for item in exclusion_details)
    (output_dir / "manifest.json").write_text(
        json.dumps(changed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    excluded_by_split = {
        split: sum(row["split"] == split for row in exclusion_details)
        for split in ("train", "val", "test")
    }
    audit = {
        "dataset": "Spider 1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(manifest_path), "input_sha256": sha256(manifest_path),
        "filtered_manifest": str(output_dir / "manifest.json"),
        "max_result_rows": max_rows, "timeout_seconds": timeout,
        "unique_gold_tasks_checked": sum(total.values()),
        "source_samples": total, "excluded_by_split": excluded_by_split,
        "eligible_samples": {k: total[k] - excluded_by_split[k] for k in total},
        "coverage_ratio": {
            split: (total[split] - excluded_by_split[split]) / total[split]
            for split in total
        },
        "exclusions": exclusion_details,
        "verified_gold_eligible": True,
        "full_official_dev_coverage": excluded_by_split["test"] == 0,
        "method_note": "All excluded sample IDs are disclosed; filtered subsets must not be reported as full official Spider scores.",
    }
    (output_dir / "gold_eligibility_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--sql-timeout", type=float, default=8.0)
    args = parser.parse_args()
    result = filter_manifest(
        args.input_manifest, args.output_dir, max_rows=args.max_rows,
        timeout=args.sql_timeout,
    )
    print(json.dumps({
        "unique_gold_tasks_checked": result["unique_gold_tasks_checked"],
        "excluded_by_split": result["excluded_by_split"],
        "eligible_samples": result["eligible_samples"],
        "full_official_dev_coverage": result["full_official_dev_coverage"],
        "output": result["filtered_manifest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
