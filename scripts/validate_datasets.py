#!/usr/bin/env python3
"""Offline, fail-closed Spider/BIRD dataset validation.

Checks actual Parquet content against manifests, cross-schema isolation, task ID
stability across ablations, referenced SQLite existence, and optional real Gold
SQL execution. Never counts a synthetic check as a real dataset download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rl_sql.dataset import read_training_parquet, unpack_task
from agentic_rl_sql.execution import execute_sql


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_parquet(
    path: Path, *, split: str, require_real_gold: bool,
    gold_cache: dict[tuple[str, str], tuple[bool, str]],
    timeout: float, max_rows: int,
) -> tuple[dict[str, Any], dict[str, tuple[str, str, str]]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared dataset: {path}")
    records = read_training_parquet(path)
    if not records:
        raise ValueError(f"empty prepared dataset: {path}")
    ids = set()
    dbs = set()
    signatures = {}
    gold_failures = []
    for record in records:
        task = unpack_task(record)
        if task.split != split:
            raise ValueError(f"{path.name}: {task.task_id} has split {task.split}, expected {split}")
        if not task.question.strip() or not task.gold_sql.strip() or not task.db_id.strip():
            raise ValueError(f"{path.name}: task {task.task_id} is missing question, gold SQL or db_id")
        if task.task_id in ids:
            raise ValueError(f"{path.name}: duplicate task id: {task.task_id}")
        ids.add(task.task_id)
        dbs.add(task.db_id)
        db_path = Path(task.db_path)
        if not db_path.is_file():
            raise FileNotFoundError(f"{path.name}: database for {task.task_id} is missing: {db_path}")
        signatures[task.task_id] = (task.db_id, task.question, task.gold_sql)
        if require_real_gold:
            key = (str(db_path.resolve()), task.gold_sql)
            if key not in gold_cache:
                outcome = execute_sql(db_path, task.gold_sql, timeout_seconds=timeout, max_rows=max_rows)
                gold_cache[key] = (
                    bool(outcome.executed and not outcome.truncated),
                    (
                        f"{outcome.error_type or ('truncated' if outcome.truncated else 'ok')}: "
                        f"{(outcome.error_message or '')[:250]}; gold_sql={task.gold_sql[:250]!r}"
                    ),
                )
            if not gold_cache[key][0]:
                gold_failures.append({"task_id": task.task_id, "error": gold_cache[key][1]})
    if gold_failures:
        raise ValueError(f"{path.name}: {len(gold_failures)} gold SQL queries failed, e.g. {gold_failures[:3]}")
    return {
        "file": str(path), "sha256": digest(path), "samples": len(records),
        "database_count": len(dbs),
        "database_ids": sorted(dbs),
        "task_ids_sha256": hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest(),
        "gold_verified": require_real_gold,
        "gold_coverage": len(records) if require_real_gold else 0,
    }, signatures


def validate_spider(
    manifest_path: str | Path, *, verify_gold: bool = True,
    timeout: float = 8.0, max_rows: int = 5000,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = manifest.get("variants", [])
    if not variants:
        raise ValueError("Spider manifest contains no experiment variants")
    if manifest.get("test_source", "").lower().find("dev.json") == -1:
        raise ValueError("Spider final evaluation must come from official dev.json")
    if max_rows < 1 or timeout <= 0:
        raise ValueError("invalid gold execution resource limits")
    gold_cache = {}
    result = {"dataset": "spider-1.0", "manifest": str(manifest_path), "variants": [],
              "verified_gold_queries": False, "complete": False}
    canonical = {}
    for variant in variants:
        name = variant["name"]
        ctx, turns, check = variant["context"], variant["turns"], variant["explicit_check"]
        actual = {}
        ids = {}
        for split in ("train", "val", "test"):
            path = Path(variant["files"][split])
            stats, signatures = validate_parquet(
                path, split=split, require_real_gold=verify_gold,
                gold_cache=gold_cache, timeout=timeout, max_rows=max_rows,
            )
            if stats["samples"] != variant["samples"][split]:
                raise ValueError(f"{name}/{split}: samples do not match manifest")
            if stats["database_count"] != variant["database_counts"][split]:
                raise ValueError(f"{name}/{split}: database count does not match manifest")
            expected_db_hash = variant["database_ids_sha256"][split]
            actual_db_hash = hashlib.sha256(json.dumps(stats["database_ids"]).encode("utf-8")).hexdigest()
            if expected_db_hash != actual_db_hash:
                raise ValueError(f"{name}/{split}: database IDs differ from manifest")
            actual[split], ids[split] = stats, signatures
            for raw in read_training_parquet(path):
                task = unpack_task(raw)
                if (task.context_limit, task.max_turns, bool(task.metadata.get("explicit_check", False))) != (ctx, turns, check):
                    raise ValueError(f"{name}: task {task.task_id} has incorrect experimental budget")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if set(actual[a]["database_ids"]) & set(actual[b]["database_ids"]):
                raise ValueError(f"{name}: {a}/{b} database leakage")
            if set(ids[a]) & set(ids[b]):
                raise ValueError(f"{name}: {a}/{b} task ID collision")
        for split in ("train", "val", "test"):
            if split not in canonical:
                canonical[split] = ids[split]
            elif canonical[split] != ids[split]:
                raise ValueError(f"{name}: {split} task content differs across ablation variants")
        result["variants"].append({"name": name, "splits": actual, "schema_disjoint": True})
    result["unique_gold_queries_executed"] = len(gold_cache)
    result["verified_gold_queries"] = verify_gold
    result["structurally_validated"] = True
    result["complete"] = verify_gold
    return result


def validate_bird(
    path: str | Path, *, require_all_records: bool = True,
    verify_gold: bool = True, timeout: float = 8, max_rows: int = 5000,
) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    skipped = manifest.get("skipped_missing_database", 0) + manifest.get("skipped_non_select", 0)
    if require_all_records and skipped:
        raise ValueError(f"BIRD manifest skipped {skipped} records; full coverage was requested")
    # The BIRD manifest already records the subset's split through the Parquet tasks.
    rows = read_training_parquet(path)
    if not rows:
        raise ValueError("BIRD prepared parquet is empty")
    splits = {unpack_task(row).split for row in rows}
    if len(splits) != 1:
        raise ValueError(f"mixed BIRD splits in {path}")
    stats, _ = validate_parquet(
        path, split=next(iter(splits)), require_real_gold=verify_gold,
        gold_cache={}, timeout=timeout, max_rows=max_rows,
    )
    if stats["samples"] != manifest.get("samples"):
        raise ValueError("BIRD data count does not match manifest")
    return {"dataset": "bird", "stats": stats,
            "dropped_records": skipped, "complete": skipped == 0 and verify_gold}


def main():
    ap = argparse.ArgumentParser(description="Validate prepared SQL datasets with real CPU SQLite execution")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--spider-manifest")
    group.add_argument("--bird-parquet")
    ap.add_argument("--skip-gold-execution", action="store_true",
                    help="Structural check only; result cannot be marked gold-verified")
    ap.add_argument("--allow-bird-subset", action="store_true")
    ap.add_argument("--sql-timeout", type=float, default=8.0)
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    try:
        if args.spider_manifest:
            data = validate_spider(args.spider_manifest,
                                   verify_gold=not args.skip_gold_execution,
                                   timeout=args.sql_timeout, max_rows=args.max_rows)
        else:
            data = validate_bird(args.bird_parquet,
                                 require_all_records=not args.allow_bird_subset,
                                 verify_gold=not args.skip_gold_execution,
                                 timeout=args.sql_timeout, max_rows=args.max_rows)
        data["passed"] = True
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        data = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": data["passed"], "error": data.get("error"),
                      "unique_gold_queries_executed": data.get("unique_gold_queries_executed")},
                     ensure_ascii=False))
    if not data["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
