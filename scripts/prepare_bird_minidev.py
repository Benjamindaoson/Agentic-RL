#!/usr/bin/env python3
"""Prepare real official BIRD Mini-Dev SQLite records, with explicit coverage.

Does NOT train or serve a model. A Mini-Dev SQLite SELECT-only subset is not
interchangeable with the BIRD full Dev/official leaderboard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_inputs(source: Path) -> tuple[Path, Path]:
    source = source.resolve()
    record_files = sorted(source.rglob("mini_dev_sqlite.json"))
    if len(record_files) != 1:
        raise ValueError(f"expected exactly one mini_dev_sqlite.json, found {len(record_files)}")
    record_file = record_files[0]
    db_candidates = sorted(p for p in source.rglob("dev_databases") if p.is_dir())
    if len(db_candidates) != 1:
        raise ValueError(f"expected exactly one dev_databases directory, found {len(db_candidates)}")
    return record_file, db_candidates[0]


def prepare(
    source_dir: Path, output_dir: Path, *, max_turns: int = 1,
    context_limit: int = 4096, full_gold_audit: bool = False,
    max_rows: int = 100000, gold_timeout: float = 30.0,
) -> dict:
    records, db_root = find_inputs(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "bird_mini_dev_select.parquet"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/prepare_bird.py"),
        "--records", str(records), "--db-root", str(db_root),
        "--output", str(path), "--split", "eval",
        "--turns", str(max_turns), "--context-limit", str(context_limit),
        "--strict-db",
    ], check=True)
    prepare_meta = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if not prepare_meta["samples"]:
        raise ValueError("Mini-Dev SQLite preparation had zero SELECT queries")
    effective = path
    gold_audit = None
    if full_gold_audit:
        effective = output_dir / "bird_mini_dev_eligible.parquet"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/filter_bird_gold.py"),
            "--input", str(path), "--output", str(effective),
            "--sql-timeout", str(gold_timeout), "--max-rows", str(max_rows),
        ], check=True)
        gold_audit = json.loads(effective.with_suffix(".gold_eligibility.json").read_text(encoding="utf-8"))
    args = [
        sys.executable, str(ROOT / "scripts/validate_datasets.py"),
        "--bird-parquet", str(effective), "--allow-bird-subset",
        "--max-rows", str(max_rows),
        "--output", str(output_dir / "bird_mini_dev_validation.json"),
    ]
    if not full_gold_audit:
        args.append("--skip-gold-execution")
    subprocess.run(args, check=True)
    validation = json.loads((output_dir / "bird_mini_dev_validation.json").read_text(encoding="utf-8"))
    result = {
        "source": "BIRD-SQL Mini-Dev official SQLite subset",
        "records_path": str(records),
        "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "prepared_file": str(path), "total_supported_sql": prepare_meta["samples"],
        "skipped_non_select": prepare_meta["skipped_non_select"],
        "skipped_missing_database": prepare_meta["skipped_missing_database"],
        "database_root": str(db_root),
        "validated_gold_sql": bool(full_gold_audit and validation.get("complete")),
        "eligible_parquet": str(effective),
        "eligible_samples": gold_audit["eligible_samples"] if gold_audit else prepare_meta["samples"],
        "gold_excluded_count": gold_audit["excluded_count"] if gold_audit else None,
        "gold_coverage_ratio": gold_audit["coverage_ratio"] if gold_audit else None,
        "full_mini_dev_coverage": gold_audit["comparable_to_full_official_mini_dev"] if gold_audit else None,
        "source_is_external_to_spider": True,
        "claimed_official_leaderboard_metric": False,
    }
    (output_dir / "bird_mini_dev_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True,
                        help="Safely extracted official minidev.zip root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--full-gold-audit", action="store_true")
    parser.add_argument("--gold-timeout", type=float, default=30.0)
    args = parser.parse_args()
    report = prepare(
        Path(args.source_dir), Path(args.output_dir),
        max_turns=args.max_turns, context_limit=args.context_limit,
        max_rows=args.max_rows, full_gold_audit=args.full_gold_audit,
        gold_timeout=args.gold_timeout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
