#!/usr/bin/env python3
"""Strict BIRD Mini-Dev SQLite 500-instance official EX adapter.

This script exports already-frozen blind rollout predictions to the input
format expected by bird-bench/mini_dev/evaluation/evaluation_ex.py. The Gold
labels are never passed to a policy. An optional --run-official invokes the
upstream scorer on the canonical Mini-Dev files. No model/GPU is required here,
but real policy predictions are required for any performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_bird import iter_json_rows

DELIMITER = "\t----- bird -----\t"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_query(value: str) -> str:
    query = " ".join(value.split())
    if not query or DELIMITER.strip() in query:
        raise ValueError("empty or invalid SQL in official export")
    return query


def verify_immutable_source(
    manifest_path: Path, *, require_gold: bool,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("source") != "BIRD-SQL Mini-Dev official SQLite subset":
        raise ValueError("not an official BIRD Mini-Dev source manifest")
    if require_gold and metadata.get("validated_gold_sql") is not True:
        raise ValueError("official EX adapter requires prevalidated Gold SQL")
    records_path = Path(metadata["records_path"]).resolve()
    if not records_path.is_file() or sha256(records_path) != metadata["records_sha256"]:
        raise ValueError("BIRD official records missing or have changed since preparation")
    records = [dict(record) for record in iter_json_rows(records_path)]
    if not records:
        raise ValueError("empty official BIRD Mini-Dev records")
    if len(records) != metadata.get("total_supported_sql"):
        raise ValueError("prepared row count differs from original BIRD data")
    if metadata.get("skipped_missing_database") or metadata.get("skipped_non_select"):
        raise ValueError("official full Mini-Dev evaluation requires no skipped samples")
    return metadata, records_path, records


def export_predictions(
    trajectories_file: Path, minidev_manifest: Path, output_dir: Path,
    *, require_full_500: bool = True,
) -> dict[str, Any]:
    metadata, source_path, records = verify_immutable_source(
        minidev_manifest, require_gold=True,
    )
    if require_full_500 and len(records) != 500:
        raise ValueError(f"expected the official 500-row Mini-Dev set; received {len(records)}")
    trajectories_file = trajectories_file.resolve()
    if not trajectories_file.is_file():
        raise FileNotFoundError(trajectories_file)
    rows = {}
    for line in trajectories_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("runner_error"):
            raise ValueError("rollout has a runner error; official export would be incomplete")
        if item.get("evaluation", {}).get("protocol") != "blind-final-v1":
            raise ValueError("official export requires posthoc blind-final-v1 evaluation")
        task = item.get("task", {})
        if "gold_sql" in task:
            raise ValueError("private Gold SQL leaked into policy trajectory")
        key = str(task.get("task_id", ""))
        if key in rows:
            raise ValueError(f"duplicate BIRD rollout task ID {key}")
        rows[key] = item

    if len(rows) != len(records):
        raise ValueError(f"expected {len(records)} blind policy predictions, got {len(rows)}")
    prediction = {}
    for i, original in enumerate(records):
        task_id = f"bird-eval-{i:05d}"
        item = rows.pop(task_id, None)
        if item is None:
            raise ValueError(f"missing official BIRD prediction {task_id}")
        db = str(original["db_id"])
        if item["task"]["db_id"] != db:
            raise ValueError(f"DB ID mismatch at index {i}: {item['task']['db_id']} != {db}")
        if item["task"]["question"] != str(original["question"]):
            raise ValueError(f"question mismatch at {task_id}")
        sql = str(item.get("final_sql") or "").strip()
        if not sql:
            raise ValueError(f"missing final SQL: {task_id}")
        prediction[str(i)] = clean_query(sql) + DELIMITER + db
    if rows:
        raise ValueError(f"unrecognized prediction task IDs: {sorted(rows)[:3]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "predicted_bird_minidev_sqlite.json"
    target.write_text(json.dumps(prediction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "source_dataset": "bird-bench Mini-Dev SQLite",
        "scope": "500 SELECT-only Mini-Dev (not BIRD full Dev)",
        "official_full_500_rows": len(records) == 500,
        "samples": len(records),
        "records_sha256": sha256(source_path),
        "rollouts_sha256": sha256(trajectories_file),
        "predictions_sha256": sha256(target),
        "prediction_path": str(target.resolve()),
        "gold_assistance_to_policy": False,
        "official_ex_scored": False,
        "upstream_scorer": "bird-bench/mini_dev/evaluation/evaluation_ex.py",
    }
    (output_dir / "official_bird_export_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return report


def run_official_ex(
    export: dict[str, Any], *, official_code: Path, db_root: Path,
    gold_file: Path, difficulty_file: Path, timeout: float, cpus: int,
    output_dir: Path,
) -> dict[str, Any]:
    if timeout <= 0 or cpus < 1:
        raise ValueError("invalid official scorer execution budget")
    evaluator = official_code.resolve() / "evaluation" / "evaluation_ex.py"
    for path in (evaluator, db_root, gold_file, difficulty_file):
        if not path.exists():
            raise FileNotFoundError(path)
    output_log = (output_dir / "official_ex_scores.txt").resolve()
    cmd = [
        sys.executable, str(evaluator),
        "--predicted_sql_path", str(export["prediction_path"]),
        "--ground_truth_path", str(gold_file.resolve()),
        "--db_root_path", str(db_root.resolve()) + "/",
        "--diff_json_path", str(difficulty_file.resolve()),
        "--num_cpus", str(cpus), "--meta_time_out", str(timeout),
        "--sql_dialect", "SQLite", "--output_log_path", str(output_log),
    ]
    result = subprocess.run(
        cmd, cwd=evaluator.parent, text=True, capture_output=True, check=False,
        timeout=max(120.0, timeout * export["samples"] / cpus * 2),
    )
    if result.returncode:
        raise RuntimeError(f"official BIRD EX scorer failed: {result.stderr[-1600:]}")
    if not output_log.is_file() or "EX" not in output_log.read_text(encoding="utf-8"):
        raise RuntimeError("BIRD scorer returned success but produced no valid EX report")
    report = {
        **export, "official_ex_scored": True,
        "upstream_evaluator": str(evaluator),
        "official_ex_report_path": str(output_log),
        "official_ex_report_sha256": sha256(output_log),
        "stdout_tail": result.stdout[-3000:],
    }
    (output_dir / "official_bird_ex_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--minidev-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-official", action="store_true")
    ap.add_argument("--official-code", default="")
    ap.add_argument("--db-root", default="")
    ap.add_argument("--gold-file", default="")
    ap.add_argument("--difficulty-file", default="")
    ap.add_argument("--num-cpus", type=int, default=4)
    ap.add_argument("--sql-timeout", type=float, default=30.0)
    args = ap.parse_args()
    output = Path(args.output_dir)
    report = export_predictions(
        Path(args.trajectories), Path(args.minidev_manifest), output,
    )
    if args.run_official:
        if not all([args.official_code, args.db_root, args.gold_file, args.difficulty_file]):
            ap.error("--run-official requires --official-code, --db-root, --gold-file and --difficulty-file")
        report = run_official_ex(
            report, official_code=Path(args.official_code),
            db_root=Path(args.db_root), gold_file=Path(args.gold_file),
            difficulty_file=Path(args.difficulty_file), timeout=args.sql_timeout,
            cpus=args.num_cpus, output_dir=output,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
