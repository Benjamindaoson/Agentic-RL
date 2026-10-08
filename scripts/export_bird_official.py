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
    if require_gold and (
        metadata.get("full_mini_dev_coverage") is not True
        or metadata.get("gold_excluded_count") != 0
        or metadata.get("eligible_samples") != metadata.get("total_supported_sql")
    ):
        raise ValueError(
            "official full BIRD EX requires all 500 Gold tasks to be executable; "
            "run the project's clearly labeled Gold-eligible BIRD subset evaluation instead"
        )
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
        evaluation = item.get("evaluation", {})
        protocol = evaluation.get("protocol")
        if protocol not in ("blind-ungraded-v1", "blind-final-v1"):
            raise ValueError("official export requires audited blind policy predictions")
        if require_full_500:
            if protocol != "blind-ungraded-v1" or evaluation.get("gold_access") != "none":
                raise ValueError("official full-500 export must use ungraded policy-only inference")
            if "success" in item or "reward" in item:
                raise ValueError("official full-500 input cannot be prefiltered by a local Gold scorer")
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
        "full_official_predictions_are_ungraded": require_full_500,
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
    # Upstream Mini-Dev package_sqls uses zip() and can silently drop unmatched
    # predicted/Gold tasks. Reject missing, re-ordered or mislabeled files.
    predictions = json.loads(Path(export["prediction_path"]).read_text(encoding="utf-8"))
    gold_lines = [
        line.strip() for line in gold_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    difficulty = [
        json.loads(line) for line in difficulty_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = int(export["samples"])
    if (len(predictions), len(gold_lines), len(difficulty)) != (n, n, n):
        raise ValueError(
            f"official BIRD task lengths differ: pred={len(predictions)}, "
            f"gold={len(gold_lines)}, difficulty={len(difficulty)}, expected={n}"
        )
    if list(predictions) != [str(i) for i in range(n)]:
        raise ValueError("official prediction indices are incomplete or not in original order")
    for i, gold_line in enumerate(gold_lines):
        parts = gold_line.rsplit("\t", 1)
        pred_parts = predictions[str(i)].split(DELIMITER)
        if len(parts) != 2 or len(pred_parts) != 2 or parts[1] != pred_parts[1]:
            raise ValueError(f"official BIRD Gold/predicted database mismatch at index {i}")
        if difficulty[i].get("difficulty") not in {"simple", "moderate", "challenging"}:
            raise ValueError(f"unknown BIRD official difficulty at index {i}")
    if len({row["difficulty"] for row in difficulty}) < 3:
        raise ValueError("official difficulty buckets are incomplete")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_ex_stdout.log").write_text(result.stdout, encoding="utf-8")
    (output_dir / "official_ex_stderr.log").write_text(result.stderr, encoding="utf-8")
    if not output_log.is_file() or "EX" not in output_log.read_text(encoding="utf-8"):
        raise RuntimeError("BIRD scorer returned success but produced no valid EX report")
    score_rows = [
        line for line in output_log.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("EX")
    ]
    scores = None
    for line in score_rows:
        fields = line.split()
        if len(fields) >= 5:
            try:
                scores = dict(zip(
                    ("simple", "moderate", "challenging", "total"),
                    [float(value) for value in fields[1:5]],
                ))
            except ValueError:
                continue
    if scores is None or any(value < 0 or value > 100 for value in scores.values()):
        raise RuntimeError("official scorer result could not be parsed into verified EX percentages")
    report = {
        **export, "official_ex_scored": True,
        "upstream_evaluator": str(evaluator),
        "official_ex_report_path": str(output_log),
        "official_ex_report_sha256": sha256(output_log),
        "official_ex_accuracy_percent": scores,
        "official_stdout_sha256": sha256(output_dir / "official_ex_stdout.log"),
        "official_stderr_sha256": sha256(output_dir / "official_ex_stderr.log"),
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
