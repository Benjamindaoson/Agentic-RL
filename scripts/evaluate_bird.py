#!/usr/bin/env python3
"""BIRD Gold-eligible held-out SQLite model evaluation (NOT official leaderboard)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_gold_eligible(path: Path, timeout: float, max_rows: int) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    audit_file = path.with_suffix(".gold_eligibility.json")
    if not audit_file.is_file():
        raise ValueError("BIRD prepared evaluation lacks Gold eligibility audit")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    if (audit.get("gold_timeout_seconds") != timeout or
            audit.get("gold_max_rows") != max_rows or
            audit.get("output") != str(path)):
        raise ValueError("BIRD evaluation budgets/path differ from eligible Gold manifest")
    if audit.get("eligible_samples", 0) < 1 or not 0 < audit.get("coverage_ratio", 0) <= 1:
        raise ValueError("invalid BIRD eligibility coverage")
    return audit


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", help="Official held-out JSON/JSONL with --db-root")
    source.add_argument("--prepared-parquet", help="Previously Gold-verified BIRD Mini-Dev subset")
    parser.add_argument("--db-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-manifest", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="sql-policy")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--sql-timeout", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.records and not args.db_root:
        parser.error("--records requires --db-root")
    if args.sql_timeout <= 0 or args.max_rows < 1:
        parser.error("SQL execution budget must be positive")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    eligible = Path(args.prepared_parquet).resolve() if args.prepared_parquet else output / "bird_heldout_eligible.parquet"
    if args.records:
        raw = output / "bird_heldout_original.parquet"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/prepare_bird.py"),
            "--records", args.records, "--db-root", args.db_root,
            "--output", str(raw), "--split", "eval",
            "--turns", str(args.max_turns),
            "--context-limit", str(args.context_limit), "--strict-db",
        ], check=True)
        subprocess.run([
            sys.executable, str(ROOT / "scripts/filter_bird_gold.py"),
            "--input", str(raw), "--output", str(eligible),
            "--sql-timeout", str(args.sql_timeout),
            "--max-rows", str(args.max_rows),
        ], check=True)
    audit = verify_gold_eligible(eligible, args.sql_timeout, args.max_rows)
    (output / "bird_model_eval_protocol.json").write_text(
        json.dumps({
            "dataset": "BIRD held-out Gold-eligible SQLite subset",
            "eligible_parquet": str(eligible), "eligible_samples": audit["eligible_samples"],
            "excluded_gold": audit["excluded_count"], "coverage_ratio": audit["coverage_ratio"],
            "sql_timeout_seconds": args.sql_timeout, "max_rows": args.max_rows,
            "scorer": "Agentic-RL internal SQLite result equivalence, NOT official BIRD leaderboard",
        }, indent=2) + "\n", encoding="utf-8",
    )
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_rollouts.py"),
        "--dataset", str(eligible),
        "--output", str(output / "bird_trajectories.jsonl"),
        "--base-url", args.base_url, "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--policy-checkpoint", args.policy_checkpoint,
        "--policy-manifest", args.policy_manifest,
        "--seed", str(args.seed), "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
        "--context-limit", str(args.context_limit),
        "--max-turns", str(args.max_turns),
        "--sql-timeout", str(args.sql_timeout),
        "--max-rows", str(args.max_rows),
    ], check=True)


if __name__ == "__main__":
    main()
