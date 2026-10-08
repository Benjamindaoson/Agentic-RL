#!/usr/bin/env python3
"""Prepare independent BIRD held-out SQLite evaluation and call blind evaluator."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Official held-out BIRD dev JSON/JSONL")
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--policy-manifest", required=True)
    ap.add_argument("--policy-checkpoint", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="sql-policy")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-limit", type=int, default=4096)
    ap.add_argument("--max-turns", type=int, default=1)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    parquet = output / "bird_heldout.parquet"
    subprocess.run([
        sys.executable, str(root / "scripts" / "prepare_bird.py"),
        "--records", args.records, "--db-root", args.db_root,
        "--output", str(parquet), "--split", "eval", "--strict-db",
        "--context-limit", str(args.context_limit), "--turns", str(args.max_turns),
    ], check=True)
    manifest = json.loads(parquet.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["samples"] == 0:
        raise ValueError("no held-out BIRD samples were prepared")
    if manifest["skipped_missing_database"] or manifest["skipped_non_select"]:
        raise ValueError("held-out BIRD data incomplete; audit skipped tasks and report subset coverage")
    subprocess.run([
        sys.executable, str(root / "scripts" / "run_rollouts.py"),
        "--dataset", str(parquet),
        "--output", str(output / "bird_trajectories.jsonl"),
        "--model", args.model, "--tokenizer", args.tokenizer,
        "--base-url", args.base_url,
        "--policy-checkpoint", args.policy_checkpoint,
        "--policy-manifest", args.policy_manifest,
        "--seed", str(args.seed), "--context-limit", str(args.context_limit),
        "--max-turns", str(args.max_turns),
    ], check=True)


if __name__ == "__main__":
    main()
