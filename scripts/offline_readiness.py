#!/usr/bin/env python3
"""CPU-only pre-GPU readiness audit. Never claims a GPU training result."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "src/agentic_rl_sql/agent.py",
    "src/agentic_rl_sql/evaluator.py",
    "scripts/prepare_spider.py",
    "scripts/prepare_bird.py",
    "scripts/train_sql_agent.py",
    "scripts/run_local_training.sh",
    "scripts/run_rollouts.py",
    "scripts/compare_experiments.py",
    "scripts/verify_evidence.py",
    "scripts/audit_leakage.py",
]


def run_check(command: list[str]) -> dict:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-2000:],
    }


def audit(
    *, run_tests: bool = True, require_datasets: bool = False,
    eligible_manifest: str | Path | None = None,
    bird_parquet: str | Path | None = None,
    require_framework_contract: bool = False,
) -> dict:
    checks = {
        "required_source_files": all((ROOT / p).is_file() for p in REQUIRED),
        "python_version": sys.version_info >= (3, 12),
        "dataset_parser_available": importlib.util.find_spec("pandas") is not None,
    }
    commands = []
    if run_tests:
        commands = [
            run_check([sys.executable, "-m", "pytest", "-q"]),
            run_check([sys.executable, "scripts/audit_leakage.py", "--output", "artifacts/leakage_audit.json"]),
            run_check([sys.executable, "scripts/run_experiment_matrix.py", "--stage", "smoke", "--output-dir", "/tmp/agentic-rl-plan"]),
            run_check([sys.executable, "-m", "compileall", "-q", "src", "agent", "scripts"]),
            run_check(["bash", "-n", "scripts/run_local_training.sh", "scripts/start_policy_server.sh"]),
        ]
        checks["offline_commands_passed"] = all(x["passed"] for x in commands)
    if require_datasets:
        from scripts.validate_datasets import validate_spider, validate_bird

        manifest = Path(eligible_manifest).resolve() if eligible_manifest else (
            ROOT / "data/spider_eligible/manifest.json"
        )
        checks["spider_eligible_manifest_present"] = manifest.is_file()
        eligibility = manifest.parent / "gold_eligibility_audit.json"
        checks["spider_gold_eligibility_audit_present"] = eligibility.is_file()
        if manifest.is_file() and eligibility.is_file():
            data = json.loads(eligibility.read_text(encoding="utf-8"))
            checks["spider_exclusions_disclosed"] = (
                data.get("verified_gold_eligible") is True
                and sum(data.get("source_samples", {}).values()) > 0
                and sum(data.get("eligible_samples", {}).values()) > 0
                and all(
                    0 < data.get("coverage_ratio", {}).get(k, 0) <= 1
                    for k in ("train", "val", "test")
                )
                and len(data.get("exclusions", [])) == sum(data.get("excluded_by_split", {}).values())
            )
            if checks["spider_exclusions_disclosed"]:
                try:
                    gold_validation = validate_spider(
                        manifest, verify_gold=True,
                        timeout=data["timeout_seconds"],
                        max_rows=data["max_result_rows"],
                    )
                    checks["spider_eligible_gold_executable"] = gold_validation["complete"]
                except (ValueError, FileNotFoundError, KeyError, OSError):
                    checks["spider_eligible_gold_executable"] = False
        checks["bird_prepared_parquet_specified"] = bird_parquet is not None
        if bird_parquet is not None:
            try:
                report = validate_bird(
                    bird_parquet, require_all_records=False,
                    verify_gold=True, timeout=8, max_rows=5000,
                )
                checks["bird_minidev_gold_executable"] = (
                    report["stats"]["gold_verified"] is True
                    and report["stats"]["samples"] > 0
                )
            except (ValueError, FileNotFoundError, KeyError, OSError):
                checks["bird_minidev_gold_executable"] = False
    if require_framework_contract:
        try:
            from scripts.verify_framework_contract import contract
            checks["agentlightning_verl_cpu_contract"] = contract()["passed"]
        except Exception:
            checks["agentlightning_verl_cpu_contract"] = False
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "commands": commands,
        "scope": "CPU-only code checks; not a substitute for real dataset or GPU framework integration",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/offline_readiness.json")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--require-datasets", action="store_true")
    parser.add_argument("--eligible-manifest", help="Explicit Gold-eligible Spider manifest")
    parser.add_argument("--bird-parquet", help="Prepared real BIRD Mini-Dev evaluation Parquet")
    parser.add_argument("--require-framework-contract", action="store_true")
    args = parser.parse_args()
    result = audit(
        run_tests=not args.skip_tests, require_datasets=args.require_datasets,
        eligible_manifest=args.eligible_manifest, bird_parquet=args.bird_parquet,
        require_framework_contract=args.require_framework_contract,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": result["checks"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
