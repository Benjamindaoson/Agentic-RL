#!/usr/bin/env python3
"""Strict evidence gate. Missing training/rollout evidence is failure, never a fake pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(training_dir: Path, comparison_dir: Path) -> dict:
    problems = []
    required_training = [
        "grpo_run_manifest.json", "training_config.json", "training.log",
        "training_metrics.jsonl", "policy_checkpoint_manifest.json",
        "gpu_telemetry.jsonl", "resource_metrics.json", "leakage_audit.json",
        "base_model_identity.json",
    ]
    required_comparison = [
        "comparison.json", "comparison.csv", "comparison.md",
        "evaluation_protocol.json", "leakage_audit.json",
        "weight_change_audit.json",
    ]
    for name in required_training:
        if not (training_dir / name).is_file() or (training_dir / name).stat().st_size == 0:
            problems.append(f"training artifact missing/empty: {name}")
    for name in required_comparison:
        if not (comparison_dir / name).is_file() or (comparison_dir / name).stat().st_size == 0:
            problems.append(f"comparison artifact missing/empty: {name}")
    if problems:
        return {"passed": False, "checks": [], "problems": problems}
    train = json.loads((training_dir / "grpo_run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((training_dir / "policy_checkpoint_manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    audit = json.loads((comparison_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    identity = json.loads((training_dir / "base_model_identity.json").read_text(encoding="utf-8"))
    weight_change = json.loads((comparison_dir / "weight_change_audit.json").read_text(encoding="utf-8"))
    eval_protocols = json.loads((comparison_dir / "evaluation_protocol.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line) for line in (training_dir / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resource = json.loads((training_dir / "resource_metrics.json").read_text(encoding="utf-8"))
    gpu = [
        json.loads(line) for line in (training_dir / "gpu_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checks = {
        "trainer_returned": train.get("status") == "trainer_returned_verify_metrics_and_checkpoints",
        "correct_protocol": train.get("eval_protocol") == comparison.get("protocol") == "blind-final-v1",
        "sql_result_cap_is_identical": train.get("sql_max_rows") ==
            comparison.get("budget", {}).get("sql_max_rows"),
        "measured_optimization_steps": any(row.get("global_step", -1) >= 1 for row in metrics),
        "checkpoint_weights_present": checkpoint.get("weight_file_count", 0) > 0,
        "gpu_telemetry_present": any(row.get("gpus") for row in gpu),
        "measured_resource_summary": resource.get("telemetry_sample_count", 0) > 0
            and resource.get("gpu_memory_peak_mib", 0) > 0,
        "leakage_audit_passed": audit.get("passed") is True,
        "base_and_grpo_evaluated": all(name in comparison.get("runs", {}) for name in ("base", "grpo")),
        "paired_statistics_present": "grpo" in comparison.get("paired_vs_base", {}),
        "base_weights_identified": identity.get("status") == "resolved_hashed"
            and identity.get("weight_file_count", 0) > 0,
        "policy_manifests_attached_to_all_evaluations": all(
            item.get("policy_identity_sha256") for item in eval_protocols.values()
        ),
        "grpo_weights_actually_changed": weight_change.get("checks", {}).get("grpo_policy_parameters_changed") is True,
        "no_update_weights_unchanged": weight_change.get("checks", {}).get("no_update_policy_parameters_unchanged") is True,
        "weight_change_audit_passed": weight_change.get("passed") is True,
        "no_update_control": "no_update" in comparison.get("runs", {}),
        "reward_ablation_control": "reward_ablation" in comparison.get("runs", {}),
    }
    for name, ok in checks.items():
        if not ok:
            problems.append(f"failed evidence check: {name}")
    return {"passed": not problems, "checks": checks, "problems": problems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", required=True)
    ap.add_argument("--comparison-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = verify(Path(args.training_dir), Path(args.comparison_dir))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
