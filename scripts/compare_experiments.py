#!/usr/bin/env python3
"""Paired, provenance-checked blind evaluation with bootstrap CI and McNemar."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path) -> tuple[dict, dict, dict[str, dict]]:
    path = path.resolve()
    stem = path.stem
    metrics = json.loads(path.with_name(stem + "_metrics.json").read_text(encoding="utf-8"))
    protocol = json.loads(path.with_name(stem + "_protocol.json").read_text(encoding="utf-8"))
    if metrics.get("trajectories_sha256") != sha256_file(path):
        raise ValueError(f"raw trajectories modified after metrics were produced: {path}")
    if metrics.get("protocol_fingerprint") != protocol.get("fingerprint"):
        raise ValueError(f"protocol/metrics mismatch for {path}")
    if protocol.get("task_count") != metrics.get("samples"):
        raise ValueError(f"task count does not match protocol for {path}")
    if protocol.get("protocol") != "blind-final-v1" or protocol.get("oracle_access") != "post_rollout_only":
        raise ValueError(f"not a blind protocol: {path}")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        task = data["task"]
        if "gold_sql" in task or "runner_error" in data:
            raise ValueError("raw rollout contains a private gold answer or an execution failure")
        if data.get("evaluation", {}).get("protocol") != "blind-final-v1":
            raise ValueError("missing posthoc blind evaluation protocol")
        task_id = task["task_id"]
        if task_id in rows:
            raise ValueError(f"duplicate task id {task_id} in {path}")
        rows[task_id] = data
    if len(rows) != metrics.get("samples") or not rows:
        raise ValueError(f"sample count mismatch: {path}")
    expected_id_hash = hashlib.sha256(
        json.dumps(sorted(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if expected_id_hash != protocol.get("task_ids_sha256"):
        raise ValueError(f"task IDs differ from signed protocol: {path}")
    return metrics, protocol, rows


def validate_paired(left: tuple, right: tuple) -> None:
    lm, lp, lr = left
    rm, rp, rr = right
    if set(lr) != set(rr):
        raise ValueError("unpaired task sets")
    for field in ("dataset_sha256", "task_ids_sha256", "seed", "temperature", "tokenizer", "budget", "gold_eligible_coverage"):
        if lp.get(field) != rp.get(field):
            raise ValueError(f"mismatched paired evaluation field {field}")
    if lm.get("runner_error_count") or rm.get("runner_error_count"):
        raise ValueError("evaluation contains runner errors")
    for task_id in lr:
        a, b = lr[task_id]["task"], rr[task_id]["task"]
        for field in ("question", "db_id", "evidence", "max_turns", "context_limit"):
            if a.get(field) != b.get(field):
                raise ValueError(f"task public input differs: {task_id}, {field}")


def paired_stats(base: dict, candidate: dict, *, seed: int = 42, n_boot: int = 2000) -> dict:
    ids = sorted(base)
    difference = [int(bool(candidate[k]["success"])) - int(bool(base[k]["success"])) for k in ids]
    improved = sum(d > 0 for d in difference)
    regressed = sum(d < 0 for d in difference)
    n = len(difference)
    gain = sum(difference) / n
    rng = random.Random(seed)
    draws = sorted(
        sum(difference[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = draws[math.floor(0.025 * (n_boot - 1))]
    hi = draws[math.ceil(0.975 * (n_boot - 1))]
    discordant = improved + regressed
    p_value = (
        min(1.0, 2.0 * sum(
            math.comb(discordant, k) / (2 ** discordant)
            for k in range(min(improved, regressed) + 1)
        )) if discordant else 1.0
    )
    return {
        "samples": n, "gain": gain, "gain_pp": 100 * gain,
        "bootstrap_95_ci_pp": [100 * lo, 100 * hi],
        "paired_bootstrap_replicates": n_boot, "bootstrap_seed": seed,
        "improved_tasks": improved, "regressed_tasks": regressed,
        "unchanged_tasks": n - discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def build_comparison(runs: dict[str, tuple], *, seed: int = 42, n_boot: int = 2000) -> dict:
    if "base" not in runs or "grpo" not in runs:
        raise ValueError("base and grpo runs are required")
    baseline = runs["base"]
    for name, run in runs.items():
        if name != "base":
            validate_paired(baseline, run)
    return {
        "protocol": "blind-final-v1", "label": "actual paired rollout outputs",
        "dataset_sha256": baseline[1]["dataset_sha256"],
        "task_ids_sha256": baseline[1]["task_ids_sha256"],
        "budget": baseline[1]["budget"],
        "gold_eligible_coverage": baseline[1].get("gold_eligible_coverage"),
        "runs": {
            name: {
                "model": run[0]["model"],
                "policy_checkpoint": run[0]["policy_checkpoint"],
                "policy_identity_sha256": run[1].get("policy_identity_sha256"),
                "samples": run[0]["samples"],
                "task_accuracy": run[0]["task_accuracy"],
                "first_turn_accuracy": run[0]["first_turn_accuracy"],
                "mean_reward": run[0]["mean_reward"],
                "mean_turns": run[0]["mean_turns"],
                "invalid_sql_rate": run[0]["invalid_sql_rate"],
                "latency_p95_ms": run[0]["latency_p95_ms"],
                "trajectories_sha256": run[0]["trajectories_sha256"],
            }
            for name, run in runs.items()
        },
        "paired_vs_base": {
            name: paired_stats(baseline[2], run[2], seed=seed, n_boot=n_boot)
            for name, run in runs.items() if name != "base"
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base *_trajectories.jsonl")
    ap.add_argument("--grpo", required=True, help="GRPO *_trajectories.jsonl")
    ap.add_argument("--no-update", help="lr=0 control")
    ap.add_argument("--reward-ablation", help="validity_only control")
    ap.add_argument("--leakage-audit", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.bootstrap_replicates < 100:
        ap.error("bootstrap must have at least 100 replicates")
    audit = json.loads(Path(args.leakage_audit).read_text(encoding="utf-8"))
    if audit.get("protocol") != "blind-final-v1" or audit.get("passed") is not True:
        raise ValueError("blind oracle leakage audit must pass before comparisons")
    paths = {"base": Path(args.base), "grpo": Path(args.grpo)}
    if args.no_update:
        paths["no_update"] = Path(args.no_update)
    if args.reward_ablation:
        paths["reward_ablation"] = Path(args.reward_ablation)
    runs = {name: load_run(path) for name, path in paths.items()}
    report = build_comparison(runs, seed=args.seed, n_boot=args.bootstrap_replicates)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment", "samples", "accuracy", "first_turn_accuracy", "mean_reward", "mean_turns", "latency_p95_ms"])
        for name, row in report["runs"].items():
            writer.writerow([name, row["samples"], row["task_accuracy"], row["first_turn_accuracy"],
                             row["mean_reward"], row["mean_turns"], row["latency_p95_ms"]])
    lines = [
        "# Blind paired evaluation results",
        "",
        "Numbers below are computed exclusively from immutable trajectory files, not reference snapshots.",
        "",
        "| Policy | N | Final EX | First-turn EX | Average turns | P95 latency (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["runs"].items():
        lines.append(
            f"| {name} | {row['samples']} | {row['task_accuracy']:.2%} | "
            f"{row['first_turn_accuracy']:.2%} | {row['mean_turns']:.2f} | "
            f"{row['latency_p95_ms']:.1f} |"
        )
    lines.extend(["", "## Paired change versus Base", "",
                  "| Policy | Gain (pp) | 95% paired bootstrap CI (pp) | Improved | Regressed | Exact McNemar p |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, diff in report["paired_vs_base"].items():
        lo, hi = diff["bootstrap_95_ci_pp"]
        lines.append(
            f"| {name} | {diff['gain_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
            f"{diff['improved_tasks']} | {diff['regressed_tasks']} | "
            f"{diff['mcnemar_exact_two_sided_p']:.4g} |"
        )
    lines.extend(["", "This comparison alone does not establish training causality; checkpoint and no-update evidence are separately required."])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "evaluation_protocol.json").write_text(
        json.dumps({name: run[1] for name, run in runs.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(args.leakage_audit, output / "leakage_audit.json") if Path(args.leakage_audit).resolve() != (output / "leakage_audit.json").resolve() else None
    print("\n".join(lines))


if __name__ == "__main__":
    main()
