#!/usr/bin/env python3
"""Assemble only evidence-backed experiment results; no seeded snapshot metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-dir", required=True)
    ap.add_argument("--evidence-verification", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    comparison_dir = Path(args.comparison_dir)
    comparison = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    evidence = json.loads(Path(args.evidence_verification).read_text(encoding="utf-8"))
    status = "VERIFIED" if evidence.get("passed") else "INCOMPLETE — DO NOT PUBLISH AS VERIFIED"
    base = comparison["runs"]["base"]
    grpo = comparison["runs"]["grpo"]
    delta = comparison["paired_vs_base"]["grpo"]
    lines = [
        "# Agentic-RL evidence-backed experiment report",
        "",
        f"**Evidence status: {status}**",
        "",
        f"Protocol: {comparison['protocol']}",
        f"Dataset SHA-256: {comparison['dataset_sha256']}",
        f"Task IDs SHA-256: {comparison['task_ids_sha256']}",
        "",
        "## Base vs GRPO: same budget and same held-out task IDs",
        "",
        "| Result | Base | GRPO |",
        "|---|---:|---:|",
        f"| Final execution accuracy | {base['task_accuracy']:.2%} | {grpo['task_accuracy']:.2%} |",
        f"| First turn accuracy | {base['first_turn_accuracy']:.2%} | {grpo['first_turn_accuracy']:.2%} |",
        f"| Average turns | {base['mean_turns']:.2f} | {grpo['mean_turns']:.2f} |",
        f"| P95 latency (ms) | {base['latency_p95_ms']:.1f} | {grpo['latency_p95_ms']:.1f} |",
        "",
        f"Paired accuracy gain: {delta['gain_pp']:+.2f} pp",
        f"Paired bootstrap 95% CI: {delta['bootstrap_95_ci_pp']}",
        f"Exact two-sided McNemar p: {delta['mcnemar_exact_two_sided_p']:.4g}",
        "",
        "## Evidence checks",
        "",
    ]
    for name, ok in evidence.get("checks", {}).items():
        lines.append(f"- [{'x' if ok else ' '}] {name}")
    for item in evidence.get("problems", []):
        lines.append(f"- Missing evidence: {item}")
    lines.extend([
        "", "## Interpretation constraints", "",
        "A measured improvement is not proof of its cause without matched no-update and reward controls.",
        "The archived reports/benchmark_snapshot.* data are provided reference figures, NOT results measured by this run.",
        "No training claims should be made from this report if Evidence status is INCOMPLETE.",
    ])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
