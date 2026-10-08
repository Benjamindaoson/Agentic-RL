#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_spec(value: str):
    if "=" not in value:
        raise ValueError("experiment must be NAME=METRICS_JSON")
    return value.split("=", 1)


def main():
    ap = argparse.ArgumentParser(description="Compare context, turn-count and explicit-check ablations.")
    ap.add_argument("--experiment", action="append", required=True, help="NAME=metrics.json")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.experiment:
        name, path = parse_spec(spec)
        metrics = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append({"experiment": name, **metrics})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "samples",
        "task_accuracy",
        "first_turn_accuracy",
        "mean_reward",
        "mean_turns",
        "invalid_sql_rate",
        "unsafe_sql_rate",
        "latency_mean_ms",
        "latency_p95_ms",
        "model",
    ]
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Agentic RL 实验对照",
        "",
        "| 实验 | 最终准确率↑ | 首轮准确率↑ | 平均奖励↑ | 平均轮次↓ | 无效SQL率↓ | P95延迟(ms)↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row.get('task_accuracy', 0):.4f} | "
            f"{row.get('first_turn_accuracy', 0):.4f} | {row.get('mean_reward', 0):.4f} | "
            f"{row.get('mean_turns', 0):.2f} | {row.get('invalid_sql_rate', 0):.4f} | "
            f"{row.get('latency_p95_ms', 0):.1f} |"
        )
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
