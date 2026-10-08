#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Build a report from actual rollout metric files.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Metrics JSON files or directories containing *_metrics.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    files = []
    for value in args.inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*_metrics.json")))
        elif path.is_file():
            files.append(path)
    if not files:
        raise SystemExit("no metrics files found")

    rows = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "task_accuracy" in data:
            rows.append((path.stem.replace("_metrics", ""), path, data))
    if not rows:
        raise SystemExit("no rollout metrics containing task_accuracy")

    lines = [
        "# 大模型在线强化学习与策略优化：实测报告",
        "",
        "该报告只读取实际 rollout 产物，不自动填充缺失指标。",
        "",
        "| 实验 | 样本 | 最终准确率 | 首轮准确率 | 平均奖励 | 平均轮次 | 无效SQL率 | P95延迟(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _, m in rows:
        lines.append(
            f"| {name} | {m.get('samples', 0)} | {m.get('task_accuracy', 0):.2%} | "
            f"{m.get('first_turn_accuracy', 0):.2%} | {m.get('mean_reward', 0):.4f} | "
            f"{m.get('mean_turns', 0):.2f} | {m.get('invalid_sql_rate', 0):.2%} | "
            f"{m.get('latency_p95_ms', 0):.1f} |"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
