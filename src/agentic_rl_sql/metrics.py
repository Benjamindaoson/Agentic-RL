from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Iterable


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    position = (len(xs) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return xs[lo]
    weight = position - lo
    return xs[lo] * (1 - weight) + xs[hi] * weight


def summarize_trajectories(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {"samples": 0}
    success = [bool(r.get("success")) for r in rows]
    rewards = [float(r.get("reward", {}).get("total", 0.0)) for r in rows]
    turns = [len(r.get("steps", [])) for r in rows]
    latencies = [float(r.get("total_elapsed_ms", 0.0)) for r in rows]
    error_types = Counter()
    invalid = unsafe = first_turn_success = 0
    for row in rows:
        steps = row.get("steps", [])
        if steps and steps[0].get("execution_match"):
            first_turn_success += 1
        for step in steps:
            ex = step.get("execution", {})
            if not ex.get("valid", False):
                invalid += 1
            if not ex.get("safe", False):
                unsafe += 1
            if ex.get("error_type"):
                error_types[str(ex["error_type"])] += 1
    total_steps = sum(turns)
    return {
        "samples": len(rows),
        "task_accuracy": sum(success) / len(rows),
        "first_turn_accuracy": first_turn_success / len(rows),
        "mean_reward": statistics.fmean(rewards),
        "mean_turns": statistics.fmean(turns),
        "invalid_sql_rate": invalid / max(total_steps, 1),
        "unsafe_sql_rate": unsafe / max(total_steps, 1),
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "error_types": dict(error_types),
    }
