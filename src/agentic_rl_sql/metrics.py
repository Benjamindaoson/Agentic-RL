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
    lo, hi = math.floor(position), math.ceil(position)
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
    first_success = [bool(r.get("first_turn_success")) for r in rows]
    errors = Counter()
    invalid = unsafe = total_tokens = measured = truncations = 0
    for row in rows:
        if row.get("runner_error"):
            errors["runner_error"] += 1
        for step in row.get("steps", []):
            ex = step.get("execution", {})
            if not ex.get("valid", False):
                invalid += 1
            if not ex.get("safe", False):
                unsafe += 1
            if ex.get("error_type"):
                errors[str(ex["error_type"])] += 1
            if step.get("schema_truncated"):
                truncations += 1
            if step.get("prompt_tokens") is not None:
                measured += 1
                total_tokens += int(step["prompt_tokens"])
    n_steps = sum(turns)
    return {
        "samples": len(rows), "task_accuracy": sum(success) / len(rows),
        "first_turn_accuracy": sum(first_success) / len(rows),
        "mean_reward": statistics.fmean(rewards),
        "mean_turns": statistics.fmean(turns),
        "invalid_sql_rate": invalid / max(n_steps, 1),
        "unsafe_sql_rate": unsafe / max(n_steps, 1),
        "prompt_tokens_measured": measured,
        "prompt_tokens_mean": total_tokens / measured if measured else None,
        "schema_truncation_rate": truncations / max(n_steps, 1),
        "runner_error_count": sum("runner_error" in r for r in rows),
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "error_types": dict(errors),
    }
