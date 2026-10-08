#!/usr/bin/env python3
"""Aggregate measured GPU memory, utilization, power and elapsed time."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


def _float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def summarize(samples: list[dict], hourly_gpu_usd: float | None = None) -> dict:
    valid = [r for r in samples if r.get("gpus")]
    if not valid:
        raise ValueError("no successful GPU telemetry samples; resource metrics unavailable")
    all_mem = []
    all_util = []
    all_power = []
    timestamps = []
    for item in valid:
        timestamps.append(datetime.fromisoformat(item["time_utc"]))
        for gpu in item["gpus"]:
            mem = _float(gpu.get("memory.used"))
            util = _float(gpu.get("utilization.gpu"))
            watt = _float(gpu.get("power.draw"))
            if mem is not None:
                all_mem.append(mem)
            if util is not None:
                all_util.append(util)
            if watt is not None:
                all_power.append(watt)
    duration = max(0.0, (max(timestamps) - min(timestamps)).total_seconds())
    result = {
        "telemetry_sample_count": len(valid),
        "telemetry_error_count": len(samples) - len(valid),
        "gpu_count": max(len(r["gpus"]) for r in valid),
        "gpu_memory_peak_mib": max(all_mem) if all_mem else None,
        "gpu_memory_mean_mib": statistics.mean(all_mem) if all_mem else None,
        "gpu_utilization_mean_percent": statistics.mean(all_util) if all_util else None,
        "gpu_power_mean_watts": statistics.mean(all_power) if all_power else None,
        "observed_span_seconds": duration,
        "span_is_lower_bound": True,
    }
    if hourly_gpu_usd is not None:
        if hourly_gpu_usd < 0:
            raise ValueError("GPU hourly price must not be negative")
        result["assumed_usd_per_gpu_hour"] = hourly_gpu_usd
        result["estimated_observed_span_cost_usd"] = hourly_gpu_usd * result["gpu_count"] * duration / 3600
        result["cost_note"] = "Price is user-supplied; estimate excludes startup/shutdown and provider charges."
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu-hourly-rate-usd", type=float)
    args = ap.parse_args()
    samples = [
        json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = summarize(samples, args.gpu_hourly_rate_usd)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
