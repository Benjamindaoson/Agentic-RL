#!/usr/bin/env python3
"""Extract genuine veRL per-step console metrics; NEVER substitute placeholder values."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
PATTERN = re.compile(r"([A-Za-z_][\w/.\-]*):\s*(" + NUMBER + r")")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_metrics(lines):
    for line in lines:
        line = ANSI.sub("", line.strip())
        payload = None
        if line.startswith("{"):
            try:
                candidate = json.loads(line)
                if isinstance(candidate, dict) and any(k in candidate for k in ("step", "global_step", "training/global_step")):
                    payload = candidate
            except json.JSONDecodeError:
                pass
        if payload is None and ("step:" in line or "global_step:" in line):
            pairs = PATTERN.findall(line)
            payload = {name: float(value) for name, value in pairs}
        if not payload:
            continue
        step = payload.get("step", payload.get("global_step", payload.get("training/global_step")))
        if step is None:
            continue
        try:
            step = int(step)
        except (TypeError, ValueError):
            continue
        if step < 0:
            continue
        numeric = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key] = value
        numeric["global_step"] = step
        if len(numeric) > 1:
            yield numeric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Captured unmodified trainer stdout/stderr")
    ap.add_argument("--output", required=True)
    ap.add_argument("--strict", action="store_true", help="Fail if no real optimizer-step metrics were found")
    args = ap.parse_args()
    log_path = Path(args.input)
    log_sha = hashlib.sha256(log_path.read_bytes()).hexdigest()
    with log_path.open(encoding="utf-8", errors="replace") as f:
        rows = list(parse_metrics(f))
    result = Path(args.output)
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.open("w", encoding="utf-8") as sink:
        for row in rows:
            row["training_log_sha256"] = log_sha
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"steps_extracted": len(rows), "training_log_sha256": log_sha}))
    if args.strict and not rows:
        raise SystemExit("NO TRAINING STEP METRICS FOUND: proof gate failed")


if __name__ == "__main__":
    main()
