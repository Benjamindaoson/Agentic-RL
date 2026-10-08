#!/usr/bin/env python3
"""Low-overhead read-only GPU telemetry, producing raw timestamped JSONL."""
from __future__ import annotations

import argparse
import csv
import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ("index", "name", "utilization.gpu", "memory.used", "memory.total", "power.draw")
STOP = False


def stop_handler(signum, frame):
    global STOP
    STOP = True


def sample() -> list[dict]:
    command = [
        "nvidia-smi", "--query-gpu=" + ",".join(FIELDS),
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, timeout=10)
    rows = []
    for line in csv.reader(output.splitlines()):
        if len(line) != len(FIELDS):
            raise ValueError("unexpected nvidia-smi CSV format")
        gpu = {field: value.strip() for field, value in zip(FIELDS, line)}
        rows.append(gpu)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()
    if args.interval < 1:
        ap.error("interval must be at least 1 second")
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        while not STOP:
            now = datetime.now(timezone.utc).isoformat()
            try:
                data = {"time_utc": now, "gpus": sample()}
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                data = {"time_utc": now, "error": f"{type(exc).__name__}: {exc}"}
            sink.write(json.dumps(data) + "\n")
            sink.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
