#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Download BIRD filtered training metadata and clone official Mini-Dev tooling.")
    ap.add_argument("--output-dir", default="data/raw/bird")
    ap.add_argument("--clone-mini-dev", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    from datasets import load_dataset
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("birdsql/bird23-train-filtered", split="train")
    metadata = out / "bird23_train_filtered.jsonl"
    with metadata.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    if args.clone_mini_dev:
        target = out / "mini_dev"
        if not target.exists():
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/bird-bench/mini_dev.git", str(target)], check=True)
    print(json.dumps({"metadata": str(metadata), "rows": len(ds), "database_note": "Download BIRD databases from the official dataset page and pass --db-root during preparation."}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
