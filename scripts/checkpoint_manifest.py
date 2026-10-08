#!/usr/bin/env python3
"""Hash every file in a locally saved policy checkpoint tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    weights = [p for p in files if p.suffix.lower() in WEIGHT_SUFFIXES]
    if not weights:
        raise ValueError("no model/optimizer weight shards under checkpoint root")
    entries = {
        str(path.relative_to(root)): {"sha256": hash_file(path), "size_bytes": path.stat().st_size}
        for path in files
    }
    return {
        "checkpoint_root": str(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": entries,
        "weight_file_count": len(weights),
        "total_bytes": sum(item["size_bytes"] for item in entries.values()),
        "note": "Hashes validate saved bytes, not learning gains or checkpoint loadability.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = build_manifest(Path(args.checkpoint_dir))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "weight_files": result["weight_file_count"],
        "total_bytes": result["total_bytes"],
        "output": str(path),
    }))


if __name__ == "__main__":
    main()
