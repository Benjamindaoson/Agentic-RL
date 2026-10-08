#!/usr/bin/env python3
"""Resolve and hash the ACTUAL base weights before training or serving."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

WEIGHT_EXTS = {".safetensors", ".bin", ".pt"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model(model: str, revision: str = "") -> tuple[Path, str]:
    local = Path(model).expanduser()
    if local.is_dir():
        return local.resolve(), revision or "LOCAL_WEIGHT_HASH"
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
        raise ValueError("Hugging Face model revision must be an immutable 40-hex commit SHA")
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(repo_id=model, revision=revision)
    return Path(snapshot).resolve(), revision.lower()


def model_identity(root: Path, *, original: str, revision: str) -> dict:
    root = root.resolve()
    weights = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in WEIGHT_EXTS]
    if not weights:
        raise ValueError("model path has no weight files")
    files = {
        str(p.relative_to(root)): {"sha256": hash_file(p), "size_bytes": p.stat().st_size}
        for p in weights
    }
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": "resolved_hashed", "original_model": original,
        "resolved_path": str(root), "revision": revision,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weight_shards": files, "weight_file_count": len(weights),
        "weights_fingerprint_sha256": aggregate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    path, revision = resolve_model(args.model, args.revision)
    result = model_identity(path, original=args.model, revision=revision)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path)  # usable in: export MODEL="$(python scripts/model_identity.py ...)"


if __name__ == "__main__":
    main()
