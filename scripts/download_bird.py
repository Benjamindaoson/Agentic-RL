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
    # Resolve the public dataset snapshot before writing any files. The SHA pins
    # the exact metadata bytes across machines and avoids silent dataset drift.
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info("birdsql/bird23-train-filtered").sha
    if not revision or len(revision) != 40:
        raise RuntimeError("could not resolve immutable BIRD filtered train revision")
    ds = load_dataset("birdsql/bird23-train-filtered", split="train", revision=revision)
    metadata = out / "bird23_train_filtered.jsonl"
    with metadata.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    import hashlib
    digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
    source_manifest = {"dataset": "birdsql/bird23-train-filtered",
                       "revision": revision, "rows": len(ds),
                       "metadata_file": str(metadata), "metadata_sha256": digest}
    (out / "download_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.clone_mini_dev:
        target = out / "mini_dev"
        if not target.exists():
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/bird-bench/mini_dev.git", str(target)], check=True)
    print(json.dumps({**source_manifest, "database_note": "Full BIRD SQLite databases are a separate download and are not verified by metadata-only checks."}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
