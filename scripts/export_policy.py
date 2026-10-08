#!/usr/bin/env python3
"""Export a genuine veRL FSDP actor checkpoint to vLLM-loadable HF weights."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from checkpoint_manifest import build_manifest


def find_latest_actor(checkpoint_root: Path) -> Path:
    root = checkpoint_root.resolve()
    if (root / "fsdp_config.json").is_file():
        return root
    candidates = []
    for path in root.rglob("actor"):
        match = re.search(r"global_step_(\d+)", str(path))
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError("No global_step_N/actor FSDP directory found")
    return max(candidates, key=lambda pair: pair[0])[1]


def export(checkpoint_root: Path, target: Path) -> dict:
    source = find_latest_actor(checkpoint_root)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty export: {target}")
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp", "--local_dir", str(source),
        "--target_dir", str(target.resolve()),
    ]
    subprocess.run(command, check=True)
    if not (target / "config.json").exists():
        raise RuntimeError("FSDP export produced no HF config.json")
    manifest = build_manifest(target)
    metadata = {
        "source_actor": str(source), "target_hf": str(target.resolve()),
        "merger": "python -m verl.model_merger merge --backend fsdp",
        "file_manifest": manifest,
    }
    (target / "export_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--target-dir", required=True)
    args = ap.parse_args()
    result = export(Path(args.checkpoint_root), Path(args.target_dir))
    print(json.dumps({
        "source_actor": result["source_actor"], "target": result["target_hf"],
        "weights": result["file_manifest"]["weight_file_count"],
    }))


if __name__ == "__main__":
    main()
