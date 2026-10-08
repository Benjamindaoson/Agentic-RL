#!/usr/bin/env python3
"""Execute a staged matrix; dry-run by default to avoid unplanned GPU spending."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def plan(
    config: dict, data_dir: Path, output_dir: Path, stage: str,
    seeds: list[int] | None = None,
) -> list[dict]:
    if seeds is not None and (not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds)):
        raise ValueError("experiment seeds must be unique nonnegative integers")
    stages = config["stages"]
    names = list(stages) if stage == "all" else [stage]
    runs = []
    for stage_name in names:
        for item in stages[stage_name]:
            ctx, turns = int(item["context_length"]), int(item["max_turns"])
            check = "_check" if item.get("explicit_check", False) else ""
            dataset = f"ctx{ctx}_turn{turns}{check}"
            for seed in (seeds if seeds is not None else [42]):
                name = f"{item['name']}_seed{seed}" if seeds is not None else item["name"]
                env = {
                    "MODEL": str(config["model"]),
                    "CONTEXT_LENGTH": str(ctx), "MAX_TURNS": str(turns),
                    "REWARD_MODE": str(item["reward_mode"]),
                    "TRAIN_FILE": str((data_dir / f"train_{dataset}.parquet").resolve()),
                    "VAL_FILE": str((data_dir / f"val_{dataset}.parquet").resolve()),
                    "RUN_NAME": str(name),
                    "RUN_DIR": str((output_dir / name).resolve()),
                }
                extra = [
                    "--seed", str(seed),
                    "--epochs", str(item["epochs"]),
                    "--learning-rate", str(item["learning_rate"]),
                    "--explicit-check" if item["explicit_check"] else "--no-explicit-check",
                ]
                runs.append({
                    "stage": stage_name, "name": name, "seed": seed,
                    "env": env, "command": ["bash", "scripts/run_local_training.sh", *extra],
                })
    if len({r["name"] for r in runs}) != len(runs):
        raise ValueError("duplicate matrix run names")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_matrix.yaml")
    ap.add_argument("--data-dir", default="data/spider_eligible")
    ap.add_argument("--output-dir", default="runs/matrix")
    ap.add_argument("--stage", default="smoke",
                    choices=["smoke", "main", "ablations", "controls", "all"])
    ap.add_argument("--execute", action="store_true", help="Launch real GPU work")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Optional explicit reproducibility seeds, e.g. --seeds 42 123 2026")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    runs = plan(cfg, Path(args.data_dir), Path(args.output_dir), args.stage, seeds=args.seeds)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry-run",
        "stage": args.stage, "seeds": args.seeds or [42], "runs": [],
    }
    for row in runs:
        record = dict(row)
        record["status"] = "planned"
        print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
        if args.execute:
            for key in ("TRAIN_FILE", "VAL_FILE"):
                if not Path(record["env"][key]).exists():
                    raise FileNotFoundError(record["env"][key])
            proc = subprocess.run(
                record["command"], cwd=root,
                env={**os.environ, **record["env"]}, check=False,
            )
            record["status"] = "succeeded" if proc.returncode == 0 else "failed"
            record["exit_code"] = proc.returncode
        manifest["runs"].append(record)
        (output / "matrix_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        if record["status"] == "failed" and not args.keep_going:
            raise SystemExit(record["exit_code"])
    if not args.execute:
        print("DRY RUN ONLY: add --execute to launch real GPU training.")


if __name__ == "__main__":
    main()
