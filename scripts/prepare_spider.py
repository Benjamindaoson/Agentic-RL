#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl_sql.dataset import write_parquet
from agentic_rl_sql.types import SqlTask


def find_spider_root(path: Path) -> Path:
    candidates = [path, path / "spider", path / "spider_data"]
    candidates.extend(p.parent for p in path.rglob("train_spider.json"))
    for candidate in candidates:
        if (candidate / "train_spider.json").exists() and (candidate / "dev.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate train_spider.json and dev.json under {path}")


def db_path(root: Path, db_id: str) -> Path:
    candidates = [root / "database" / db_id / f"{db_id}.sqlite", root / "database" / db_id / f"{db_id}.db"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"database not found for {db_id}")


def load_tasks(root: Path, filename: str, split: str, max_turns: int, context_limit: int, explicit_check: bool) -> list[SqlTask]:
    rows = json.loads((root / filename).read_text(encoding="utf-8"))
    tasks: list[SqlTask] = []
    for index, row in enumerate(rows):
        db_id = str(row["db_id"])
        tasks.append(SqlTask(
            task_id=f"spider-{split}-{index:05d}", dataset="spider-1.0", split=split, db_id=db_id,
            db_path=str(db_path(root, db_id)), question=str(row["question"]),
            gold_sql=str(row.get("query", row.get("SQL", ""))), evidence="", max_turns=max_turns,
            context_limit=context_limit, metadata={"explicit_check": explicit_check},
        ))
    return tasks


def main():
    ap = argparse.ArgumentParser(description="Prepare official Spider 1.0 for Agent Lightning training and evaluation.")
    ap.add_argument("--spider-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--turns", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--contexts", nargs="+", type=int, default=[2048, 4096])
    ap.add_argument("--include-check-ablation", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    root = find_spider_root(Path(args.spider_root))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"spider_root": str(root), "variants": []}
    for context in args.contexts:
        for turns in args.turns:
            train = load_tasks(root, "train_spider.json", "train", turns, context, False)
            val = load_tasks(root, "dev.json", "val", turns, context, False)
            train_path = out / f"train_ctx{context}_turn{turns}.parquet"
            val_path = out / f"val_ctx{context}_turn{turns}.parquet"
            write_parquet(train, train_path)
            write_parquet(val, val_path)
            manifest["variants"].append({"context": context, "turns": turns, "explicit_check": False, "train": str(train_path), "val": str(val_path), "train_samples": len(train), "val_samples": len(val)})
        if args.include_check_ablation and 3 in args.turns:
            train = load_tasks(root, "train_spider.json", "train", 3, context, True)
            val = load_tasks(root, "dev.json", "val", 3, context, True)
            train_path = out / f"train_ctx{context}_turn3_check.parquet"
            val_path = out / f"val_ctx{context}_turn3_check.parquet"
            write_parquet(train, train_path)
            write_parquet(val, val_path)
            manifest["variants"].append({"context": context, "turns": 3, "explicit_check": True, "train": str(train_path), "val": str(val_path), "train_samples": len(train), "val_samples": len(val)})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
