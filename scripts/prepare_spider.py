#!/usr/bin/env python3
"""Spider grouped split: train DBs / internal-validation DBs / untouched official Dev test."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
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
    candidates = [
        root / "database" / db_id / f"{db_id}.sqlite",
        root / "database" / db_id / f"{db_id}.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"database not found for {db_id}")


def load_tasks(
    root: Path, filename: str, split: str, max_turns: int,
    context_limit: int, explicit_check: bool,
) -> list[SqlTask]:
    rows = json.loads((root / filename).read_text(encoding="utf-8"))
    tasks = []
    for index, row in enumerate(rows):
        db_id = str(row["db_id"])
        tasks.append(SqlTask(
            task_id=f"spider-{split}-{index:05d}",
            dataset="spider-1.0", split=split, db_id=db_id,
            db_path=str(db_path(root, db_id)), question=str(row["question"]),
            gold_sql=str(row.get("query", row.get("SQL", ""))), evidence="",
            max_turns=max_turns, context_limit=context_limit,
            metadata={"explicit_check": explicit_check},
        ))
    return tasks


def grouped_split(
    raw_train: list[SqlTask], *, seed: int = 42, val_fraction: float = 0.10,
) -> tuple[list[SqlTask], list[SqlTask]]:
    """Select entire unseen DBs for internal validation; official Dev is untouched."""
    if not (0 < val_fraction < 1):
        raise ValueError("internal validation fraction must be in (0,1)")
    schemas = sorted({t.db_id for t in raw_train})
    if len(schemas) < 2:
        raise ValueError("need at least two training databases for schema-disjoint split")
    random.Random(seed).shuffle(schemas)
    n_val = max(1, min(len(schemas) - 1, round(val_fraction * len(schemas))))
    val_db = set(schemas[:n_val])
    train, val = [], []
    for task in raw_train:
        if task.db_id in val_db:
            task.split = "val"
            task.task_id = task.task_id.replace("spider-train-", "spider-val-")
            val.append(task)
        else:
            train.append(task)
    if not train or not val:
        raise ValueError("empty group after Spider internal schema split")
    return train, val


def validate_disjoint(train: list[SqlTask], val: list[SqlTask], test: list[SqlTask]):
    sets = [{t.db_id for t in group} for group in (train, val, test)]
    labels = ("train", "val", "test")
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = sets[i] & sets[j]
            if overlap:
                raise ValueError(f"{labels[i]}/{labels[j]} database schema overlap: {sorted(overlap)}")


def main():
    ap = argparse.ArgumentParser(description="Prepare Spider with 3-way schema-disjoint split.")
    ap.add_argument("--spider-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--turns", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--contexts", nargs="+", type=int, default=[2048, 4096])
    ap.add_argument("--include-check-ablation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--internal-val-fraction", type=float, default=0.10)
    args = ap.parse_args()
    root = find_spider_root(Path(args.spider_root))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "spider_root": str(root),
        "split_seed": args.split_seed, "internal_val_fraction": args.internal_val_fraction,
        "test_source": "official Spider 1.0 dev.json (untouched by training/validation)",
        "test_is_independent": True, "variants": [],
    }
    for context in args.contexts:
        for turns in args.turns:
            variants = [False]
            if args.include_check_ablation and turns == 3:
                variants.append(True)
            for check in variants:
                if check and turns != 3:
                    continue
                train_raw = load_tasks(root, "train_spider.json", "train", turns, context, check)
                train, val = grouped_split(
                    train_raw, seed=args.split_seed, val_fraction=args.internal_val_fraction,
                )
                test = load_tasks(root, "dev.json", "test", turns, context, check)
                validate_disjoint(train, val, test)
                label = f"ctx{context}_turn{turns}" + ("_check" if check else "")
                paths = {name: out / f"{name}_{label}.parquet" for name in ("train", "val", "test")}
                sizes = {
                    "train": write_parquet(train, paths["train"]),
                    "val": write_parquet(val, paths["val"]),
                    "test": write_parquet(test, paths["test"]),
                }
                split_dbs = {
                    "train": sorted({t.db_id for t in train}),
                    "val": sorted({t.db_id for t in val}),
                    "test": sorted({t.db_id for t in test}),
                }
                manifest["variants"].append({
                    "name": label, "context": context, "turns": turns,
                    "explicit_check": check,
                    "files": {k: str(v.resolve()) for k, v in paths.items()},
                    "samples": sizes,
                    "database_counts": {k: len(v) for k, v in split_dbs.items()},
                    "database_ids_sha256": {
                        k: hashlib.sha256(json.dumps(v).encode("utf-8")).hexdigest()
                        for k, v in split_dbs.items()
                    },
                })
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
