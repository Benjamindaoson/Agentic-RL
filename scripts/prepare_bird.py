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


def find_database(db_root: Path, db_id: str) -> Path | None:
    candidates = [db_root / db_id / f"{db_id}.sqlite", db_root / db_id / f"{db_id}.db", db_root / f"{db_id}.sqlite", db_root / f"{db_id}.db"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(db_root.rglob(f"{db_id}.sqlite")) + list(db_root.rglob(f"{db_id}.db"))
    return matches[0].resolve() if matches else None


def iter_json_rows(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            yield from value
        elif isinstance(value, dict) and isinstance(value.get("data"), list):
            yield from value["data"]
        else:
            raise ValueError("expected JSON list or JSONL")


def main():
    ap = argparse.ArgumentParser(description="Prepare BIRD-SQL metadata with locally downloaded SQLite databases.")
    ap.add_argument("--records", required=True)
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--context-limit", type=int, default=4096)
    ap.add_argument("--select-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--strict-db", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()
    db_root = Path(args.db_root).resolve()
    tasks = []
    skipped_db = skipped_non_select = 0
    for index, row in enumerate(iter_json_rows(Path(args.records))):
        db_id = str(row.get("db_id", row.get("database_id", "")))
        gold = str(row.get("SQL", row.get("query", row.get("sql", ""))))
        if args.select_only and not gold.lstrip().upper().startswith(("SELECT", "WITH")):
            skipped_non_select += 1
            continue
        db = find_database(db_root, db_id)
        if db is None:
            skipped_db += 1
            if args.strict_db:
                raise FileNotFoundError(f"database not found: {db_id}")
            continue
        tasks.append(SqlTask(
            task_id=f"bird-{args.split}-{index:05d}", dataset="bird-sql", split=args.split, db_id=db_id,
            db_path=str(db), question=str(row.get("question", "")), gold_sql=gold,
            evidence=str(row.get("evidence", "") or ""), max_turns=args.turns,
            context_limit=args.context_limit, metadata={"explicit_check": False},
        ))
    count = write_parquet(tasks, args.output)
    report = {"output": str(Path(args.output).resolve()), "samples": count, "skipped_missing_database": skipped_db, "skipped_non_select": skipped_non_select, "select_only": args.select_only}
    Path(args.output).with_suffix(".manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
