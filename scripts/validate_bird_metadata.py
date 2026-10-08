#!/usr/bin/env python3
"""Real downloaded BIRD metadata audit (database files are a separate input)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit_metadata(file: str | Path, *, expected_rows: int = 6601) -> dict:
    source = Path(file).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    sha = hashlib.sha256()
    count = 0
    schema_ids = set()
    empty = []
    for line in source.read_bytes().splitlines(keepends=True):
        sha.update(line)
        if not line.strip():
            continue
        count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at record {count}: {exc}") from exc
        db_id = str(record.get("db_id", "") or "")
        sql = str(record.get("SQL", "") or "")
        question = str(record.get("question", "") or "")
        if not db_id or not sql or not question:
            empty.append(count)
        schema_ids.add(db_id)
    if expected_rows and count != expected_rows:
        raise ValueError(f"BIRD metadata expected {expected_rows} records, found {count}")
    if empty:
        raise ValueError(f"BIRD metadata missing db_id/question/SQL in {len(empty)} rows, e.g. {empty[:10]}")
    return {
        "passed": True, "scope": "official filtered train metadata only (not full BIRD DB coverage)",
        "metadata_file": str(source), "metadata_sha256": sha.hexdigest(),
        "rows": count, "unique_db_ids": len(schema_ids),
        "all_records_have_sql_and_question": True,
        "databases_downloaded_and_verified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-rows", type=int, default=6601)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_metadata(args.input, expected_rows=args.expected_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
