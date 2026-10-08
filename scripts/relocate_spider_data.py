#!/usr/bin/env python3
"""Relocate prepared Spider datasets onto a different machine without GPU.

Rewrites ONLY SQLite paths in task records; database IDs, questions, Gold SQL,
split membership and experimental budgets remain unchanged. The output has
fresh SHA manifests and optionally reruns all Gold SQL on the destination DBs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_rl_sql.dataset import read_training_parquet, unpack_task, write_parquet
from agentic_rl_sql.db import connect_read_only
from scripts.validate_datasets import validate_spider


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def database_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    found: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".sqlite", ".db"):
            found.setdefault(path.stem, []).append(path.resolve())
    result = {}
    for name, paths in found.items():
        canonical = [p for p in paths if p.parent.name == name] or paths
        if len(canonical) != 1:
            raise ValueError(f"ambiguous SQLite database {name}: {canonical[:3]}")
        result[name] = canonical[0]
    if not result:
        raise ValueError(f"no SQLite files found under {root}")
    return result


def relocate(manifest: Path, db_root: Path, output_dir: Path, *,
             verify_gold: bool = False, timeout: float = 8.0,
             max_rows: int = 5000) -> dict:
    manifest = manifest.resolve()
    output_dir = output_dir.resolve()
    if timeout <= 0 or max_rows < 1:
        raise ValueError("invalid SQLite execution limits")
    if output_dir == manifest.parent:
        raise ValueError("output must differ from source")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    source = json.loads(manifest.read_text(encoding="utf-8"))
    if not source.get("variants"):
        raise ValueError("not a Spider experiment manifest")
    paths = {}
    for variant in source["variants"]:
        for split in ("train", "val", "test"):
            src = Path(variant["files"][split]).resolve()
            if not src.is_file():
                raise FileNotFoundError(src)
            paths[src] = output_dir / src.name
    if len(set(paths.values())) != len(paths):
        raise ValueError("target Parquet basenames collide")
    index = database_index(db_root.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for src, target in paths.items():
        rows = [unpack_task(record) for record in read_training_parquet(src)]
        if not rows:
            raise ValueError(f"empty Parquet: {src}")
        for task in rows:
            db = index.get(task.db_id)
            if db is None:
                raise FileNotFoundError(f"missing database: {task.db_id}")
            conn = connect_read_only(db)
            try:
                conn.execute("SELECT 1").fetchone()
            finally:
                conn.close()
            task.db_path = str(db)
        if write_parquet(rows, target) != len(rows):
            raise AssertionError("record count changed during relocation")
        files.append({
            "source": str(src), "source_sha256": file_hash(src),
            "target": str(target), "target_sha256": file_hash(target),
            "samples": len(rows),
        })
    for variant in source["variants"]:
        for split in ("train", "val", "test"):
            variant["files"][split] = str(paths[Path(variant["files"][split]).resolve()])
    source["relocation_only"] = True
    source["source_manifest_sha256"] = file_hash(manifest)
    source["relocation_db_root"] = str(db_root.resolve())
    source["relocated_utc"] = datetime.now(timezone.utc).isoformat()
    target_manifest = output_dir / "manifest.json"
    target_manifest.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_file = manifest.parent / "gold_eligibility_audit.json"
    if audit_file.is_file():
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        audit["filtered_manifest"] = str(target_manifest)
        audit["source_gold_eligibility_audit_sha256"] = file_hash(audit_file)
        (output_dir / "gold_eligibility_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    check = validate_spider(target_manifest, verify_gold=verify_gold,
                            timeout=timeout, max_rows=max_rows)
    if not check["structurally_validated"]:
        raise ValueError("relocated dataset failed structural integrity")
    evidence = {
        "status": "verified_relocation",
        "source_manifest_sha256": file_hash(manifest),
        "relocated_manifest": str(target_manifest),
        "relocated_manifest_sha256": file_hash(target_manifest),
        "relocated_files": files, "gold_reexecuted": verify_gold,
        "schema_disjoint": True, "gold_validation_complete": check["complete"],
    }
    (output_dir / "relocation_audit.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--verify-gold", action="store_true")
    ap.add_argument("--sql-timeout", type=float, default=8.0)
    ap.add_argument("--max-rows", type=int, default=5000)
    args = ap.parse_args()
    report = relocate(Path(args.source_manifest), Path(args.db_root), Path(args.output_dir),
                      verify_gold=args.verify_gold, timeout=args.sql_timeout,
                      max_rows=args.max_rows)
    print(json.dumps({"status": report["status"],
                      "relocated_files": len(report["relocated_files"]),
                      "gold_reexecuted": report["gold_reexecuted"]}, indent=2))


if __name__ == "__main__":
    main()
