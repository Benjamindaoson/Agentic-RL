#!/usr/bin/env python3
"""Official Spider 1.0 downloader with ZIP path/bomb protections and SHA manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SPIDER_FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
MAX_TOTAL_UNCOMPRESSED_BYTES = 5 * 1024**3
MAX_MEMBER_BYTES = 2 * 1024**3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_archive(zf: zipfile.ZipFile) -> dict:
    total = 0
    n = 0
    for entry in zf.infolist():
        raw = entry.filename.replace("\\", "/")
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or not raw.strip() or raw.startswith("//"):
            raise ValueError(f"unsafe ZIP path: {entry.filename!r}")
        if ":" in path.parts[0]:
            raise ValueError(f"unsafe absolute Windows-style ZIP path: {entry.filename!r}")
        mode = entry.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"ZIP symlink not allowed: {entry.filename!r}")
        if entry.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"ZIP member exceeds size budget: {entry.filename}")
        total += entry.file_size
        if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP exceeds maximum allowed uncompressed size")
        n += 1
    return {"members": n, "uncompressed_bytes": total}


def extract_verified_archive(archive: Path, output: Path) -> dict:
    with zipfile.ZipFile(archive) as zf:
        stats = validate_archive(zf)
        zf.extractall(output)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/spider")
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--archive", help="Optional already-downloaded official Spider ZIP (offline)")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive).resolve() if args.archive else output / "spider_data.zip"
    if not archive.is_file():
        try:
            import gdown
        except ImportError as exc:
            raise SystemExit("Install gdown: pip install gdown") from exc
        result = gdown.download(id=SPIDER_FILE_ID, output=str(archive), quiet=False)
        if not result or not archive.is_file():
            raise RuntimeError("Spider download failed; do not treat partial data as valid")
    digest = sha256(archive)
    stats = extract_verified_archive(archive, output)
    candidates = list(output.rglob("train_spider.json"))
    if not candidates:
        raise ValueError("ZIP was extracted but train_spider.json was not found")
    if not any((path.parent / "dev.json").is_file() for path in candidates):
        raise ValueError("Spider dataset has no dev.json beside train_spider.json")
    report = {
        "source": "Spider 1.0 official Google Drive archive", "gdrive_file_id": SPIDER_FILE_ID,
        "archive_sha256": digest, "archive_bytes": archive.stat().st_size,
        **stats, "extracted_root": str(candidates[0].parent.resolve()),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "verified_archive_structure": True,
    }
    (output / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.keep_zip and not args.archive:
        archive.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
