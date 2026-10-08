#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

SPIDER_FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"


def main():
    ap = argparse.ArgumentParser(description="Download the official Spider 1.0 release from the Yale/XLang link.")
    ap.add_argument("--output-dir", default="data/raw/spider")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Install gdown: pip install gdown") from exc
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "spider_data.zip"
    if not archive.exists():
        gdown.download(id=SPIDER_FILE_ID, output=str(archive), quiet=False)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out)
    if not args.keep_zip:
        archive.unlink(missing_ok=True)
    candidates = list(out.rglob("train_spider.json"))
    if not candidates:
        raise SystemExit("Spider archive extracted but train_spider.json was not found")
    print(candidates[0].parent)


if __name__ == "__main__":
    main()
