#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
SPIDER_RAW="${SPIDER_RAW:-$DATA_ROOT/raw/spider}"

python "$ROOT/scripts/download_spider.py" --output-dir "$SPIDER_RAW"
python "$ROOT/scripts/prepare_spider.py" \
  --spider-root "$SPIDER_RAW" \
  --output-dir "$DATA_ROOT/spider"

python "$ROOT/scripts/download_bird.py" \
  --output-dir "$DATA_ROOT/raw/bird"

echo "Spider is ready. For BIRD, download official databases and run prepare_bird.py with --db-root."
