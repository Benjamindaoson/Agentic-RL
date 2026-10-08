#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
SPIDER_RAW="${SPIDER_RAW:-$DATA_ROOT/raw/spider}"
SPIDER_ELIGIBLE="${SPIDER_ELIGIBLE:-$DATA_ROOT/spider_eligible}"
SQL_MAX_ROWS="${SQL_MAX_ROWS:-5000}"

python scripts/download_spider.py --output-dir "$SPIDER_RAW"
python scripts/prepare_spider.py \
  --spider-root "$SPIDER_RAW" \
  --output-dir "$DATA_ROOT/spider"
python scripts/validate_datasets.py \
  --spider-manifest "$DATA_ROOT/spider/manifest.json" \
  --skip-gold-execution \
  --output "$DATA_ROOT/spider/structural_audit.json"
python scripts/filter_gold_sql.py \
  --input-manifest "$DATA_ROOT/spider/manifest.json" \
  --output-dir "$SPIDER_ELIGIBLE" \
  --max-rows "$SQL_MAX_ROWS"
python scripts/validate_datasets.py \
  --spider-manifest "$SPIDER_ELIGIBLE/manifest.json" \
  --max-rows "$SQL_MAX_ROWS" \
  --output "$SPIDER_ELIGIBLE/gold_execution_audit.json"

python scripts/download_bird.py --output-dir "$DATA_ROOT/raw/bird" --no-clone-mini-dev
python scripts/validate_bird_metadata.py \
  --input "$DATA_ROOT/raw/bird/bird23_train_filtered.jsonl" \
  --output "$DATA_ROOT/raw/bird/metadata_audit.json"

echo "Spider Gold-eligible data and BIRD filtered-train metadata validated."
echo "For independent BIRD Mini-Dev SQLite data, use scripts/prepare_bird_minidev.py."
