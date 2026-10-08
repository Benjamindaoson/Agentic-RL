#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/run_experiment_matrix.py --stage ablations --execute \
  --data-dir "${DATA_DIR:-$ROOT/data/spider_eligible}" \
  --output-dir "${RUN_ROOT:-$ROOT/runs/matrix}" "$@"
