#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data/spider}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"

run_one() {
  local context="$1" turns="$2" suffix="$3"
  export MODEL
  export TRAIN_FILE="$DATA_DIR/train_ctx${context}_turn${turns}${suffix}.parquet"
  export VAL_FILE="$DATA_DIR/val_ctx${context}_turn${turns}${suffix}.parquet"
  export RUN_NAME="qwen25_coder_3b_ctx${context}_turn${turns}${suffix}"
  bash "$ROOT/scripts/run_local_training.sh" --context-length "$context"
}

run_one 2048 1 ""
run_one 2048 3 ""
run_one 2048 3 "_check"
run_one 4096 1 ""
run_one 4096 3 ""
