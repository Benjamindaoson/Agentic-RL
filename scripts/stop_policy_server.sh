#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${LOG_DIR:-./logs/policy}"
PID_FILE="$LOG_DIR/vllm.pid"
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  kill "$PID" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Policy server stopped"
fi
