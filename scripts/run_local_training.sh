#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"

AGL_SERVER_PORT="${AGL_SERVER_PORT:-8181}"
AGL_KEY="${AGL_KEY:-agentic-rl-dev-key}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/spider/train_ctx4096_turn3.parquet}"
VAL_FILE="${VAL_FILE:-$ROOT/data/spider/val_ctx4096_turn3.parquet}"
RUN_NAME="${RUN_NAME:-qwen25_coder_3b_ctx4096_turn3}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/$RUN_NAME}"
mkdir -p "$LOG_DIR"

cleanup() {
  pkill -f 'agl-server' 2>/dev/null || true
  pkill -f 'agl-controller' 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
}
cleanup
trap cleanup EXIT INT TERM

ray start --head --dashboard-host=0.0.0.0 >"$LOG_DIR/ray.log" 2>&1

agl-server \
  port="$AGL_SERVER_PORT" \
  key="$AGL_KEY" \
  default_proxy.model_name="$MODEL" \
  >"$LOG_DIR/agl-server.log" 2>&1 &

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$AGL_SERVER_PORT/healthz" >/dev/null; then break; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$AGL_SERVER_PORT/healthz" >/dev/null

agl-controller \
  runner_type=local \
  agl_server.url="http://127.0.0.1:$AGL_SERVER_PORT" \
  agl_server.key="$AGL_KEY" \
  >"$LOG_DIR/agl-controller.log" 2>&1 &

python "$ROOT/scripts/train_sql_agent.py" \
  --train-file "$TRAIN_FILE" \
  --val-file "$VAL_FILE" \
  --model "$MODEL" \
  --run-name "$RUN_NAME" \
  --agl-base-url "http://127.0.0.1:$AGL_SERVER_PORT" \
  --agl-key "$AGL_KEY" \
  "$@"
