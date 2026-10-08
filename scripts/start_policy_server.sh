#!/usr/bin/env bash
set -euo pipefail
MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
PORT="${POLICY_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LOG_DIR="${LOG_DIR:-./logs/policy}"
mkdir -p "$LOG_DIR"

nohup vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --served-model-name sql-policy \
  >"$LOG_DIR/vllm.log" 2>&1 &
echo $! > "$LOG_DIR/vllm.pid"
echo "Policy endpoint: http://127.0.0.1:${PORT}/v1"
