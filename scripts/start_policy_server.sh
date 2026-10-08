#!/usr/bin/env bash
set -euo pipefail
MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
PORT="${POLICY_PORT:-8000}"
PROMPT_BUDGET="${PROMPT_TOKEN_BUDGET:-4096}"
OUTPUT_BUDGET="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((PROMPT_BUDGET + OUTPUT_BUDGET))}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
LOG_DIR="${LOG_DIR:-./logs/policy}"
if (( MAX_MODEL_LEN < PROMPT_BUDGET + OUTPUT_BUDGET )); then
  echo "MAX_MODEL_LEN must include prompt and output token budgets" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"
if [[ -f "$LOG_DIR/vllm.pid" ]] && kill -0 "$(cat "$LOG_DIR/vllm.pid")" 2>/dev/null; then
  echo "Existing vLLM process detected in $LOG_DIR" >&2
  exit 2
fi
nohup vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --served-model-name sql-policy \
  >"$LOG_DIR/vllm.log" 2>&1 &
echo $! >"$LOG_DIR/vllm.pid"
echo "Started vLLM PID $(cat "$LOG_DIR/vllm.pid"); inspect logs and health before evaluation."
