#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
export AGL_SERVER_PORT="${AGL_SERVER_PORT:-8181}"
export AGL_KEY="${AGL_KEY:-agentic-rl-dev-key}"
export MODEL="${MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/spider/train_ctx4096_turn1.parquet}"
export VAL_FILE="${VAL_FILE:-$ROOT/data/spider/val_ctx4096_turn1.parquet}"
export RUN_NAME="${RUN_NAME:-qwen25_coder_3b_ctx4096_turn1}"
export RUN_DIR="${RUN_DIR:-$ROOT/runs/$RUN_NAME}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
export MAX_TURNS="${MAX_TURNS:-1}"
export REWARD_CONFIG="${REWARD_CONFIG:-$ROOT/configs/reward.yaml}"
export REWARD_MODE="${REWARD_MODE:-execution}"
export POLICY_TOKENIZER_PATH="${POLICY_TOKENIZER_PATH:-$MODEL}"
export POLICY_PROMPT_TOKEN_LIMIT="$CONTEXT_LENGTH"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-1024}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"

RESUME_ARGS=()
TEE_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  if [[ ! -f "$RUN_DIR/grpo_run_manifest.json" ]]; then
    echo "RESUME=1 requires an existing experiment manifest" >&2
    exit 2
  fi
  RESUME_ARGS+=(--resume-mode auto)
  TEE_ARGS+=(-a)
elif [[ -f "$RUN_DIR/grpo_run_manifest.json" ]]; then
  echo "Run manifest already exists: $RUN_DIR. Select another RUN_NAME or RESUME=1." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
export MODEL_REVISION="${MODEL_REVISION:-}"
if [[ -d "$MODEL" || -n "$MODEL_REVISION" ]]; then
  MODEL="$(python scripts/model_identity.py --model "$MODEL" --revision "$MODEL_REVISION" --output "$RUN_DIR/base_model_identity.json")"
  export MODEL
  export POLICY_TOKENIZER_PATH="$MODEL"
else
  echo "Base model not pinned. To pass publication evidence gate, set MODEL_REVISION to an immutable Hugging Face commit SHA." >&2
fi
python scripts/preflight.py --require-gpus "${GPUS:-1}"

SERVER_PID=""
CONTROLLER_PID=""
GPU_PID=""
RAY_STARTED=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$GPU_PID" ]]; then kill "$GPU_PID" 2>/dev/null || true; fi
  if [[ -n "$CONTROLLER_PID" ]]; then kill "$CONTROLLER_PID" 2>/dev/null || true; fi
  if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi
  if [[ "$RAY_STARTED" == 1 ]]; then ray stop --force >/dev/null 2>&1 || true; fi
  exit "$status"
}
trap cleanup EXIT INT TERM

ray start --head --dashboard-host=127.0.0.1 >"$RUN_DIR/ray.log" 2>&1
RAY_STARTED=1
agl-server \
  port="$AGL_SERVER_PORT" \
  key="$AGL_KEY" \
  default_proxy.model_name="$MODEL" \
  >"$RUN_DIR/agl-server.log" 2>&1 &
SERVER_PID=$!

ready=0
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$AGL_SERVER_PORT/healthz" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "Agent Lightning server failed readiness check" >&2
  exit 1
fi

agl-controller \
  runner_type=local \
  agl_server.url="http://127.0.0.1:$AGL_SERVER_PORT" \
  agl_server.key="$AGL_KEY" \
  >"$RUN_DIR/agl-controller.log" 2>&1 &
CONTROLLER_PID=$!

python scripts/gpu_telemetry.py --output "$RUN_DIR/gpu_telemetry.jsonl" &
GPU_PID=$!

python -u scripts/train_sql_agent.py \
  --train-file "$TRAIN_FILE" \
  --val-file "$VAL_FILE" \
  --model "$MODEL" \
  --base-model-revision "$MODEL_REVISION" \
  --run-name "$RUN_NAME" \
  --run-dir "$RUN_DIR" \
  --context-length "$CONTEXT_LENGTH" \
  --max-turns "$MAX_TURNS" \
  --max-response-length "$ROLLOUT_MAX_TOKENS" \
  --reward-config "$REWARD_CONFIG" \
  --reward-mode "$REWARD_MODE" \
  --gpus "${GPUS:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.65}" \
  --agl-base-url "http://127.0.0.1:$AGL_SERVER_PORT" \
  --agl-key "$AGL_KEY" \
  "${RESUME_ARGS[@]}" "$@" 2>&1 | tee "${TEE_ARGS[@]}" "$RUN_DIR/training.log"

python scripts/extract_training_metrics.py \
  --input "$RUN_DIR/training.log" \
  --output "$RUN_DIR/training_metrics.jsonl" --strict
python scripts/checkpoint_manifest.py \
  --checkpoint-dir "$RUN_DIR/checkpoints" \
  --output "$RUN_DIR/policy_checkpoint_manifest.json"
if [[ -n "$GPU_PID" ]]; then
  kill "$GPU_PID" 2>/dev/null || true
  wait "$GPU_PID" 2>/dev/null || true
  GPU_PID=""
fi
python scripts/summarize_resources.py \
  --input "$RUN_DIR/gpu_telemetry.jsonl" \
  --output "$RUN_DIR/resource_metrics.json"
python scripts/audit_leakage.py --output "$RUN_DIR/leakage_audit.json"
echo "Training, checkpoint and leakage gates returned. Export the policy and run blind evaluation."
