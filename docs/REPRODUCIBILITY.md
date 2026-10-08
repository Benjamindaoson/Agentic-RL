# Reproducibility and GPU acceptance

## 1. Preflight

Use Python 3.12, a CUDA-compatible GPU stack, matching Agent Lightning/veRL/vLLM versions, and datasets with locally accessible SQLite database files.

~~~bash
python scripts/preflight.py --require-gpus 1
python scripts/audit_leakage.py --output artifacts/leakage_audit.json
python -m pytest -q
~~~

## 2. Fixed datasets

~~~bash
python scripts/download_spider.py --output-dir data/raw/spider
python scripts/prepare_spider.py --spider-root data/raw/spider --output-dir data/spider
python scripts/run_experiment_matrix.py --stage main
~~~

Confirm all three database partitions are disjoint: `train_*` and `val_*` come from official Spider Train (group split by DB), while `test_*` is reserved official Spider Dev for final blind evaluation. The trainer rejects Train/Val overlap. Base vs GRPO must use `test_ctx4096_turn1.parquet`, not `val_*`.

## 3. GPU training

~~~bash
export RUN_NAME=ctx4096_turn1_seed42
export MODEL_REVISION=YOUR_40_CHARACTER_HF_COMMIT_SHA
export TRAIN_FILE="$PWD/data/spider/train_ctx4096_turn1.parquet"
export VAL_FILE="$PWD/data/spider/val_ctx4096_turn1.parquet"
export CONTEXT_LENGTH=4096 MAX_TURNS=1
export ROLLOUT_MAX_TOKENS=1024
bash scripts/run_local_training.sh --seed 42 --epochs 1 --save-freq 1
~~~

Run status is **not** verified merely because the trainer returns. Inspect:

- `grpo_run_manifest.json` / `training_config.json`: exact model, config, seed, reward and dataset hashes.
- `training.log` / `training_metrics.jsonl`: real optimizer steps, loss, KL and GRPO metrics.
- `gpu_telemetry.jsonl` and `resource_metrics.json`: real peak memory, utilization and time span.
- `base_model_identity.json`: resolved immutable base weight files and SHA-256.
- `policy_checkpoint_manifest.json`: full checkpoint weights SHA-256.
- `leakage_audit.json`: Gold mutation invariance.

Pin the actual Hugging Face base model revision and validate the local weight identity before making a publication-level provenance claim.

## 4. FSDP export and vLLM

~~~bash
python scripts/export_policy.py \
  --checkpoint-root runs/ctx4096_turn1_seed42/checkpoints \
  --target-dir runs/ctx4096_turn1_seed42/hf-policy
~~~

Start one server at a time with `start_policy_server.sh`, run `run_rollouts.py` on the same Parquet/Token/turn budget, stop server before switching checkpoint. Use different output paths and real immutable identifiers via `--policy-checkpoint`. A saved checkpoint hash without a working vLLM reload is not enough.

## 5. Paired comparisons and control evidence

~~~bash
python scripts/compare_experiments.py \
  --base runs/eval_base/base_trajectories.jsonl \
  --grpo runs/eval_grpo/grpo_trajectories.jsonl \
  --no-update runs/eval_no_update/no_update_trajectories.jsonl \
  --reward-ablation runs/eval_validity_only/validity_trajectories.jsonl \
  --leakage-audit runs/ctx4096_turn1_seed42/leakage_audit.json \
  --output-dir runs/comparison
python scripts/check_weight_change.py \
  --base-hf-dir /path/to/pinned-base-hf \
  --grpo-hf-dir runs/ctx4096_turn1_seed42/hf-policy \
  --no-update-hf-dir /path/to/no-update-exported-hf \
  --output runs/comparison/weight_change_audit.json
python scripts/verify_evidence.py \
  --training-dir runs/ctx4096_turn1_seed42 \
  --comparison-dir runs/comparison \
  --output runs/comparison/evidence_verification.json
python scripts/build_report.py \
  --comparison-dir runs/comparison \
  --evidence-verification runs/comparison/evidence_verification.json \
  --output runs/comparison/REPORT.md
~~~

An incomplete evidence gate intentionally produces exit code 1. Fix the missing evidence; never replace it with reference snapshot numbers.

## 6. Acceptance criteria

- Deterministic leakage audit passes.
- At least one verified real GRPO optimizer update and a saved weight checkpoint.
- Hash actual pinned base weights and prove GRPO tensor values changed but No-update tensor values did not.
- Exported updated policy successfully reloads under vLLM.
- Frozen held-out task set, same public observations and compute budget.
- Paired Base / GRPO / No-update / validity-only results, with confidence interval.
- Database-generalization tests on untrained schemas, plus independent BIRD evaluation.
- Report actual GPU time, max memory and inference token usage; do not estimate these as measurements.

**Status:** Until the actual GPU run exists, all training claims remain unverified irrespective of green CPU CI.
