# Agentic-RL: implementation plan and acceptance gates

**Repo:** Benjamindaoson/Agentic-RL

## Stage 1 — P0 oracle isolation
- [x] Introduce public `PolicyTask` and private `SqlTask`.
- [x] Remove Gold-based continuation, success stopping and incorrect-answer prompts.
- [x] Separate `SqlEvaluator` from blind rollout.
- [x] Add deterministic metamorphic leakage audit and tests.
- [ ] Verify model-generated behavior using real vLLM rollouts on the target GPU.

## Stage 2 — Real GRPO training
- [x] Agent Lightning / veRL entrypoint with synchronized token budgets.
- [x] FSDP and optimizer offload, gradient checkpointing, vLLM GPU setting.
- [x] Save dataset hashes, validated train/val Schema split and training configuration.
- [x] Collect trainer log, step metrics, GPU telemetry and SHA-256 checkpoint evidence.
- [x] FSDP actor → HF export command.
- [ ] Complete GPU GRPO run and verify optimizer steps/weight change.
- [ ] Export and reload the trained model under vLLM.

## Stage 3 — Base versus trained policy
- [x] Deterministic blind evaluation and file-level immutable protocol.
- [x] No mixed-run resume; reject missing Gold and runner errors.
- [x] Per-task trajectories, aggregate metrics, runtime and token counts.
- [ ] Run Base and trained checkpoints on a frozen held-out task set.

## Stage 4 — Causal controls
- [x] Stage-separated GRPO experiment matrix.
- [x] No-update LR=0 and validity-only reward ablation.
- [x] Strict paired Bootstrap CI and McNemar analysis.
- [ ] Obtain real control checkpoints and measurements.
- [ ] Execute three or more independent training seeds.

## Stage 5 — Generalization and performance
- [x] BIRD data metadata + DB bind scripts already in repo.
- [x] Record schema truncation, invalid SQL, P95 latency and GPU telemetry.
- [ ] Run BIRD held-out external evaluation.
- [ ] Verify official BIRD scoring and compare with internal EX.
- [ ] Produce tokens/sec, peak GPU memory and cost per successful task from real runs.

## Stage 6 — Publication-quality reproducibility
- [x] CPU CI tests, leakage audit, matrix dry-run and shell syntax gate.
- [x] Evidence completeness checker and non-fabricating report builder.
- [x] README, strict evaluation protocol and reproducibility guide.
- [ ] Complete all proof gates against a real training run.
- [ ] Archive dataset/checkpoint/trajectory manifests in stable storage.

**Current status:** Software workflow implemented; experimental claims are NOT yet verified. The checked boxes represent authored code/tests, not GPU outcomes. No planned task can be marked experimentally complete without raw evidence.
