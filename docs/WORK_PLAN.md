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
- [x] Add SHA-256 locked canonical Spider mirror with source provenance and cache.
- [x] Add auditable, consistent Gold-SQL eligibility filtering for all Spider ablations.
- [x] Split Spider official Train into disjoint Train and Internal-Val databases, reserving official Dev as final Test.
- [x] Resolve pinned base checkpoint and hash actual weight shards.
- [x] Collect trainer log, step metrics, GPU telemetry and SHA-256 checkpoint evidence.
- [x] FSDP actor → HF export command.
- [x] Compare tensor values among Base, GRPO and No-update exports.
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
- [x] BIRD public filtered-train metadata downloaded and SHA/pinned revision validated in CPU CI.
- [x] Add official BIRD Mini-Dev SQLite CPU download, 500 SELECT-only preparation and explicit Gold-exclusion manifest.
- [x] BIRD data metadata + DB bind scripts already in repo.
- [x] Record schema truncation, invalid SQL, P95 latency and GPU telemetry.
- [x] Aggregate observed GPU memory/power/time; optional price assumptions remain labeled as estimates.
- [x] Implement strict held-out BIRD evaluation command.
- [ ] Run real *model* BIRD held-out external evaluation (requires trained or baseline model inference).
- [ ] Verify official BIRD scoring and compare with internal EX.
- [ ] Produce tokens/sec, peak GPU memory and cost per successful task from real runs.

## Stage 6 — Publication-quality reproducibility
- [x] CPU CI tests, leakage audit, matrix dry-run and shell syntax gate.
- [x] Verify actual Agent Lightning 1.0.2 and veRL 0.8.0 Hydra configuration (CPU).
- [x] Add real OpenAI SDK HTTP wire-level integration with SQLite and posthoc reward.
- [x] Add complete dataset integrity auditing with per-task Gold errors and disclosed test coverage.
- [x] Evidence completeness checker and non-fabricating report builder.
- [x] README, strict evaluation protocol and reproducibility guide.
- [ ] Complete all proof gates against a real training run.
- [ ] Archive dataset/checkpoint/trajectory manifests in stable storage.

**Current status:** Software workflow implemented; experimental claims are NOT yet verified. The checked boxes represent authored code/tests, not GPU outcomes. No planned task can be marked experimentally complete without raw evidence.
