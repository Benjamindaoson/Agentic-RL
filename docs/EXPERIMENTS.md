# Experiments and causal evidence

**Primary target**: Qwen2.5-Coder-3B-Instruct; held-out Spider databases.

## Required ordering

1. Test P0 leakage invariance and CPU integration.
2. GPU smoke GRPO (single-turn). Store optimizer metrics, model checkpoint and hashes.
3. Freeze held-out sample IDs and inference protocol: split official Spider Train into Train and Internal-Val by database; reserve official Dev exclusively as `test_*` for final comparison.
4. Evaluate the Base policy.
5. Export updated HF actor, evaluate it on the same tasks and budget.
6. Evaluate No-update LR=0 (identical training pathway but zero optimizer learning rate).
7. Evaluate validity-only reward ablation (success signal removed).
8. Repeat seeds and expand context/multi-turn matrix.
9. Run BIRD external held-out dataset.

## Main comparisons

| Experiment | Max prompt tokens | Max turns | Explicit checker | Causal question |
|---|---:|---:|---|---|
| ctx2048_turn1 | 2048 | 1 | No | Short-context Base/GRPO gain |
| ctx2048_turn3 | 2048 | 3 | No | Additional autonomous interaction |
| ctx2048_turn3_check | 2048 | 3 | Yes | Self-checker benefit/cost |
| ctx4096_turn1 | 4096 | 1 | No | Longer context at fixed turns |
| ctx4096_turn3 | 4096 | 3 | No | Longer context and multi-turn |
| no_update_ctx4096_turn1 | 4096 | 1 | No | Does the training pathway alone explain gain? |
| validity_only_ctx4096_turn1 | 4096 | 1 | No | Is the execution-match reward important? |

The 2048↔4096 and 1↔3 contrasts are not fixed-compute Base/GRPO tests. A training condition's Base and GRPO evaluations must use exactly its same context and turn budget.

## Artifacts and verdicts

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands. The evidence verifier requires real logged optimizer metrics, checkpoint weight files, GPU telemetry, a hashed pinned base model, demonstrated actual GRPO tensor changes, unchanged no-update tensor values, a passing leakage audit and all four compared policy categories.

**Do not publish the preexisting reference benchmark values (80.4%, 80.2%, etc.) as this repository's experimentally reproduced results.** Only the paired comparator and strict evidence report can support new measured claims.

Reward-hacking threats: coincidental SQL result match, SQL that just returns all rows, query truncation, insufficient enforcement of structured feedback, and optimizer changing format compliance instead of semantic reasoning. Report error types and qualitative corrections in addition to accuracy.
