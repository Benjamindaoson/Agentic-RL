# Blind Evaluation Protocol — v1

## Scientific question

Does online GRPO update a Text-to-SQL policy in a way that improves correctness **without** granting the trained policy extra access to oracle correctness or additional inference opportunities?

## Non-negotiable data separation

- Private `SqlTask` contains `gold_sql`; evaluator/trainer can access it.
- `PolicyTask` is a strict allowlist: question, schema location, evidence, context and turn budget, public metadata.
- `SqlAgentRunner` only accepts `PolicyTask`. It never receives Gold SQL or a success flag.
- `SqlEvaluator` consumes frozen trajectories after all policy calls terminate; only it evaluates Gold execution equivalence.
- Grading cannot change prompts, checker invocation, continuation or stopping decisions.
- A real SQLite execution error is allowed feedback; oracle correctness of a successfully executed query is not.
- Model may request `inspect`, and execution failures can cause a retry if there is remaining budget.
- Default `final` terminates after execution even if the Gold evaluator would subsequently mark the answer incorrect.

## P0 metamorphic leakage test

`python scripts/audit_leakage.py --output artifacts/leakage_audit.json`

Construct two datasets with identical observable inputs but different reference SQL. Re-run the same deterministic policy outputs and assert:

1. All prompts and policy-generated SQL are identical.
2. Policy call count, stopping reason, and verifier routing are identical.
3. No private Gold SQL or correctness flag appears in the blind trajectory.
4. Posthoc grading changes as expected for different labels.

The audit is a **synthetic deterministic check**, not proof about external model accuracy.

## Comparison contract

Base and GRPO are evaluated on exactly the same task IDs and Parquet SHA, with:

- Same policy-visible task text, schema, evidence, max turns and explicit-check mode.
- Same tokenizer, prompt limit, maximum response tokens, model sampling temperature and seed.
- Same read-only SQLite environment, execution timeout, max rows and Schema render budget.
- Same reward/evaluator protocol; gold SQL may only be read after the entire rollout completes.
- No failed samples silently dropped; any runner error makes the experiment invalid.
- The Grading batch computes final EX and first-turn EX **after** policy termination.
- Checkpoint and base revisions must be documented separately from the served model alias.

`compare_experiments.py` enforces paired task sets, dataset fingerprint, sampler settings and budget. A comparison between 1-turn and 3-turn policies is an inference-budget study, not evidence of fixed-budget policy improvement.

## Statistics

- Primary metric: held-out Final Execution Accuracy.
- Secondary: first-turn EX, invalid/dangerous SQL rate, average turns, latency mean and P95, context truncation, token budget.
- Paired change: per-task (GRPO success - Base success) averaged across exactly matched task IDs.
- Uncertainty: paired bootstrap 95% percentile interval with recorded bootstrap seed and replicate count.
- McNemar: exact two-sided binomial statistic of improved versus regressed tasks.
- Repeat across at least three independent training seeds before making a robust generalization claim.

## Important limitations

Execution-equivalence on one fixed SQLite instance can be an imperfect oracle: coincidental result equality does not imply SQL semantic equivalence. Row caps and timeout can exclude queries; invalid Gold is a dataset error, not a failed policy prediction. Scored results should include the handling and coverage of such cases.

Self-checker receives only question/Schema/observed execution. A model's own self-critique is **not** a ground-truth semantic verifier. Report oracle-assisted experiments separately if ever reintroduced.

External BIRD evaluation must be described as internal execution-result evaluation unless run and validated with official benchmark scripts.
