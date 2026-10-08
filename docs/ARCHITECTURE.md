# Architecture: offline oracle, online blind policy

~~~text
Private SqlTask(dataset):
  question / schema / gold_sql / budgets
       | policy_view()        \ posthoc grading only
       v                       v
Public PolicyTask           SqlEvaluator
       |                       ^
       v                       |
Blind SqlAgentRunner --> Frozen Trajectory
       |                       |
       +-----------------------+
       |
  model selects SQL + final/inspect
       |
read-only SQLite execution
       |
observed rows / errors; gold NEVER consulted
       |
next model call only on inspect / observed failure / truncation
~~~

The model cannot read the label or determine whether its execution matched the expected result. Only the posthoc grader may calculate match/reward after the blind rollout has terminated.

Modules:
- `src/agentic_rl_sql/types.py` — private/public task boundary and raw trajectory types.
- `src/agentic_rl_sql/agent.py` — policy-facing state machine.
- `src/agentic_rl_sql/context.py` — tokenizer-based prompt budget with explicit Schema-only truncation.
- `src/agentic_rl_sql/execution.py` — SQLite execution and normalized result comparison.
- `src/agentic_rl_sql/evaluator.py` — Gold-only posthoc scoring.
- `src/agentic_rl_sql/reward.py` — YAML-driven RLVR reward / controlled ablation.
- `agent/sql_agent_entrypoint.py` — Agent Lightning reward event after trajectory completion.
- `scripts/train_sql_agent.py` — veRL config and training manifests.
- `scripts/run_rollouts.py` — blinded evaluation, immutable protocol, JSONL evidence.
- `scripts/compare_experiments.py` — paired statistical testing.
- `scripts/verify_evidence.py` — strict no-snapshot evidence gate.

Security note: the private task is loaded by the Agent Lightning wrapper for scoring. All model requests are constructed inside the blind runner from the allowlisted PolicyTask, not the private object. The architecture is therefore a software-enforced policy/grade boundary, not a hardware sandbox.
