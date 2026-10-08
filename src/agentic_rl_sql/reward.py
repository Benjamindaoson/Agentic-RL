from __future__ import annotations

from .types import ExecutionResult, RewardBreakdown


def compute_reward(execution: ExecutionResult, execution_match: bool, turn: int, *, retry_cost: float = 0.02) -> RewardBreakdown:
    """Only execution-equivalent results can receive a reward above 0.5."""
    valid_sql = 0.03 if execution.valid else 0.0
    executable_sql = 0.07 if execution.executed else 0.0
    success = 0.90 if execution_match else 0.0
    unsafe_penalty = -1.0 if not execution.safe else 0.0
    invalid_penalty = -0.20 if not execution.valid else 0.0
    timeout_penalty = -0.15 if execution.error_type == "timeout" else 0.0
    retry_penalty = -retry_cost * max(turn - 1, 0)
    total = success + valid_sql + executable_sql + unsafe_penalty + invalid_penalty + timeout_penalty + retry_penalty
    total = max(-1.0, min(1.0, total))
    notes: list[str] = []
    if execution_match:
        notes.append("execution-equivalent result")
    if not execution.safe:
        notes.append("unsafe SQL blocked")
    if execution.error_type:
        notes.append(execution.error_type)
    return RewardBreakdown(total, success, valid_sql, executable_sql, unsafe_penalty, invalid_penalty, retry_penalty, timeout_penalty, notes)
