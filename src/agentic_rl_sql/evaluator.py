from __future__ import annotations

from dataclasses import dataclass

from .execution import execute_sql, result_sets_equal
from .reward import compute_reward
from .types import SqlTask, Trajectory


class InvalidGoldSQL(RuntimeError):
    """Evaluation cannot proceed if reference query is invalid or truncated."""


@dataclass(slots=True)
class SqlEvaluator:
    timeout_seconds: float = 8.0
    max_rows: int = 5000

    def evaluate(self, task: SqlTask, trajectory: Trajectory) -> dict:
        """Oracle is consulted only AFTER the whole policy trajectory is frozen."""
        if task.task_id != trajectory.task.task_id or task.db_path != trajectory.task.db_path:
            raise ValueError("task and trajectory identity mismatch")
        gold = execute_sql(
            task.db_path, task.gold_sql, timeout_seconds=self.timeout_seconds,
            max_rows=self.max_rows,
        )
        if not gold.executed or gold.truncated:
            raise InvalidGoldSQL(f"Gold SQL invalid or truncated: {task.task_id}: {gold.error_type}")
        matches = [result_sets_equal(s.execution, gold) for s in trajectory.steps]
        success = bool(matches and matches[-1])
        first_turn_success = bool(matches and matches[0])
        if not trajectory.steps:
            raise ValueError("cannot grade empty trajectory")
        reward = compute_reward(trajectory.steps[-1].execution, success, len(trajectory.steps))
        report = trajectory.to_dict()
        report.update({
            "success": success, "first_turn_success": first_turn_success,
            "reward": reward.to_dict(),
            "evaluation": {
                "protocol": "blind-final-v1",
                "gold_access": "post_rollout_only",
                "final_execution_match": success,
                "first_turn_execution_match": first_turn_success,
                "per_turn_execution_match_posthoc": matches,
                "gold_executed": True,
            },
        })
        return report
