from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SqlTask:
    task_id: str
    dataset: str
    split: str
    db_id: str
    db_path: str
    question: str
    gold_sql: str
    evidence: str = ""
    max_turns: int = 3
    context_limit: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SqlTask":
        known = {
            "task_id", "dataset", "split", "db_id", "db_path", "question", "gold_sql",
            "evidence", "max_turns", "context_limit", "metadata",
        }
        payload = {k: value[k] for k in known if k in value}
        required = {"task_id", "dataset", "split", "db_id", "db_path", "question", "gold_sql"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"SqlTask missing fields: {missing}")
        payload.setdefault("evidence", "")
        payload.setdefault("max_turns", 3)
        payload.setdefault("context_limit", 4096)
        payload.setdefault("metadata", {})
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionResult:
    sql: str
    valid: bool
    safe: bool
    executed: bool
    rows: list[list[Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    elapsed_ms: float = 0.0
    truncated: bool = False
    order_sensitive: bool = False
    row_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RewardBreakdown:
    total: float
    execution_match: float
    valid_sql: float
    executable_sql: float
    unsafe_penalty: float
    invalid_penalty: float
    retry_penalty: float
    timeout_penalty: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrajectoryStep:
    turn: int
    prompt_kind: str
    raw_model_output: str
    sql: str | None
    execution: ExecutionResult
    execution_match: bool
    feedback: str
    checker_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "prompt_kind": self.prompt_kind,
            "raw_model_output": self.raw_model_output,
            "sql": self.sql,
            "execution": self.execution.to_dict(),
            "execution_match": self.execution_match,
            "feedback": self.feedback,
            "checker_output": self.checker_output,
        }


@dataclass(slots=True)
class Trajectory:
    task: SqlTask
    steps: list[TrajectoryStep]
    reward: RewardBreakdown
    success: bool
    final_sql: str | None
    total_elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "reward": self.reward.to_dict(),
            "success": self.success,
            "final_sql": self.final_sql,
            "total_elapsed_ms": self.total_elapsed_ms,
        }
