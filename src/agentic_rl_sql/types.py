from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SqlTask:
    """Private dataset record. Gold SQL must never be passed to the policy."""
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
    def from_dict(cls, value: dict[str, Any]) -> SqlTask:
        required = {"task_id", "dataset", "split", "db_id", "db_path", "question", "gold_sql"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"SqlTask missing fields: {missing}")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: value[k] for k in known if k in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def policy_view(self) -> PolicyTask:
        """Allowlisted observable inputs only: no answer or oracle-derived state."""
        return PolicyTask(
            task_id=self.task_id, dataset=self.dataset, split=self.split,
            db_id=self.db_id, db_path=self.db_path, question=self.question,
            evidence=self.evidence, max_turns=self.max_turns,
            context_limit=self.context_limit,
            metadata={"explicit_check": bool(self.metadata.get("explicit_check", False))},
        )


@dataclass(slots=True)
class PolicyTask:
    """Only information available to a deployed policy."""
    task_id: str
    dataset: str
    split: str
    db_id: str
    db_path: str
    question: str
    evidence: str = ""
    max_turns: int = 3
    context_limit: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)

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
    feedback: str
    decision: str = "final"
    checker_output: str | None = None
    prompt_tokens: int | None = None
    schema_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn, "prompt_kind": self.prompt_kind,
            "raw_model_output": self.raw_model_output, "sql": self.sql,
            "execution": self.execution.to_dict(), "feedback": self.feedback,
            "decision": self.decision, "checker_output": self.checker_output,
            "prompt_tokens": self.prompt_tokens, "schema_truncated": self.schema_truncated,
        }


@dataclass(slots=True)
class Trajectory:
    """Blind rollout; no success, reward, correctness or gold fields."""
    task: PolicyTask
    steps: list[TrajectoryStep]
    final_sql: str | None
    total_elapsed_ms: float
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "final_sql": self.final_sql, "total_elapsed_ms": self.total_elapsed_ms,
            "stop_reason": self.stop_reason,
        }
