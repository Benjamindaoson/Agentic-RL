from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .types import ExecutionResult, RewardBreakdown


@dataclass(frozen=True, slots=True)
class RewardConfig:
    execution_match: float = 0.90
    valid_sql: float = 0.03
    executable_sql: float = 0.07
    unsafe_penalty: float = -1.0
    invalid_penalty: float = -0.20
    timeout_penalty: float = -0.15
    retry_cost_per_extra_turn: float = -0.02
    mode: str = "execution"

    def validate(self) -> RewardConfig:
        if self.mode not in {"execution", "validity_only"}:
            raise ValueError("reward mode must be execution or validity_only")
        if self.execution_match < 0 or self.valid_sql < 0 or self.executable_sql < 0:
            raise ValueError("reward gains must be non-negative")
        if any(v > 0 for v in (
            self.unsafe_penalty, self.invalid_penalty, self.timeout_penalty,
            self.retry_cost_per_extra_turn,
        )):
            raise ValueError("reward penalties cannot be positive")
        if self.valid_sql + self.executable_sql > 0.5:
            raise ValueError("incorrect execution must never earn more than 0.5")
        if self.mode == "execution" and self.execution_match + self.valid_sql + self.executable_sql <= 0.5:
            raise ValueError("a correct execution should earn more than 0.5")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def with_mode(self, mode: str) -> RewardConfig:
        return replace(self, mode=mode).validate()


def load_reward_config(path: str | Path, mode: str | None = None) -> RewardConfig:
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("reward"), dict):
        raise ValueError("reward config must contain a reward mapping")
    payload = data["reward"]
    unknown = set(payload) - set(RewardConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown reward fields: {sorted(unknown)}")
    cfg = RewardConfig(**payload).validate()
    return cfg.with_mode(mode) if mode else cfg


def compute_reward(
    execution: ExecutionResult, execution_match: bool, turn: int, *,
    retry_cost: float | None = None, config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Compute the final trajectory reward from isolated posthoc correctness."""
    cfg = (config or RewardConfig()).validate()
    valid_sql = cfg.valid_sql if execution.valid else 0.0
    executable_sql = cfg.executable_sql if execution.executed else 0.0
    match_reward = (
        cfg.execution_match if execution_match and cfg.mode == "execution" else 0.0
    )
    unsafe_penalty = cfg.unsafe_penalty if not execution.safe else 0.0
    invalid_penalty = cfg.invalid_penalty if not execution.valid else 0.0
    timeout_penalty = cfg.timeout_penalty if execution.error_type == "timeout" else 0.0
    retry_delta = -retry_cost if retry_cost is not None else cfg.retry_cost_per_extra_turn
    if retry_delta > 0:
        raise ValueError("retry cost must be a nonnegative cost")
    retry_penalty = retry_delta * max(turn - 1, 0)
    total = max(-1.0, min(1.0, (
        match_reward + valid_sql + executable_sql + unsafe_penalty
        + invalid_penalty + timeout_penalty + retry_penalty
    )))
    notes: list[str] = [f"mode={cfg.mode}"]
    if execution_match:
        notes.append("execution-equivalent result")
    if not execution.safe:
        notes.append("unsafe SQL blocked")
    if execution.error_type:
        notes.append(execution.error_type)
    return RewardBreakdown(
        total, match_reward, valid_sql, executable_sql, unsafe_penalty,
        invalid_penalty, retry_penalty, timeout_penalty, notes,
    )
