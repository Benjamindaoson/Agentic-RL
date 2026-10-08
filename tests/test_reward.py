from pathlib import Path

import pytest

from agentic_rl_sql.reward import RewardConfig, compute_reward, load_reward_config
from agentic_rl_sql.types import ExecutionResult


def test_only_success_can_receive_high_reward():
    valid_wrong = ExecutionResult(sql="SELECT 1", valid=True, safe=True, executed=True)
    wrong_reward = compute_reward(valid_wrong, False, turn=1)
    assert wrong_reward.total <= 0.5
    correct = compute_reward(valid_wrong, True, turn=1)
    assert correct.total == 1.0


def test_retry_and_unsafe_penalties():
    unsafe = ExecutionResult(sql="DROP TABLE x", valid=True, safe=False, executed=False)
    reward = compute_reward(unsafe, False, turn=3)
    assert reward.total < 0
    assert reward.retry_penalty == -0.04


def test_reward_yaml_drives_weights():
    path = Path(__file__).resolve().parents[1] / "configs" / "reward.yaml"
    cfg = load_reward_config(path)
    assert cfg.mode == "execution"
    assert cfg.execution_match == 0.9
    valid = ExecutionResult(sql="SELECT 1", valid=True, safe=True, executed=True)
    assert compute_reward(valid, True, 1, config=cfg).total == 1.0
    ablated = cfg.with_mode("validity_only")
    assert compute_reward(valid, True, 1, config=ablated).total == 0.1


def test_invalid_reward_config_fails_loudly():
    with pytest.raises(ValueError):
        RewardConfig(valid_sql=0.8).validate()
    with pytest.raises(ValueError):
        RewardConfig(retry_cost_per_extra_turn=0.02).validate()
