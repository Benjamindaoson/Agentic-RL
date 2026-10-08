from agentic_rl_sql.reward import compute_reward
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
