"""Deterministic CPU optimizer smoke test for the local GRPO math.

This tests an actual torch optimizer step and nonzero weight updates, but is
NOT veRL on-policy GPU training and must never be cited as a benchmark gain.
"""
import torch

from agentic_rl_sql.grpo import group_relative_advantages, grpo_clipped_objective


def test_local_grpo_objective_changes_parameters_under_real_adam_step():
    torch.manual_seed(79)
    policy = torch.nn.Linear(4, 3, bias=False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.015, weight_decay=0.0)
    features = torch.eye(4, dtype=torch.float32)
    sampled_token = torch.tensor([0, 1, 2, 1])
    rewards = torch.tensor([1.0, 0.0, 0.2, 0.8])
    groups = torch.tensor([0, 0, 1, 1])
    advantages = group_relative_advantages(rewards, groups)
    assert advantages.shape == (4,)
    before = policy.weight.detach().clone()
    logits = policy(features)
    log_probs = torch.log_softmax(logits, dim=-1)
    generated = log_probs.gather(-1, sampled_token[:, None])
    old = generated.detach().clone()
    loss, metrics = grpo_clipped_objective(
        generated, old, advantages, torch.ones_like(generated),
        reference_log_probs=old.detach() - 0.01, kl_coefficient=0.001,
    )
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert policy.weight.grad is not None
    assert torch.count_nonzero(policy.weight.grad).item() > 0
    optimizer.step()
    assert not torch.allclose(before, policy.weight.detach())
    assert torch.isfinite(policy.weight).all()
