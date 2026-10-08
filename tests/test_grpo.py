import torch

from agentic_rl_sql.grpo import grpo_clipped_objective, group_relative_advantages


def test_group_relative_advantages_center_each_group():
    rewards = torch.tensor([1.0, 0.0, 0.5, 2.0, 2.0, 2.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    adv = group_relative_advantages(rewards, groups)
    assert torch.isclose(adv[groups == 0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(adv[groups == 1], torch.zeros(3))


def test_grpo_objective_is_finite_and_differentiable():
    new = torch.tensor([[-1.0, -0.8], [-1.1, -0.9]], requires_grad=True)
    old = torch.tensor([[-1.1, -0.9], [-1.0, -0.8]])
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones_like(new)
    loss, metrics = grpo_clipped_objective(new, old, adv, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert new.grad is not None
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
