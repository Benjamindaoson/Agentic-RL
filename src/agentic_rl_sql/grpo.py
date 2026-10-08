from __future__ import annotations

import torch


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize rewards within each prompt group, as used by GRPO."""
    if rewards.ndim != 1 or group_ids.ndim != 1 or len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must be aligned 1-D tensors")
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        values = rewards[mask].float()
        mean = values.mean()
        std = values.std(unbiased=False)
        advantages[mask] = values - mean if std < eps else (values - mean) / (std + eps)
    return advantages


def grpo_clipped_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.3,
    reference_log_probs: torch.Tensor | None = None,
    kl_coefficient: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level clipped GRPO objective with optional reference KL control."""
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != response_mask.shape:
        raise ValueError("log probabilities and response mask must have the same shape")
    if advantages.ndim != 1 or advantages.shape[0] != new_log_probs.shape[0]:
        raise ValueError("advantages must have one value per sampled sequence")
    mask = response_mask.float()
    ratio = torch.exp(new_log_probs - old_log_probs)
    sequence_adv = advantages[:, None]
    unclipped = ratio * sequence_adv
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    clipped = clipped_ratio * sequence_adv
    policy_term = torch.minimum(unclipped, clipped)
    denom = mask.sum().clamp_min(1.0)
    policy_loss = -(policy_term * mask).sum() / denom
    kl_loss = torch.zeros((), device=new_log_probs.device, dtype=new_log_probs.dtype)
    if reference_log_probs is not None and kl_coefficient > 0:
        if reference_log_probs.shape != new_log_probs.shape:
            raise ValueError("reference_log_probs shape mismatch")
        log_ratio = reference_log_probs - new_log_probs
        per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
        kl_loss = (per_token_kl * mask).sum() / denom
    total = policy_loss + kl_coefficient * kl_loss
    clipped_fraction = (((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)).float() * mask).sum() / denom
    metrics = {
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "clip_fraction": float(clipped_fraction.detach().cpu()),
        "mean_ratio": float(((ratio * mask).sum() / denom).detach().cpu()),
    }
    return total, metrics
