"""Compact clipped PPO implementation for connectome actor-critic policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flydoom.training.rollout import RolloutBuffer


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    batch_size: int = 64


class PPO:
    def __init__(self, policy: nn.Module, config: PPOConfig) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    def update(self, buffer: RolloutBuffer, last_value: torch.Tensor) -> dict[str, float]:
        advantages, returns = buffer.advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )
        observations = torch.cat(buffer.observations)
        actions = torch.stack(buffer.actions).flatten()
        old_log_probs = torch.stack(buffer.log_probs).detach().flatten()
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
        metrics: dict[str, float] = {}
        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(actions), device=actions.device)
            for start in range(0, len(actions), self.config.batch_size):
                indices = permutation[start : start + self.config.batch_size]
                log_probs, entropy, values = self.policy.evaluate_actions(
                    observations[indices], actions[indices]
                )
                ratio = (log_probs - old_log_probs[indices]).exp()
                unclipped = ratio * advantages[indices]
                clipped = (
                    ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range)
                    * advantages[indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = (returns[indices] - values).square().mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy.mean()
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                metrics = {
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "policy_entropy": float(entropy.mean().detach()),
                    "gradient_norm": float(gradient_norm),
                }
        return metrics
