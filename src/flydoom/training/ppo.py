"""Compact clipped PPO implementation for connectome actor-critic policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flydoom.training.rollout import RecurrentState, RolloutBuffer


def _stack_states(states: list[RecurrentState]) -> RecurrentState:
    first = states[0]
    if isinstance(first, tuple):
        return tuple(torch.cat([state[index] for state in states]) for index in range(2))
    return torch.cat(states)


def _select_state(state: RecurrentState, indices: torch.Tensor) -> RecurrentState:
    if isinstance(state, tuple):
        return state[0][indices], state[1][indices]
    return state[indices]


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
    reward_scale: float = 1.0


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
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(1e-8)
        recurrent_states = (
            _stack_states(buffer.recurrent_states) if buffer.recurrent_states else None
        )
        samples: list[dict[str, float]] = []
        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(actions), device=actions.device)
            for start in range(0, len(actions), self.config.batch_size):
                indices = permutation[start : start + self.config.batch_size]
                if recurrent_states is None:
                    log_probs, entropy, values = self.policy.evaluate_actions(
                        observations[indices], actions[indices]
                    )
                else:
                    log_probs, entropy, values = self.policy.evaluate_actions_recurrent(
                        observations[indices],
                        actions[indices],
                        _select_state(recurrent_states, indices),
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
                samples.append({
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "policy_entropy": float(entropy.mean().detach()),
                    "gradient_norm": float(gradient_norm),
                    "approx_kl": float(
                        ((ratio - 1) - (log_probs - old_log_probs[indices])).mean().detach()
                    ),
                    "clip_fraction": float(
                        ((ratio - 1).abs() > self.config.clip_range).float().mean().detach()
                    ),
                })
        return {
            name: sum(sample[name] for sample in samples) / len(samples)
            for name in samples[0]
        }
