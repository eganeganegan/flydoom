"""On-policy rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RolloutBuffer:
    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def advantages(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantage = torch.zeros_like(last_value)
        advantages: list[torch.Tensor] = []
        next_value = last_value
        for index in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[index])
            delta = self.rewards[index] + gamma * next_value * mask - self.values[index]
            advantage = delta + gamma * gae_lambda * mask * advantage
            advantages.append(advantage)
            next_value = self.values[index]
        result = torch.stack(list(reversed(advantages))).detach().flatten()
        returns = result + torch.stack(self.values).detach().flatten()
        return result, returns

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
