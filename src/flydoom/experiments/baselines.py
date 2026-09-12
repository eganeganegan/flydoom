"""Conventional actor-critic baselines for parameter-budget matching."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from flydoom.env.observations import SimpleCNNEncoder


class MLPPolicy(nn.Module):
    def __init__(self, hidden_size: int, action_count: int = 5) -> None:
        super().__init__()
        self.encoder = SimpleCNNEncoder(hidden_size)
        self.hidden = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden(self.encoder(observation))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def act(self, observation: torch.Tensor, deterministic: bool = False):
        logits, value = self(observation)
        distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor):
        logits, value = self(observation)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(action), distribution.entropy(), value


class GRUPolicy(MLPPolicy):
    def __init__(self, hidden_size: int, action_count: int = 5) -> None:
        super().__init__(hidden_size, action_count)
        self.hidden = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observation)
        hidden = self.hidden(encoded, torch.zeros_like(encoded))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class LSTMPolicy(MLPPolicy):
    """Parameter-budgeted LSTM actor-critic control."""

    def __init__(self, hidden_size: int, action_count: int = 5) -> None:
        super().__init__(hidden_size, action_count)
        self.hidden = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observation)
        zeros = torch.zeros_like(encoded)
        hidden, _ = self.hidden(encoded, (zeros, zeros))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)
