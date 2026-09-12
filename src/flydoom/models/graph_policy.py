"""Actor-critic policy constrained to sensory inputs and descending readouts."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from flydoom.env.observations import FlyInspiredVisualEncoder, SimpleCNNEncoder
from flydoom.models.connectome_network import ConnectomeRateNetwork


class ConnectomePolicy(nn.Module):
    """Encode pixels into sensory nodes and read actions/value from descending nodes."""

    def __init__(
        self,
        network: ConnectomeRateNetwork,
        sensory_indices: torch.Tensor,
        descending_indices: torch.Tensor,
        action_count: int = 5,
        encoder: str = "cnn",
        propagation_steps: int = 5,
        training_mode: str = "trainable_internal",
    ) -> None:
        super().__init__()
        if not len(sensory_indices) or not len(descending_indices):
            raise ValueError("Policy requires sensory and descending neuron populations")
        self.network = network
        self.register_buffer("sensory_indices", sensory_indices.long())
        self.register_buffer("descending_indices", descending_indices.long())
        if encoder not in {"cnn", "fly"}:
            raise ValueError("encoder must be 'cnn' or 'fly'")
        self.encoder = SimpleCNNEncoder(64) if encoder == "cnn" else FlyInspiredVisualEncoder()
        self.input_mapping = nn.Linear(self.encoder.output_dim, len(sensory_indices))
        self.action_readout = nn.Linear(len(descending_indices), action_count)
        self.value_readout = nn.Linear(len(descending_indices), 1)
        self.propagation_steps = propagation_steps
        if training_mode not in {"trainable_internal", "fixed_internal", "readout_only"}:
            raise ValueError(f"Unknown training mode: {training_mode}")
        if training_mode in {"fixed_internal", "readout_only"}:
            self.network.edge_weight.requires_grad_(False)
            self.network.bias.requires_grad_(False)
        if training_mode == "readout_only":
            for module in (self.encoder, self.input_mapping):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value, _ = self.forward_with_activity(observation)
        return logits, value

    def forward_with_activity(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return policy outputs plus the full node state for synchronized recording."""
        latent = self.encoder(observation)
        external = latent.new_zeros((latent.shape[0], self.network.node_count))
        external[:, self.sensory_indices] = torch.tanh(self.input_mapping(latent))
        state = self.network(external, steps=self.propagation_steps)
        descending = state[:, self.descending_indices]
        return self.action_readout(descending), self.value_readout(descending).squeeze(-1), state

    def act(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(observation)
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value

    def evaluate_actions(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(observation)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(action), distribution.entropy(), value
