"""Sparse rate-based network whose trainable synapses follow a fixed graph topology."""

from __future__ import annotations

import torch
from torch import nn


class ConnectomeRateNetwork(nn.Module):
    """Leaky recurrent scalar-node network with sparse edge message passing."""

    def __init__(
        self,
        node_count: int,
        edge_index: torch.Tensor,
        synapse_count: torch.Tensor,
        *,
        leak: float = 0.5,
        weight_scale: float = 0.05,
        trainable_weights: bool = True,
        normalize_incoming: bool = True,
        transmitter_sign: torch.Tensor | None = None,
        trainable_bias: bool = True,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if edge_index.shape[0] != 2 or edge_index.shape[1] != len(synapse_count):
            raise ValueError("edge_index must have shape [2, E] matching synapse_count")
        self.node_count = node_count
        self.leak = leak
        if not 0.0 <= leak <= 1.0:
            raise ValueError("leak must be between 0 and 1")
        if activation not in {"tanh", "relu"}:
            raise ValueError("activation must be 'tanh' or 'relu'")
        self.activation = activation
        self.register_buffer("edge_index", edge_index.long())
        weights = torch.log1p(synapse_count.float()) * weight_scale
        if normalize_incoming and len(weights):
            denominator = torch.zeros(node_count, dtype=weights.dtype)
            denominator.index_add_(0, edge_index[1], weights.abs())
            weights = weights / denominator[edge_index[1]].clamp_min(1e-6)
        self.edge_weight = nn.Parameter(weights, requires_grad=trainable_weights)
        signs = torch.zeros_like(weights) if transmitter_sign is None else transmitter_sign.float()
        if signs.shape != weights.shape or not torch.all(
            (signs == -1) | (signs == 0) | (signs == 1)
        ):
            raise ValueError("transmitter_sign must contain one value from {-1, 0, 1} per edge")
        self.register_buffer("transmitter_sign", signs)
        self.bias = nn.Parameter(torch.zeros(node_count), requires_grad=trainable_bias)

    @property
    def effective_edge_weight(self) -> torch.Tensor:
        """Weights with optional Dale-style edge signs enforced at every forward pass."""
        return torch.where(
            self.transmitter_sign == 0,
            self.edge_weight,
            self.transmitter_sign * self.edge_weight.abs(),
        )

    def forward(
        self,
        external_input: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        steps: int = 1,
    ) -> torch.Tensor:
        """Advance node states; inputs and states are shaped ``[batch, nodes]``."""
        if external_input.ndim != 2 or external_input.shape[1] != self.node_count:
            raise ValueError(f"external_input must have shape [batch, {self.node_count}]")
        hidden = torch.zeros_like(external_input) if state is None else state
        source, target = self.edge_index
        for _ in range(steps):
            messages = hidden[:, source] * self.effective_edge_weight
            aggregated = torch.zeros_like(hidden).index_add(1, target, messages)
            drive = self.leak * hidden + aggregated + external_input + self.bias
            hidden = torch.tanh(drive) if self.activation == "tanh" else torch.relu(drive)
        return hidden
