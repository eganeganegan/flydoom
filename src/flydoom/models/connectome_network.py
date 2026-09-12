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
    ) -> None:
        super().__init__()
        if edge_index.shape[0] != 2 or edge_index.shape[1] != len(synapse_count):
            raise ValueError("edge_index must have shape [2, E] matching synapse_count")
        self.node_count = node_count
        self.leak = leak
        self.register_buffer("edge_index", edge_index.long())
        weights = torch.log1p(synapse_count.float()) * weight_scale
        if normalize_incoming and len(weights):
            denominator = torch.zeros(node_count, dtype=weights.dtype)
            denominator.index_add_(0, edge_index[1], weights)
            weights = weights / denominator[edge_index[1]].clamp_min(1e-6)
        self.edge_weight = nn.Parameter(weights, requires_grad=trainable_weights)

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
            messages = hidden[:, source] * self.edge_weight
            aggregated = torch.zeros_like(hidden).index_add(1, target, messages)
            hidden = torch.tanh(self.leak * hidden + aggregated + external_input)
        return hidden
