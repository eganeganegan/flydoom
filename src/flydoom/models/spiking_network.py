"""Optional sparse leaky integrate-and-fire controller with surrogate gradients."""

from __future__ import annotations

import torch
from torch import nn


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (voltage,) = ctx.saved_tensors
        return (gradient / (1 + voltage.abs()).square(),)


class SparseLIFNetwork(nn.Module):
    """Simple LIF approximation; it is not a physiological fly-neuron simulation."""

    def __init__(
        self, node_count: int, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> None:
        super().__init__()
        self.node_count = node_count
        self.register_buffer("edge_index", edge_index.long())
        self.edge_weight = nn.Parameter(edge_weight.float())

    def forward(
        self, current: torch.Tensor, voltage: torch.Tensor | None = None, decay: float = 0.9
    ) -> tuple[torch.Tensor, torch.Tensor]:
        voltage = torch.zeros_like(current) if voltage is None else voltage
        source, target = self.edge_index
        previous_spikes = _SurrogateSpike.apply(voltage - 1.0)
        messages = previous_spikes[:, source] * self.edge_weight
        incoming = torch.zeros_like(voltage).index_add(1, target, messages)
        voltage = decay * voltage + incoming + current
        spikes = _SurrogateSpike.apply(voltage - 1.0)
        voltage = voltage * (1.0 - spikes)
        return spikes, voltage
