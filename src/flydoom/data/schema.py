"""Canonical tables and indexed graph representation used by all data adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

NEURON_COLUMNS = (
    "neuron_id",
    "cell_type",
    "super_class",
    "neurotransmitter",
    "hemisphere",
    "brain_region",
    "is_sensory",
    "is_motor",
    "is_descending",
    "metadata",
)
EDGE_COLUMNS = (
    "pre_neuron_id",
    "post_neuron_id",
    "synapse_count",
    "confidence",
    "neurotransmitter",
    "region",
)


@dataclass(slots=True)
class ConnectomeGraph:
    """Validated canonical connectome tables with stable contiguous node indices."""

    neurons: pd.DataFrame
    edges: pd.DataFrame

    def __post_init__(self) -> None:
        self.neurons = self.neurons.copy()
        self.edges = self.edges.copy()
        self._fill_optional_columns()
        self.validate()

    def _fill_optional_columns(self) -> None:
        defaults: dict[str, Any] = {
            "cell_type": "unknown",
            "super_class": "unknown",
            "neurotransmitter": "unknown",
            "hemisphere": "unknown",
            "brain_region": "unknown",
            "is_sensory": False,
            "is_motor": False,
            "is_descending": False,
            "metadata": "{}",
        }
        for column, value in defaults.items():
            if column not in self.neurons:
                self.neurons[column] = value
        edge_defaults: dict[str, Any] = {
            "synapse_count": 1.0,
            "confidence": 1.0,
            "neurotransmitter": "unknown",
            "region": "unknown",
        }
        for column, value in edge_defaults.items():
            if column not in self.edges:
                self.edges[column] = value

    def validate(self) -> None:
        """Reject ambiguous IDs, invalid endpoints, and non-positive edge weights."""
        required_neurons = {"neuron_id"}
        required_edges = {"pre_neuron_id", "post_neuron_id"}
        if missing := required_neurons.difference(self.neurons.columns):
            raise ValueError(f"Missing neuron columns: {sorted(missing)}")
        if missing := required_edges.difference(self.edges.columns):
            raise ValueError(f"Missing edge columns: {sorted(missing)}")
        if self.neurons["neuron_id"].duplicated().any():
            raise ValueError("neuron_id values must be unique")
        known = set(self.neurons["neuron_id"].tolist())
        endpoints = set(self.edges["pre_neuron_id"]) | set(self.edges["post_neuron_id"])
        if unknown := endpoints.difference(known):
            raise ValueError(f"Edges reference unknown neuron IDs: {list(unknown)[:5]}")
        if (pd.to_numeric(self.edges["synapse_count"], errors="coerce") <= 0).any():
            raise ValueError("synapse_count must be positive")

    @property
    def node_count(self) -> int:
        return len(self.neurons)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def id_to_index(self) -> dict[Any, int]:
        return {neuron_id: index for index, neuron_id in enumerate(self.neurons["neuron_id"])}

    def sparse_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source/destination indices and synapse counts without a dense matrix."""
        mapping = self.id_to_index
        sources = self.edges["pre_neuron_id"].map(mapping).to_numpy()
        targets = self.edges["post_neuron_id"].map(mapping).to_numpy()
        edge_index = torch.from_numpy(np.stack((sources, targets))).long()
        edge_weight = torch.tensor(
            self.edges["synapse_count"].to_numpy(dtype="float32"), dtype=torch.float32
        )
        return edge_index, edge_weight
