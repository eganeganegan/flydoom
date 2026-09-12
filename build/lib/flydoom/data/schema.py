"""Canonical Male CNS tables and sparse graph representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

NEURON_COLUMNS = (
    "body_id",
    "type",
    "class",
    "side",
    "neurotransmitter",
    "region",
    "pre_synapses",
    "post_synapses",
    "is_sensory",
    "is_motor",
    "is_descending",
    "x",
    "y",
    "z",
    "metadata",
)
EDGE_COLUMNS = (
    "pre_body_id",
    "post_body_id",
    "synapse_count",
    "confidence",
    "neurotransmitter",
    "region",
)

NEURON_ALIASES = {
    "neuron_id": "body_id",
    "bodyId": "body_id",
    "body": "body_id",
    "cell_type": "type",
    "super_class": "class",
    "hemisphere": "side",
    "brain_region": "region",
}
EDGE_ALIASES = {
    "pre_neuron_id": "pre_body_id",
    "post_neuron_id": "post_body_id",
    "bodyId_pre": "pre_body_id",
    "bodyId_post": "post_body_id",
    "body_pre": "pre_body_id",
    "body_post": "post_body_id",
    "weight": "synapse_count",
}


def _rename_missing(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    rename = {
        source: target
        for source, target in aliases.items()
        if source in frame.columns and target not in frame.columns
    }
    return frame.rename(columns=rename)


@dataclass(slots=True)
class ConnectomeGraph:
    """Validated canonical tables with stable contiguous PyTorch node indices."""

    neurons: pd.DataFrame
    edges: pd.DataFrame

    def __post_init__(self) -> None:
        self.neurons = _rename_missing(self.neurons.copy(), NEURON_ALIASES)
        self.edges = _rename_missing(self.edges.copy(), EDGE_ALIASES)
        self._fill_optional_columns()
        self.validate()

    def _fill_optional_columns(self) -> None:
        neuron_defaults: dict[str, Any] = {
            "type": "unknown",
            "class": "unknown",
            "side": "unknown",
            "neurotransmitter": "unknown",
            "region": "unknown",
            "pre_synapses": 0,
            "post_synapses": 0,
            "is_sensory": False,
            "is_motor": False,
            "is_descending": False,
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
            "metadata": "{}",
        }
        for column, value in neuron_defaults.items():
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
        """Reject ambiguous IDs, invalid endpoints, and invalid weights."""
        if "body_id" not in self.neurons:
            raise ValueError("Missing neuron column: body_id")
        required_edges = {"pre_body_id", "post_body_id"}
        if missing := required_edges.difference(self.edges.columns):
            raise ValueError(f"Missing edge columns: {sorted(missing)}")
        if self.neurons["body_id"].isna().any() or self.neurons["body_id"].duplicated().any():
            raise ValueError("body_id values must be non-null and unique")
        known = set(self.neurons["body_id"].tolist())
        endpoints = set(self.edges["pre_body_id"]) | set(self.edges["post_body_id"])
        if unknown := endpoints.difference(known):
            raise ValueError(f"Edges reference unknown body IDs: {list(unknown)[:5]}")
        weights = pd.to_numeric(self.edges["synapse_count"], errors="coerce")
        if weights.isna().any() or (weights <= 0).any():
            raise ValueError("synapse_count must contain finite positive values")

    @property
    def node_count(self) -> int:
        return len(self.neurons)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def id_to_index(self) -> dict[Any, int]:
        return {body_id: index for index, body_id in enumerate(self.neurons["body_id"])}

    def sparse_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return COO edge indices and values; never constructs an N×N matrix."""
        mapping = self.id_to_index
        sources = self.edges["pre_body_id"].map(mapping).to_numpy(dtype=np.int64)
        targets = self.edges["post_body_id"].map(mapping).to_numpy(dtype=np.int64)
        edge_index = torch.from_numpy(np.stack((sources, targets))).long()
        edge_weight = torch.from_numpy(
            self.edges["synapse_count"].to_numpy(dtype=np.float32, copy=True)
        )
        return edge_index, edge_weight

    def sparse_adjacency(self, layout: Literal["coo", "csr"] = "coo") -> torch.Tensor:
        """Return sparse adjacency A[post, pre] in COO or CSR layout."""
        edge_index, values = self.sparse_tensors()
        adjacency = torch.sparse_coo_tensor(
            edge_index.flip(0),
            values,
            (self.node_count, self.node_count),
            check_invariants=True,
        ).coalesce()
        if layout == "coo":
            return adjacency
        if layout == "csr":
            return adjacency.to_sparse_csr()
        raise ValueError(f"Unsupported sparse layout: {layout}")
