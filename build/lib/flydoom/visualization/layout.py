"""Real centroids, SWC skeleton loading, and topology-based 3D fallbacks."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from flydoom.data.schema import ConnectomeGraph


def graph_positions(graph: ConnectomeGraph, seed: int = 0) -> np.ndarray:
    coordinates = graph.neurons[["x", "y", "z"]].to_numpy(dtype=np.float32)
    if np.isfinite(coordinates).all():
        return _normalise(coordinates)
    network = nx.DiGraph()
    network.add_nodes_from(range(graph.node_count))
    edge_index, _ = graph.sparse_tensors()
    network.add_edges_from(edge_index.T.tolist())
    iterations = 40 if graph.node_count <= 2_000 else 15
    layout = nx.spring_layout(network, dim=3, seed=seed, iterations=iterations)
    return _normalise(np.asarray([layout[index] for index in range(graph.node_count)]))


def _normalise(positions: np.ndarray) -> np.ndarray:
    centered = positions - np.nanmean(positions, axis=0, keepdims=True)
    scale = np.nanpercentile(np.linalg.norm(centered, axis=1), 95)
    return np.nan_to_num(centered / max(float(scale), 1e-6)).astype(np.float32)


def load_swc(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an SWC file as points plus zero-based line endpoint pairs."""
    table = np.loadtxt(path, comments="#", ndmin=2)
    if table.shape[1] < 7:
        raise ValueError(f"Invalid SWC file: {path}")
    node_ids = table[:, 0].astype(int)
    parents = table[:, 6].astype(int)
    lookup = {node_id: index for index, node_id in enumerate(node_ids)}
    lines = np.asarray(
        [(lookup[parent], index) for index, parent in enumerate(parents) if parent in lookup],
        dtype=np.int64,
    ).reshape(-1, 2)
    return table[:, 2:5].astype(np.float32), lines
