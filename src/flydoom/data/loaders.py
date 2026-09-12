"""Adapters from documented tabular exports into the canonical connectome schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flydoom.data.schema import ConnectomeGraph


def _read_table(path: Path, hdf_key: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".h5", ".hdf5"}:
        return pd.read_hdf(path, key=hdf_key)
    raise ValueError(f"Unsupported table format: {suffix}")


def load_connectome(
    neurons_path: str | Path,
    edges_path: str | Path,
    *,
    neuron_columns: dict[str, str] | None = None,
    edge_columns: dict[str, str] | None = None,
    neuron_hdf_key: str | None = None,
    edge_hdf_key: str | None = None,
) -> ConnectomeGraph:
    """Load tables, optionally mapping source-specific names to canonical columns."""
    neurons = _read_table(Path(neurons_path), neuron_hdf_key).rename(columns=neuron_columns or {})
    edges = _read_table(Path(edges_path), edge_hdf_key).rename(columns=edge_columns or {})
    return ConnectomeGraph(neurons, edges)


def generate_mock_connectome(node_count: int = 1_000, seed: int = 0) -> ConnectomeGraph:
    """Generate a reproducible layered graph through the same canonical interface."""
    if node_count < 20:
        raise ValueError("Mock connectome requires at least 20 neurons")
    rng = np.random.default_rng(seed)
    ids = np.arange(10_000, 10_000 + node_count, dtype=np.int64)
    layer = np.minimum((np.arange(node_count) * 5) // node_count, 4)
    regions = np.array(["visual", "optic", "central", "navigation", "descending"])[layer]
    neurons = pd.DataFrame(
        {
            "neuron_id": ids,
            "cell_type": np.char.add("mock_", regions),
            "super_class": regions,
            "neurotransmitter": rng.choice(["acetylcholine", "gaba", "glutamate"], node_count),
            "hemisphere": rng.choice(["left", "right", "midline"], node_count),
            "brain_region": regions,
            "is_sensory": layer == 0,
            "is_motor": layer == 4,
            "is_descending": layer == 4,
            "metadata": '{"synthetic": true}',
        }
    )
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for source_layer in range(5):
        source_nodes = np.flatnonzero(layer == source_layer)
        target_layers = [source_layer]
        if source_layer < 4:
            target_layers.append(source_layer + 1)
        targets = np.flatnonzero(np.isin(layer, target_layers))
        degree = 6
        source_parts.append(np.repeat(source_nodes, degree))
        target_parts.append(rng.choice(targets, len(source_nodes) * degree, replace=True))
    sources = np.concatenate(source_parts)
    targets = np.concatenate(target_parts)
    keep = sources != targets
    edge_pairs = pd.DataFrame({"source": sources[keep], "target": targets[keep]}).drop_duplicates()
    edges = pd.DataFrame(
        {
            "pre_neuron_id": ids[edge_pairs["source"].to_numpy()],
            "post_neuron_id": ids[edge_pairs["target"].to_numpy()],
            "synapse_count": rng.integers(1, 21, len(edge_pairs)),
            "confidence": rng.uniform(0.8, 1.0, len(edge_pairs)),
        }
    )
    return ConnectomeGraph(neurons, edges)
