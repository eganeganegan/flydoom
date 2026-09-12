"""Generic tabular adapters and a deterministic synthetic smoke-test graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flydoom.data.schema import ConnectomeGraph


def read_table(path: str | Path, hdf_key: str | None = None) -> pd.DataFrame:
    """Read a supported connectome table without guessing its semantic schema."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".feather", ".arrow"}:
        return pd.read_feather(path)
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
    """Load generic tables, with explicit optional source-to-canonical mappings."""
    neurons = read_table(neurons_path, neuron_hdf_key).rename(columns=neuron_columns or {})
    edges = read_table(edges_path, edge_hdf_key).rename(columns=edge_columns or {})
    return ConnectomeGraph(neurons, edges)


def generate_mock_connectome(node_count: int = 1_000, seed: int = 0) -> ConnectomeGraph:
    """Generate a layered graph for tests only; it is never labeled biological data."""
    if node_count < 20:
        raise ValueError("Mock connectome requires at least 20 neurons")
    rng = np.random.default_rng(seed)
    ids = np.arange(10_000, 10_000 + node_count, dtype=np.int64)
    layer = np.minimum((np.arange(node_count) * 5) // node_count, 4)
    regions = np.array(["visual", "optic", "central", "navigation", "descending"])[layer]
    neurons = pd.DataFrame(
        {
            "body_id": ids,
            "type": np.char.add("mock_", regions),
            "class": regions,
            "neurotransmitter": rng.choice(["acetylcholine", "gaba", "glutamate"], node_count),
            "side": rng.choice(["LHS", "RHS", "Midline"], node_count),
            "region": regions,
            "is_sensory": layer == 0,
            "is_motor": layer == 4,
            "is_descending": layer == 4,
            "x": rng.normal(layer * 2.0, 0.6),
            "y": rng.normal(0.0, 1.0, node_count),
            "z": rng.normal(0.0, 1.0, node_count),
            "metadata": '{"synthetic": true}',
        }
    )
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for source_layer in range(5):
        source_nodes = np.flatnonzero(layer == source_layer)
        targets = np.flatnonzero(np.isin(layer, [source_layer, min(source_layer + 1, 4)]))
        source_parts.append(np.repeat(source_nodes, 6))
        target_parts.append(rng.choice(targets, len(source_nodes) * 6, replace=True))
    sources = np.concatenate(source_parts)
    targets = np.concatenate(target_parts)
    keep = sources != targets
    pairs = pd.DataFrame({"source": sources[keep], "target": targets[keep]}).drop_duplicates()
    edge_sources = pairs["source"].to_numpy()
    edges = pd.DataFrame(
        {
            "pre_body_id": ids[edge_sources],
            "post_body_id": ids[pairs["target"].to_numpy()],
            "synapse_count": rng.integers(1, 21, len(pairs)),
            "confidence": rng.uniform(0.8, 1.0, len(pairs)),
            "neurotransmitter": neurons.loc[edge_sources, "neurotransmitter"].to_numpy(),
        }
    )
    return ConnectomeGraph(neurons, edges)
