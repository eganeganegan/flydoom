"""Topology-matched control graphs for scientific comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flydoom.data.schema import ConnectomeGraph


def erdos_renyi_matched(graph: ConnectomeGraph, seed: int) -> ConnectomeGraph:
    """Create a loop-free directed random graph with exactly the same N and E."""
    node_count = graph.node_count
    edge_count = graph.edge_count
    possible = node_count * (node_count - 1)
    if edge_count > possible:
        raise ValueError("Too many edges for a simple loop-free matched graph")
    rng = np.random.default_rng(seed)
    flat = rng.choice(possible, size=edge_count, replace=False)
    sources = flat // (node_count - 1)
    targets = flat % (node_count - 1)
    targets += targets >= sources
    ids = graph.neurons["neuron_id"].to_numpy()
    weights = rng.permutation(graph.edges["synapse_count"].to_numpy())
    edges = pd.DataFrame(
        {
            "pre_neuron_id": ids[sources],
            "post_neuron_id": ids[targets],
            "synapse_count": weights,
            "confidence": 1.0,
            "region": "synthetic_control",
        }
    )
    return ConnectomeGraph(graph.neurons.copy(), edges)


def shuffle_weights(graph: ConnectomeGraph, seed: int) -> ConnectomeGraph:
    """Keep topology fixed while permuting anatomical synapse counts."""
    edges = graph.edges.copy()
    edges["synapse_count"] = np.random.default_rng(seed).permutation(edges["synapse_count"])
    return ConnectomeGraph(graph.neurons.copy(), edges)


def degree_preserving_rewire(
    graph: ConnectomeGraph, seed: int, swaps_per_edge: int = 5
) -> ConnectomeGraph:
    """Randomize directed edges while preserving every node's in/out degree exactly."""
    rng = np.random.default_rng(seed)
    pairs = list(
        graph.edges[["pre_neuron_id", "post_neuron_id"]].itertuples(index=False, name=None)
    )
    edge_set = set(pairs)
    if len(edge_set) != len(pairs):
        raise ValueError("Degree-preserving rewiring requires unique directed edges")
    if len(pairs) < 2:
        return ConnectomeGraph(graph.neurons.copy(), graph.edges.copy())
    successful = 0
    attempts = 0
    target_swaps = swaps_per_edge * len(pairs)
    max_attempts = max(100, target_swaps * 20)
    while successful < target_swaps and attempts < max_attempts:
        first, second = rng.choice(len(pairs), size=2, replace=False)
        source_a, target_a = pairs[first]
        source_b, target_b = pairs[second]
        proposed_a = (source_a, target_b)
        proposed_b = (source_b, target_a)
        attempts += 1
        if source_a == target_b or source_b == target_a:
            continue
        if proposed_a in edge_set or proposed_b in edge_set or proposed_a == proposed_b:
            continue
        edge_set.remove(pairs[first])
        edge_set.remove(pairs[second])
        pairs[first], pairs[second] = proposed_a, proposed_b
        edge_set.add(proposed_a)
        edge_set.add(proposed_b)
        successful += 1
    edges = graph.edges.copy()
    edges["pre_neuron_id"] = [pair[0] for pair in pairs]
    edges["post_neuron_id"] = [pair[1] for pair in pairs]
    return ConnectomeGraph(graph.neurons.copy(), edges)
