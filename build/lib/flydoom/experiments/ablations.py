"""Reproducible structural lesions on canonical connectome graphs."""

from __future__ import annotations

import numpy as np

from flydoom.data.schema import ConnectomeGraph


def remove_nodes(graph: ConnectomeGraph, neuron_ids: set[object]) -> ConnectomeGraph:
    neurons = graph.neurons.loc[~graph.neurons.body_id.isin(neuron_ids)].reset_index(drop=True)
    edges = graph.edges.loc[
        ~graph.edges.pre_body_id.isin(neuron_ids) & ~graph.edges.post_body_id.isin(neuron_ids)
    ].reset_index(drop=True)
    return ConnectomeGraph(neurons, edges)


def remove_region(graph: ConnectomeGraph, region: str) -> ConnectomeGraph:
    ids = set(graph.neurons.loc[graph.neurons.region.eq(region), "body_id"])
    return remove_nodes(graph, ids)


def remove_by_degree(
    graph: ConnectomeGraph, fraction: float, *, highest: bool, seed: int | None = None
) -> ConnectomeGraph:
    degree = graph.edges.pre_body_id.value_counts().add(
        graph.edges.post_body_id.value_counts(), fill_value=0
    )
    count = max(1, round(graph.node_count * fraction))
    if seed is None:
        selected = degree.sort_values(ascending=not highest).head(count).index
    else:
        selected = np.random.default_rng(seed).choice(graph.neurons.body_id, count, replace=False)
    return remove_nodes(graph, set(selected))


def remove_inter_region_edges(graph: ConnectomeGraph) -> ConnectomeGraph:
    regions = graph.neurons.set_index("body_id").region
    edges = graph.edges.loc[
        graph.edges.pre_body_id.map(regions).eq(graph.edges.post_body_id.map(regions))
    ].reset_index(drop=True)
    return ConnectomeGraph(graph.neurons.copy(), edges)
