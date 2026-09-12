"""Memory-bounded connectome subgraph extraction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from flydoom.data.schema import ConnectomeGraph


@dataclass(frozen=True, slots=True)
class SubgraphSpec:
    source_class: str | None = None
    target_class: str | None = None
    source_region: str | None = None
    target_region: str | None = None
    min_synapses: float = 1.0
    max_neurons: int = 5_000
    num_hops: int = 4


def extract_subgraph(graph: ConnectomeGraph, spec: SubgraphSpec) -> ConnectomeGraph:
    """Expand forward from sources, retaining target-reachable nodes when specified."""
    neurons = graph.neurons
    edges = graph.edges.loc[graph.edges["synapse_count"] >= spec.min_synapses].copy()
    source_mask = pd.Series(True, index=neurons.index)
    if spec.source_class:
        source_mask &= neurons["super_class"].eq(spec.source_class) | neurons["cell_type"].eq(
            spec.source_class
        )
    if spec.source_region:
        source_mask &= neurons["brain_region"].eq(spec.source_region)
    sources = set(neurons.loc[source_mask, "neuron_id"])
    if not sources:
        raise ValueError("No source neurons matched the extraction specification")

    target_mask = pd.Series(True, index=neurons.index)
    if spec.target_class:
        target_mask &= neurons["super_class"].eq(spec.target_class) | neurons["cell_type"].eq(
            spec.target_class
        )
    if spec.target_region:
        target_mask &= neurons["brain_region"].eq(spec.target_region)
    targets = set(neurons.loc[target_mask, "neuron_id"])

    adjacency: dict[object, list[object]] = {}
    reverse: dict[object, list[object]] = {}
    for pre, post in edges[["pre_neuron_id", "post_neuron_id"]].itertuples(index=False):
        adjacency.setdefault(pre, []).append(post)
        reverse.setdefault(post, []).append(pre)

    reached = set(sources)
    frontier = set(sources)
    for _ in range(spec.num_hops):
        frontier = {node for current in frontier for node in adjacency.get(current, ())} - reached
        reached.update(frontier)
        if not frontier or len(reached) >= spec.max_neurons:
            break

    target_filters_active = bool(spec.target_class or spec.target_region)
    if target_filters_active:
        reachable_targets = reached & targets
        keep = set(reachable_targets)
        queue = deque(reachable_targets)
        while queue:
            node = queue.popleft()
            for predecessor in reverse.get(node, ()):
                if predecessor in reached and predecessor not in keep:
                    keep.add(predecessor)
                    queue.append(predecessor)
        reached = keep
    if not reached:
        raise ValueError("No source-to-target subgraph was found")

    if len(reached) > spec.max_neurons:
        ranked = (
            edges.loc[edges["pre_neuron_id"].isin(reached) & edges["post_neuron_id"].isin(reached)]
            .groupby("post_neuron_id")["synapse_count"]
            .sum()
            .sort_values(ascending=False)
        )
        mandatory = list(sources | (targets & reached))[: spec.max_neurons]
        remainder = [node for node in ranked.index if node not in mandatory]
        reached = set((mandatory + remainder)[: spec.max_neurons])

    selected_neurons = neurons.loc[neurons["neuron_id"].isin(reached)].reset_index(drop=True)
    selected_edges = edges.loc[
        edges["pre_neuron_id"].isin(reached) & edges["post_neuron_id"].isin(reached)
    ].reset_index(drop=True)
    return ConnectomeGraph(selected_neurons, selected_edges)
