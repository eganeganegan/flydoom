"""Memory-bounded filtering, k-hop expansion, and shortest-path extraction."""

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
    neuron_types: tuple[str, ...] = ()
    neuron_classes: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    seed_body_ids: tuple[int, ...] = ()
    min_synapses: float = 1.0
    max_neurons: int = 5_000
    num_hops: int = 4
    shortest_path_union: bool = True
    top_connected: int | None = None


def _matches(neurons: pd.DataFrame, *, kind: str | None, region: str | None) -> pd.Series:
    mask = pd.Series(True, index=neurons.index)
    if kind:
        mask &= neurons["class"].fillna("").astype(str).str.contains(
            kind, case=False, regex=True
        ) | neurons["type"].fillna("").astype(str).str.contains(kind, case=False, regex=True)
    if region:
        mask &= (
            neurons["region"].fillna("").astype(str).str.contains(region, case=False, regex=True)
        )
    return mask


def extract_subgraph(graph: ConnectomeGraph, spec: SubgraphSpec) -> ConnectomeGraph:
    """Extract only real edges, with bounded forward expansion and path pruning."""
    if spec.max_neurons < 1 or spec.num_hops < 0:
        raise ValueError("max_neurons must be positive and num_hops non-negative")
    neurons = graph.neurons
    edges = graph.edges.loc[graph.edges["synapse_count"] >= spec.min_synapses].copy()
    eligible = pd.Series(True, index=neurons.index)
    if spec.neuron_types:
        eligible &= neurons["type"].isin(spec.neuron_types)
    if spec.neuron_classes:
        eligible &= neurons["class"].isin(spec.neuron_classes)
    if spec.regions:
        eligible &= neurons["region"].isin(spec.regions)

    source_mask = _matches(neurons, kind=spec.source_class, region=spec.source_region) & eligible
    if not spec.source_class and not spec.source_region and neurons["is_sensory"].any():
        source_mask &= neurons["is_sensory"].astype(bool)
    sources = set(neurons.loc[source_mask, "body_id"])
    sources.update(spec.seed_body_ids)
    if not sources:
        raise ValueError("No source neurons matched the extraction specification")

    target_mask = _matches(neurons, kind=spec.target_class, region=spec.target_region) & eligible
    targets = set(neurons.loc[target_mask, "body_id"])
    target_filters_active = bool(spec.target_class or spec.target_region)

    allowed = set(neurons.loc[eligible, "body_id"]) | set(sources) | set(targets)
    edges = edges.loc[edges["pre_body_id"].isin(allowed) & edges["post_body_id"].isin(allowed)]
    adjacency: dict[object, list[object]] = {}
    reverse: dict[object, list[object]] = {}
    for pre, post in edges[["pre_body_id", "post_body_id"]].itertuples(index=False):
        adjacency.setdefault(pre, []).append(post)
        reverse.setdefault(post, []).append(pre)

    reached = set(sources)
    frontier = set(sources)
    for _ in range(spec.num_hops):
        frontier = {node for current in frontier for node in adjacency.get(current, ())} - reached
        reached.update(frontier)
        if not frontier:
            break

    if target_filters_active and spec.shortest_path_union:
        reachable_targets = reached & targets
        keep = set(reachable_targets)
        distance = {node: 0 for node in sources}
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            if distance[node] >= spec.num_hops:
                continue
            for successor in adjacency.get(node, ()):
                if successor in reached and successor not in distance:
                    distance[successor] = distance[node] + 1
                    queue.append(successor)
        queue = deque(reachable_targets)
        while queue:
            node = queue.popleft()
            for predecessor in reverse.get(node, ()):
                if distance.get(predecessor) == distance.get(node, 0) - 1:
                    if predecessor not in keep:
                        keep.add(predecessor)
                        queue.append(predecessor)
        reached = keep
    if not reached:
        raise ValueError("No source-to-target subgraph was found")

    cap = min(spec.max_neurons, spec.top_connected or spec.max_neurons)
    if len(reached) > cap:
        internal = edges.loc[
            edges["pre_body_id"].isin(reached) & edges["post_body_id"].isin(reached)
        ]
        strength = (
            pd.concat(
                [
                    internal.groupby("pre_body_id")["synapse_count"].sum(),
                    internal.groupby("post_body_id")["synapse_count"].sum(),
                ],
                axis=1,
            )
            .fillna(0)
            .sum(axis=1)
            .sort_values(ascending=False)
        )
        mandatory = list((sources | (targets & reached)) & reached)
        if len(mandatory) > cap:
            mandatory = sorted(mandatory, key=lambda node: strength.get(node, 0), reverse=True)[
                :cap
            ]
        ranked = [node for node in strength.index if node not in mandatory]
        reached = set((mandatory + ranked)[:cap])

    selected_neurons = neurons.loc[neurons["body_id"].isin(reached)].reset_index(drop=True)
    selected_edges = edges.loc[
        edges["pre_body_id"].isin(reached) & edges["post_body_id"].isin(reached)
    ].reset_index(drop=True)
    return ConnectomeGraph(selected_neurons, selected_edges)
