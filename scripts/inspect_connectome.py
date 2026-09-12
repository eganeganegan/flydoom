#!/usr/bin/env python3
"""Inspect a filtered connectome without constructing a dense adjacency matrix."""

from __future__ import annotations

import argparse
import json

import networkx as nx

from flydoom.data.loaders import load_connectome

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", required=True)
    parser.add_argument("--synapses", required=True)
    parser.add_argument("--sample-edges", type=int, default=100000)
    args = parser.parse_args()
    connectome = load_connectome(args.neurons, args.synapses)
    sampled = connectome.edges.nlargest(args.sample_edges, "synapse_count")
    graph = nx.from_pandas_edgelist(
        sampled, "pre_neuron_id", "post_neuron_id", "synapse_count", create_using=nx.DiGraph
    )
    summary = {
        "neuron_count": connectome.node_count,
        "edge_count": connectome.edge_count,
        "synapse_count": float(connectome.edges.synapse_count.sum()),
        "sampled_strong_components": nx.number_strongly_connected_components(graph),
        "sampled_weak_components": nx.number_weakly_connected_components(graph),
        "cell_types": connectome.neurons.cell_type.value_counts().head(20).to_dict(),
        "regions": connectome.neurons.brain_region.value_counts().head(20).to_dict(),
        "neurotransmitters": connectome.neurons.neurotransmitter.value_counts().to_dict(),
    }
    print(json.dumps(summary, indent=2))
