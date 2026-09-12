#!/usr/bin/env python3
"""Build a sparse sensory-to-descending subgraph from canonical tabular data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from flydoom.data.loaders import load_connectome
from flydoom.data.subgraph import SubgraphSpec, extract_subgraph


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--neurons", required=True)
    result.add_argument("--synapses", required=True)
    result.add_argument("--source-class")
    result.add_argument("--target-class")
    result.add_argument("--source-region")
    result.add_argument("--target-region")
    result.add_argument("--max-neurons", type=int, default=5000)
    result.add_argument("--min-synapses", type=float, default=3)
    result.add_argument("--num-hops", type=int, default=8)
    result.add_argument("--output", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    graph = load_connectome(args.neurons, args.synapses)
    selected = extract_subgraph(
        graph,
        SubgraphSpec(
            source_class=args.source_class,
            target_class=args.target_class,
            source_region=args.source_region,
            target_region=args.target_region,
            min_synapses=args.min_synapses,
            max_neurons=args.max_neurons,
            num_hops=args.num_hops,
        ),
    )
    edge_index, edge_weight = selected.sparse_tensors()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "selected_neurons": selected.node_count,
        "selected_edges": selected.edge_count,
        "total_synapses": float(selected.edges["synapse_count"].sum()),
        "sensory_nodes": int(selected.neurons["is_sensory"].sum()),
        "descending_nodes": int(selected.neurons["is_descending"].sum()),
        "source_sha256": hashlib.sha256(
            Path(args.neurons).read_bytes() + Path(args.synapses).read_bytes()
        ).hexdigest(),
    }
    torch.save(
        {
            "neurons": selected.neurons.to_dict(orient="list"),
            "edges": selected.edges.to_dict(orient="list"),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "metadata": metadata,
        },
        output,
    )
    estimated_mb = (selected.edge_count * 20 + selected.node_count * 4) / (1024**2)
    print(f"Selected neurons: {selected.node_count:,}")
    print(f"Selected edges: {selected.edge_count:,}")
    print(f"Total synapses represented: {metadata['total_synapses']:,.0f}")
    print(f"Sensory nodes: {metadata['sensory_nodes']:,}")
    print(f"Descending nodes: {metadata['descending_nodes']:,}")
    print(f"Estimated graph tensor memory: {estimated_mb:.1f} MB")
    (output.with_suffix(output.suffix + ".json")).write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
