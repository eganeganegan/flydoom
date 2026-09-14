#!/usr/bin/env python3
"""Build and save a sparse Male CNS or generic tabular subgraph."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from flydoom.data.loaders import load_connectome
from flydoom.data.male_cns import (
    BULK_ROOT,
    MALE_CNS_DOWNLOAD_URL,
    MALE_CNS_LICENSE,
    MALE_CNS_LICENSE_URL,
    MALE_CNS_PAPER_DOI,
    MALE_CNS_PROJECT_URL,
    NEUPRINT_DATASET,
    NEUPRINT_PAPER_DOI,
    NEUPRINT_SERVER,
    load_male_cns_bulk,
    load_male_cns_neuprint,
)
from flydoom.data.subgraph import SubgraphSpec, extract_subgraph


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("bulk", "neuprint", "generic"), default="bulk")
    result.add_argument("--bulk-dir", default="data/raw/male-cns-v1.0")
    result.add_argument("--neurons")
    result.add_argument("--edges")
    result.add_argument("--source-class")
    result.add_argument("--target-class")
    result.add_argument("--source-region")
    result.add_argument("--target-region")
    result.add_argument("--type", action="append", default=[])
    result.add_argument("--class", dest="classes", action="append", default=[])
    result.add_argument("--region", action="append", default=[])
    result.add_argument("--body-id", action="append", type=int, default=[])
    result.add_argument("--max-neurons", type=int, default=5000)
    result.add_argument("--top-connected", type=int)
    result.add_argument("--min-synapses", type=float, default=3)
    result.add_argument("--num-hops", type=int, default=4)
    result.add_argument("--no-path-pruning", action="store_true")
    result.add_argument("--output", required=True)
    return result


def _load(args: argparse.Namespace):
    if args.mode == "bulk":
        return load_male_cns_bulk(args.bulk_dir)
    if args.mode == "neuprint":
        return load_male_cns_neuprint(
            types=args.type or None,
            classes=args.classes or None,
            rois=args.region or None,
            body_ids=args.body_id or None,
            min_weight=round(args.min_synapses),
            max_neurons=args.max_neurons,
            num_hops=args.num_hops,
        )
    if not args.neurons or not args.edges:
        raise SystemExit("--mode generic requires --neurons and --edges")
    return load_connectome(args.neurons, args.edges)


def main() -> None:
    args = parser().parse_args()
    graph = _load(args)
    selected = extract_subgraph(
        graph,
        SubgraphSpec(
            source_class=args.source_class,
            target_class=args.target_class,
            source_region=args.source_region,
            target_region=args.target_region,
            neuron_types=tuple(args.type) if args.mode != "neuprint" else (),
            neuron_classes=tuple(args.classes) if args.mode != "neuprint" else (),
            regions=tuple(args.region) if args.mode != "neuprint" else (),
            seed_body_ids=tuple(args.body_id),
            min_synapses=args.min_synapses,
            max_neurons=args.max_neurons,
            num_hops=args.num_hops,
            shortest_path_union=not args.no_path_pruning,
            top_connected=args.top_connected,
        ),
    )
    edge_index, edge_weight = selected.sparse_tensors()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = list(filter(None, (args.neurons, args.edges)))
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(Path(path).read_bytes())
    biological_source = args.mode in {"bulk", "neuprint"}
    metadata = {
        "source_mode": args.mode,
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset_name": "Male CNS Connectome" if biological_source else "user-supplied",
        "dataset_identifier": NEUPRINT_DATASET if biological_source else None,
        "dataset_version": "v1.0" if biological_source else None,
        "dataset_url": MALE_CNS_PROJECT_URL if biological_source else None,
        "dataset_download_url": MALE_CNS_DOWNLOAD_URL if biological_source else None,
        "dataset_license": MALE_CNS_LICENSE if biological_source else None,
        "dataset_license_url": MALE_CNS_LICENSE_URL if biological_source else None,
        "dataset_citation_doi": MALE_CNS_PAPER_DOI if biological_source else None,
        "neuprint_citation_doi": NEUPRINT_PAPER_DOI if args.mode == "neuprint" else None,
        "access_endpoint": (
            f"{NEUPRINT_SERVER} ({NEUPRINT_DATASET})"
            if args.mode == "neuprint"
            else BULK_ROOT if args.mode == "bulk" else None
        ),
        "selection": {
            "types": args.type,
            "classes": args.classes,
            "regions": args.region,
            "body_ids": args.body_id,
            "source_class": args.source_class,
            "target_class": args.target_class,
            "source_region": args.source_region,
            "target_region": args.target_region,
            "min_synapses": args.min_synapses,
            "num_hops": args.num_hops,
            "max_neurons": args.max_neurons,
            "top_connected": args.top_connected,
            "shortest_path_union": not args.no_path_pruning,
        },
        "selected_neurons": selected.node_count,
        "selected_edges": selected.edge_count,
        "total_synapses": float(selected.edges["synapse_count"].sum()),
        "sensory_nodes": int(selected.neurons["is_sensory"].sum()),
        "descending_nodes": int(selected.neurons["is_descending"].sum()),
        "source_files": source_paths,
        "source_sha256": digest.hexdigest() if source_paths else None,
        "sparse_layout": "coo",
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
    print(json.dumps(metadata, indent=2))
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
