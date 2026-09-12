#!/usr/bin/env python3
"""Create and summarize a structural lesion for subsequent matched training."""

import argparse
import json

from flydoom.data.loaders import load_connectome
from flydoom.experiments.ablations import remove_region

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", required=True)
    parser.add_argument("--synapses", required=True)
    parser.add_argument("--remove-region", required=True)
    args = parser.parse_args()
    original = load_connectome(args.neurons, args.synapses)
    ablated = remove_region(original, args.remove_region)
    print(
        json.dumps(
            {
                "neurons_before": original.node_count,
                "neurons_after": ablated.node_count,
                "edges_before": original.edge_count,
                "edges_after": ablated.edge_count,
            },
            indent=2,
        )
    )
