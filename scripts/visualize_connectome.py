#!/usr/bin/env python3
"""Render degree distributions from canonical connectome tables."""

import argparse

from flydoom.data.loaders import load_connectome
from flydoom.visualization.graph import plot_degree_distribution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", required=True)
    parser.add_argument("--synapses", required=True)
    parser.add_argument("--output", default="degree_distribution.png")
    args = parser.parse_args()
    plot_degree_distribution(load_connectome(args.neurons, args.synapses), args.output)
