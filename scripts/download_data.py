#!/usr/bin/env python3
"""Explain safe acquisition of public connectome exports."""

from pathlib import Path

MESSAGE = """FlyDoom does not scrape or redistribute connectome data.
Download neuron and synapse tables through the official access mechanism for
FlyWire (https://flywire.ai/) or the relevant published dataset, accepting its
license and terms. Place exports at data/raw/neurons.csv and
data/raw/synapses.csv, then map source-specific columns in the loader if needed.
Required IDs: neuron_id, pre_neuron_id, post_neuron_id. See data/README.md.
"""

if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    print(MESSAGE)
