# Data layout and canonical schema

Raw downloads belong under `data/raw/` and are git-ignored. Processed bounded subgraphs belong under `data/processed/`. The official Male CNS filenames and acquisition commands are documented in the repository README.

The canonical neuron table always contains these fields where the source provides them:

| Field | Meaning |
|---|---|
| `body_id` | Stable Male CNS segment/body ID |
| `type`, `class`, `side` | Curated annotations |
| `neurotransmitter` | Aggregate predicted/consensus transmitter |
| `region` | Region or neuromere summary |
| `pre_synapses`, `post_synapses` | Body-level synapse statistics |
| `is_sensory`, `is_motor`, `is_descending` | Input/readout population flags |
| `x`, `y`, `z` | Soma/centroid coordinates when available |
| `metadata` | Source-specific metadata placeholder |

The canonical edge table contains `pre_body_id`, `post_body_id`, `synapse_count`, `confidence`, `neurotransmitter`, and `region`. Required identifiers and weights are validated; body IDs map to contiguous indices only after extraction. Missing optional fields remain explicitly `unknown`, false, zero, or NaN rather than being fabricated.

Processed `.pt` files contain the canonical tables, COO `edge_index`, edge weights, and provenance/selection metadata. They do not contain dense adjacency matrices.
