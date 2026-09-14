# Data layout and canonical schema

Raw downloads belong under `data/raw/` and are git-ignored. Processed bounded subgraphs belong under `data/processed/`. The official Male CNS filenames and acquisition commands are documented in the [repository README](../README.md#official-male-cns-v10-data).

## Source and attribution

All built-in biological data are derived from the **Male CNS Connectome v1.0**
(`male-cns:v1.0`, released June 8, 2026), produced by FlyEM at HHMI Janelia with the
University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research.
The source dataset is CC BY 4.0. Cite both the
[dataset](../README.md#ref-malecns-data) and its
[associated paper](../README.md#ref-malecns-paper); also cite
[neuPrint](../README.md#ref-neuprint) when Mode 1 was used.

| Local/source artifact | Upstream meaning | Required attribution |
|---|---|---|
| `body-annotations-*.feather` | Curated body, type, class, side, and anatomical annotations | MaleCNS v1.0 dataset and paper |
| `body-neurotransmitters-*.feather` | Aggregate predicted/consensus transmitter labels | MaleCNS v1.0 dataset and paper; labels are predictions, not direct physiological measurements |
| `body-stats-*.feather` | Body-level pre/postsynapse summary statistics | MaleCNS v1.0 dataset and paper |
| `connectome-weights-*.feather` | Directed body-to-body edge weights measured as synapse counts | MaleCNS v1.0 dataset and paper; counts are not synaptic efficacies |
| `syn-points-*`, `syn-partners-*` | Optional synapse coordinates and partner relationships | MaleCNS v1.0 dataset and paper |
| `tbar-neurotransmitters-*` | Optional per-presynapse transmitter probabilities | MaleCNS v1.0 dataset and paper |
| `skeletons-swc/*.swc` | Centerline morphology in Male CNS EM coordinates | MaleCNS v1.0 dataset and paper |
| neuPrint query results | A selected view of `male-cns:v1.0` | MaleCNS v1.0 dataset, paper, and neuPrint paper |

Official source and license: <https://male-cns.janelia.org/download/>. Copy-ready
citations are provided in [`REFERENCES.bib`](../REFERENCES.bib).

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

When sharing a processed graph, keep its `.pt.json` sidecar and report the upstream
dataset version, access mode, query/selection arguments, minimum-synapse threshold,
hop count, neuron cap, source hash when available, and creation date. A processed
subgraph remains a derivative of the CC BY 4.0 MaleCNS data and must retain attribution.
