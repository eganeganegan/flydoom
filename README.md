# FlyDoom

FlyDoom is a research prototype for testing whether a **Drosophila-connectome-derived topology** provides useful inductive biases for reinforcement learning compared with matched synthetic neural graphs. The first milestone is runnable: it generates a 1,000-neuron mock connectome, reloads it through the canonical data adapter, extracts a visual-to-descending subgraph, trains a sparse rate controller with PPO, trains an edge-count-matched Erdos-Renyi control, and writes comparable artifacts.

> This project does not emulate the subjective experience or complete biological dynamics of a fruit fly. It uses experimentally reconstructed neural connectivity as a computational graph.

It is not an exact brain simulation, a claim about consciousness, or evidence that synapse count is physiological synaptic strength. Real-connectome experiments require appropriately licensed public data and multiple-seed statistical analysis.

## Research question

Does the topology of a biological connectome provide useful inductive biases for reinforcement learning compared with matched synthetic neural graphs?

The planned hypotheses concern sample efficiency, degree-preserving rewiring, modularity, weight shuffling, and navigation-specific effects. They are hypotheses, not expected conclusions; negative findings are valid.

## Architecture

```text
RGB frame -> visual encoder -> sensory nodes -> sparse connectome recurrence
          -> descending nodes only -> 5-action policy + scalar value
```

The fixed edge index constrains every recurrent synapse. There is no dense adjacency matrix and no MLP bypass around the graph. The learned CNN or fly-inspired feature encoder maps only onto annotated sensory nodes. Action and value heads read only annotated descending nodes.

The required rate model computes

$$
h_i(t+1)=\tanh\left(\lambda h_i(t)+\sum_j W_{ji}h_j(t)+x_i(t)\right).
$$

Weights start from $\log(1+\text{synapse count})$ and may be normalized and trained. Synapse count is an anatomical proxy, not measured efficacy. Neurotransmitter sign mappings are configurable; unknown or context-dependent signs remain explicitly uncertain.

Two visual encoders are included:

- `SimpleCNNEncoder`: a small learned image baseline.
- `FlyInspiredVisualEncoder`: approximate luminance, temporal change, local contrast, directional edge, and looming features. These are engineering approximations, not identified retinal physiology.

## Installation

Python 3.12 or newer is required. A CUDA build of PyTorch is recommended for larger subgraphs but not required for tests.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For VizDoom and TensorBoard support:

```bash
python -m pip install -e '.[doom,tracking]'
```

The mock fallback requires no DOOM installation. It uses the same image, action, and Gymnasium interfaces as the VizDoom boundary.

## First milestone

Run the requested comparison:

```bash
python scripts/train.py \
  experiment=flydoom_basic \
  data=mock_connectome \
  model=connectome_rate \
  seed=42
```

A short CPU smoke run is:

```bash
python scripts/train.py experiment=flydoom_basic data=mock_connectome \
  model=connectome_rate seed=42 device=cpu data.node_count=100 \
  data.max_neurons=100 training.total_steps=128 training.rollout_steps=32 \
  training.epochs=1 env.max_episode_steps=32
```

Use `vizdoom=basic env.backend=vizdoom` after installing the DOOM extra. Configuration groups are YAML files under `configs/`; dotted `key=value` arguments override values. Enable TensorBoard with `output.tensorboard=true`, then run `tensorboard --logdir outputs`.

Each run records the resolved config, seed, git commit when available, dataset fingerprint, subgraph metadata, model checkpoints, training CSVs, deterministic evaluation JSON, a comparison CSV, and a learning-curve PNG:

```text
outputs/YYYY-MM-DD/fly_connectome_seed_42/
  config.yaml
  subgraph_metadata.json
  metrics.csv
  comparison.csv
  learning_curves.png
  fly_connectome/{checkpoint.pt,metrics.csv,evaluation.json}
  erdos_renyi/{checkpoint.pt,metrics.csv,evaluation.json}
```

Inspect evaluation artifacts with:

```bash
python scripts/evaluate.py outputs/YYYY-MM-DD/fly_connectome_seed_42
```

## Real connectome data

FlyDoom does not scrape or redistribute connectome exports. Obtain neuron and synapse tables through official FlyWire or publication access channels and comply with their terms. Run `python scripts/download_data.py` for placement instructions and see [data/README.md](data/README.md) for the canonical schema.

Supported adapters are CSV/TSV, parquet, edge-list CSV, and optional HDF5. Source-specific column names must be mapped explicitly rather than guessed.

Inspect a table pair without dense adjacency construction:

```bash
python scripts/inspect_connectome.py \
  --neurons data/raw/neurons.csv \
  --synapses data/raw/synapses.csv \
  --sample-edges 100000
```

Build a real subgraph:

```bash
python scripts/build_subgraph.py \
  --neurons data/raw/neurons.csv \
  --synapses data/raw/synapses.csv \
  --source-class visual \
  --target-class descending \
  --max-neurons 5000 \
  --min-synapses 3 \
  --output data/processed/visual_descending.pt
```

The builder reports selected neurons and edges, represented synapses, sensory/descending counts, a source fingerprint, and estimated graph tensor memory. Extraction supports class/region filters, synaptic thresholds, bounded forward hops, target reachability, and a hard neuron cap.

## Controls and ablations

Implemented graph controls include exact-N/E directed Erdos-Renyi, exact in/out-degree-preserving rewiring, and weight shuffling. Conventional MLP and GRU actor-critic policies are available for parameter-budget matching. Structural utilities remove selected nodes, high/low/random-degree populations, regions, or inter-region edges.

Ablation helper example:

```bash
python scripts/run_ablation.py --neurons data/raw/neurons.csv \
  --synapses data/raw/synapses.csv --remove-region central
```

Multi-seed experiment orchestration and the full real-connectome/rewired/ER/MLP benchmark remain subsequent milestones. Statistical helpers already provide bootstrap confidence intervals, Mann-Whitney comparisons, and rank-biserial effect sizes; conclusions should report uncertainty and effect sizes rather than p-values alone.

## Visualization

Training produces learning curves. Degree distributions can be rendered with:

```bash
python scripts/visualize_connectome.py --neurons data/raw/neurons.csv \
  --synapses data/raw/synapses.csv --output degree_distribution.png
```

Region activity heatmap support is included for episode instrumentation. Composite DOOM-frame/activity/action video export remains a stretch goal.

## Testing and quality

```bash
pytest
ruff check .
mypy src/flydoom
```

Tests cover canonical loading, contiguous ID mapping, sparse edges, source-to-target extraction, matched controls and degree preservation, both observation encoders, action/readout mapping, sparse forward gradients, PPO updates, deterministic evaluation, and checkpoint save/load.

## Scientific limitations

- Anatomical connectivity is not a complete dynamical brain model.
- Synapse count is not equivalent to physiological synaptic strength.
- Neurotransmitter-to-excitation/inhibition mapping may be receptor-, compartment-, and context-dependent.
- DOOM visual inputs and action outputs are artificial interfaces.
- The mock graph validates software, not the biological hypothesis.
- Fair claims require matched optimization budgets, environment interactions, parameter counts, and at least 5, preferably 10, preregistered seeds.
- Subgraph selection can introduce strong researcher degrees of freedom and must be recorded and justified.

Use terms such as **connectome-constrained neural controller**, **Drosophila-connectome-derived topology**, and **biological graph prior**. Do not describe the controller as a conscious fruit fly.
