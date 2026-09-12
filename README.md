# FlyDoom

FlyDoom is a research framework for asking a deliberately neutral question: **does the topology of the real Drosophila male CNS connectome provide a useful inductive bias for reinforcement learning?** It uses the official HHMI Janelia Male CNS Connectome v1.0 as the fixed edge set of a sparse recurrent controller and compares it with matched random graphs and conventional neural-network baselines on simple VizDoom tasks.

This is not Doom running in a biological brain, a whole-brain simulation, or a claim about consciousness. Synapse count is an anatomical proxy rather than measured synaptic efficacy, and the activity visualization shows model state.

## What is implemented

- Official `male-cns:v1.0` neuPrint access and official bulk Feather ingestion.
- Canonical neuron/edge tables with contiguous PyTorch indexing and COO/CSR export.
- Type, class, region, body-ID, k-hop, edge-weight, shortest-path-union, and top-connectivity subgraph selection.
- Sparse leaky recurrence with no dense N × N adjacency and no MLP bypass.
- Fixed internal weights, trainable internal weights, or trainable-readout-only modes.
- Optional neurotransmitter sign constraints; unknown/context-dependent signs remain unconstrained.
- CNN and fly-inspired visual encoders.
- Scenario-native four-action VizDoom interfaces with no-op and shoot.
- PPO training, deterministic evaluation, seeding, checkpoints, CSV metrics, and comparison plots.
- topology, matched Erdős–Rényi, degree-preserving rewiring, MLP, GRU, and LSTM controls.
- Synchronized Doom-frame/model-activity/action/reward traces.
- A dark teal PyVista player with play/pause, scrubbing, speed, camera control, filtering, PNG, MP4, 16:9, and 9:16 output.
- SWC skeleton rendering when skeleton files are supplied; otherwise soma centroids or a 3D graph layout are used.

## Install

Python 3.12 or newer is required. On macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,all]'
```

For a smaller installation, the core package is `pip install -e '.[dev]'`; extras are `doom`, `data`, `viz`, and `tracking`. Install the CUDA build of PyTorch appropriate for the local driver before installing FlyDoom when using NVIDIA hardware. The intended initial subgraphs contain roughly 1,000–10,000 neurons and fit comfortably on a consumer GPU around 12 GB; start small and measure before scaling.

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1 -InstallVizDoom
```

## Official Male CNS v1.0 data

The only built-in biological source is the [HHMI Janelia Male CNS Connectome](https://www.janelia.org/project-team/flyem/male-cns-connectome), using its [official download page](https://male-cns.janelia.org/download/).

### Mode 1: neuPrint

1. Create an account at [neuprint.janelia.org](https://neuprint.janelia.org).
2. Sign in, open the account menu, and copy the API token.
3. Export it as `NEUPRINT_TOKEN`.

```bash
export NEUPRINT_TOKEN='paste-token-here'
python scripts/build_subgraph.py \
  --mode neuprint \
  --type 'LC4' \
  --target-class descending \
  --min-synapses 3 \
  --num-hops 4 \
  --max-neurons 5000 \
  --output data/processed/visual-descending.pt
```

The adapter constructs `Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=...)` and uses `neuprint-python` targeted adjacency queries. Never commit the token; `.env` files and raw data are ignored.

`LC4` is a confirmed visual projection-neuron type in the Male CNS v1.0 dataset. The
photoreceptor-style pattern `R1-6.*` does not match the dataset's `type` field.

### Mode 2: official bulk Feather files

The official bucket is:

```text
gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
```

List the exact sources, or stream the required files from their official HTTPS equivalents:

```bash
python scripts/download_data.py
python scripts/download_data.py --download required --output data/raw/male-cns-v1.0
```

Equivalent `gcloud` commands are:

```bash
gcloud storage cp gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather data/raw/male-cns-v1.0/
gcloud storage cp gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather data/raw/male-cns-v1.0/
gcloud storage cp gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/body-stats-male-cns-v1.0-minconf-0.5.feather data/raw/male-cns-v1.0/
gcloud storage cp gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather data/raw/male-cns-v1.0/
```

Required files:

- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`
- `body-stats-male-cns-v1.0-minconf-0.5.feather`
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

Optional richer files are `syn-points-male-cns-v1.0-minconf-0.5.feather`, `syn-partners-male-cns-v1.0-minconf-0.5.feather`, and `tbar-neurotransmitters-male-cns-v1.0.feather`. They are several additional gigabytes and are not needed for the first controller.

Build a bounded bulk subgraph:

```bash
python scripts/build_subgraph.py \
  --mode bulk \
  --bulk-dir data/raw/male-cns-v1.0 \
  --source-class sensory \
  --target-class descending \
  --min-synapses 3 \
  --num-hops 6 \
  --top-connected 5000 \
  --output data/processed/visual-descending.pt
```

The source files are not redistributed by FlyDoom. Male CNS v1.0 is identified by the official project as CC BY; derived artifacts should retain dataset attribution.

## Controller

For node state `h`, fixed connectome edge set `E`, learned or fixed edge values `w`, input drive `x`, and bias `b`, the controller evaluates:

```text
h_next = activation(leak * h + sparse_message_passing(h, E, w) + x + b)
```

Weights initialize from `log1p(synapse_count)`, optionally normalize by incoming magnitude, and can only exist on published subgraph edges. Message passing uses edge gathers plus `index_add`; `ConnectomeGraph.sparse_adjacency()` also exposes COO/CSR tensors for analysis. Configure:

```yaml
model:
  training_mode: trainable_internal  # fixed_internal | readout_only
  sign_constraints: false
  encoder: cnn                       # cnn | fly
```

The fly-inspired encoder returns pooled luminance, local contrast, temporal intensity change, directional motion/edge responses, looming, and edge magnitude. These are engineering features, not a retinal physiology model. Features drive designated sensory nodes; only designated descending nodes feed the policy/value heads.

## Train and evaluate

Run the install-free mock first:

```bash
python scripts/train.py \
  experiment=flydoom_basic data=mock_connectome model=connectome_rate \
  device=cpu seed=42 \
  data.node_count=100 data.max_neurons=100 \
  training.total_steps=128 training.rollout_steps=32 training.epochs=1 \
  env.max_episode_steps=32
```

Train a processed biological subgraph in the `basic` scenario:

```bash
python scripts/train.py \
  experiment=flydoom_basic data=processed model=connectome_rate vizdoom=basic \
  device=cuda seed=42 training.total_steps=1000000
```

The `basic` environment uses the scenario's native actions: no-op, move left, move
right, and shoot. `defend_the_center` instead uses no-op, turn left, turn right, and
shoot. Both VizDoom groups scale rewards by `0.01` during optimization while reporting
unscaled episode rewards.

Use `vizdoom=defend_the_center` for the alternate scenario. Configuration groups are ordinary YAML in `configs/`; dotted `key=value` arguments override them. By default one seed trains all required variants: `real_connectome`, `erdos_renyi`, `degree_rewired`, `mlp`, `gru`, and `lstm`. The connectome, GRU, and LSTM policies carry recurrent state between observations and reset it at episode boundaries. Run multiple seeds before drawing scientific conclusions.

Each run logs reward, episode length, kills, survival time, policy/value losses, entropy, gradient norm, approximate KL, clipping fraction, per-action frequencies, FPS, graph size, total/trainable parameter counts, parameter-change norm, paired before/after deterministic evaluations, reward AUC, resolved configuration, source fingerprint, final checkpoints, and a learning curve. A repeat with the same date and seed is written to a numbered run directory instead of overwriting the previous run. Inspect results with:

```bash
python scripts/evaluate.py outputs/YYYY-MM-DD/fly_connectome_seed_42
python scripts/evaluate.py outputs/YYYY-MM-DD/fly_connectome_seed_42 --rerun --episodes 10
```

## WebGL neural cinema

The primary activity viewer is a browser-based Three.js application. It avoids WSL/X11
rendering problems and adds GPU bloom, live activity shaders, orbit controls, a Doom
picture-in-picture feed, metrics, scrubbing, playback speed, activity gain, and
fullscreen mode. The pinned toolchain supports Node.js `18+`, including the Node
`18.19.1` release commonly installed by Ubuntu/WSL.

Export a recorded trace together with the exact graph used by the policy:

```bash
python scripts/export_web_viewer.py \
  outputs/.../real_connectome/episode_trace.npz \
  --graph data/processed/visual-descending.pt \
  --output viewer/public/session

cd viewer
npm install
npm run dev
```

Open `http://localhost:5173` in the Windows browser. Drag to orbit, scroll to zoom,
Space plays or pauses, Left/Right steps through frames, and the bottom controls adjust
speed, activity gain, auto-orbit, morphology, and fullscreen. The exporter writes
compact typed-array binaries rather than embedding a large trace in JSON.

Without skeleton files, the viewer renders the real graph layout and connectome edges.
For anatomical neurites like the reference visualization, add the official SWC folder:

```bash
python scripts/export_web_viewer.py \
  outputs/.../real_connectome/episode_trace.npz \
  --graph data/processed/visual-descending.pt \
  --skeleton-dir data/raw/skeletons-swc \
  --output viewer/public/session
```

Run `npm run build` to create a deployable static build in `viewer/dist/`.

## Desktop activity player and export

Training records `real_connectome/episode_trace.npz`. Open it with:

```bash
python scripts/render_activity.py outputs/.../real_connectome/episode_trace.npz
```

Controls: Space plays/pauses; Left/Right scrubs; `[`/`]` changes playback speed; drag or `c` rotates the 3D camera; the slider seeks; `r`/`t` cycles region/type filters; and `s` exports the current window. Initial filters are also available with `--region` and `--type`.

Export video:

```bash
python scripts/export_demo_video.py trace.npz demo-16x9.mp4 --aspect 16:9
python scripts/export_demo_video.py trace.npz demo-9x16.mp4 --aspect 9:16
python scripts/export_demo_video.py trace.npz still.png --frame 20 --aspect 16:9
```

To render true neuron morphology, download only the desired SWCs from the official skeleton prefix and pass the directory:

```text
gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/
```

```bash
python scripts/render_activity.py trace.npz --skeleton-dir data/raw/skeletons-swc
python scripts/export_demo_video.py trace.npz demo.mp4 --skeleton-dir data/raw/skeletons-swc
```

Files must be named `{body_id}.swc`, as in the official release. Missing skeletons are skipped; if none match, the player falls back to node centroids/embedding.

## Quality checks

```bash
pytest
ruff check .
```

Tests cover canonical/Feather loading, ID mapping, COO/CSR sparsity, subgraph paths, matched graph controls, degree preservation, observation/action mapping, constrained sparse gradients, PPO updates, deterministic checkpoint evaluation, and trace serialization.

## Scientific interpretation

Treat topology choice, neuron-selection rules, weight threshold, signs, parameter budget, environment interactions, optimization budget, and seeds as controlled experimental variables. Report effect sizes and uncertainty. A null or negative result is a valid answer; this project does not assume that biological topology helps.
