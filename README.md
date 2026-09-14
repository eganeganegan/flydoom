# FlyDoom

FlyDoom is a research framework for asking a deliberately neutral question: **does the topology of the real Drosophila male CNS connectome provide a useful inductive bias for reinforcement learning?** It uses the official HHMI Janelia Male CNS Connectome v1.0 ([MaleCNS consortium, 2026](#ref-malecns-data); [Berg et al., 2026](#ref-malecns-paper)) as the fixed edge set of a sparse recurrent controller and compares it with matched random graphs and conventional neural-network baselines on [VizDoom](#ref-vizdoom) tasks.

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
- Gradient-free three-factor learning with local eligibility traces, dopamine prediction
  error, homeostasis, weight bounds, and optional KC→MBON-only plasticity.
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

The only built-in biological source is the [HHMI Janelia Male CNS Connectome](https://male-cns.janelia.org/), release `male-cns:v1.0` dated June 8, 2026. The data, annotations, aggregate neurotransmitter predictions, connection-weight tables, and skeletons come from the [official download page](https://male-cns.janelia.org/download/) and should be cited as the [MaleCNS v1.0 dataset](#ref-malecns-data) together with the associated [MaleCNS paper](#ref-malecns-paper).

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

The adapter constructs `Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=...)` and uses `neuprint-python` targeted adjacency queries. Cite the [neuPrint platform paper](#ref-neuprint) in work that uses this access path. Never commit the token; `.env` files and raw data are ignored.

`LC4` is a confirmed visual projection-neuron type in the [Male CNS Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/LC4.html). The
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

The source files are not redistributed by FlyDoom. Male CNS v1.0 is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); derived artifacts and figures must retain dataset attribution. The dataset license does not automatically license FlyDoom's source code.

## Controller

For node state `h`, fixed connectome edge set `E`, learned or fixed edge values `w`, input drive `x`, and bias `b`, the controller evaluates:

```text
h_next = activation(leak * h + sparse_message_passing(h, E, w) + x + b)
```

Weights initialize from `log1p(synapse_count)`, optionally normalize by incoming magnitude, and can only exist on published subgraph edges. Message passing uses edge gathers plus `index_add`; `ConnectomeGraph.sparse_adjacency()` also exposes COO/CSR tensors for analysis. Configure:

```yaml
model:
  training_mode: trainable_internal  # fixed_internal | readout_only | three_factor
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
shoot. Both scenarios use an inspectable dopamine-like reward signal: living costs
`-0.01` per agent step, a hit is `+0.25`, a kill is `+10`, damage costs `-0.05` per
health point, death is `-5`, ammunition costs `-0.02`, and recovered health earns
`+0.02` per point. The scenario-native reward is disabled by default to avoid silently
double-counting these events. PPO receives the shaped signal multiplied by `0.1`, while
metrics and visualizations report the readable unscaled values. These are engineering
reward terms—a dopamine analogue—not a biological dopamine model.

Use `vizdoom=defend_the_center` for the alternate scenario. The environment is provided by [VizDoom](#ref-vizdoom) through the [Gymnasium](#ref-gymnasium) interface. Configuration groups are ordinary YAML in `configs/`; dotted `key=value` arguments override them. By default one seed trains all required variants: `real_connectome`, `erdos_renyi`, `degree_rewired`, `mlp`, `gru`, and `lstm`. The random-graph controls follow the [Erdos-Renyi model](#ref-erdos-renyi) and degree-preserving edge-swap null models ([Maslov and Sneppen, 2002](#ref-maslov-sneppen)). The connectome, GRU, and LSTM policies carry recurrent state between observations and reset it at episode boundaries. Run multiple seeds before drawing scientific conclusions.

Each run logs reward, every reward component, episode length, kills, hits, damage,
health, ammunition, deaths, survival time, policy/value losses, entropy, gradient norm,
approximate KL, clipping fraction, per-action frequencies, FPS, graph size,
total/trainable parameter counts, parameter-change norm, paired before/after
deterministic evaluations, success rate, reward AUC, resolved configuration, source
fingerprint, final checkpoints, and a learning curve. A repeat with the same date and
seed is written to a numbered run directory instead of overwriting the previous run.
Inspect results with:

```bash
python scripts/evaluate.py outputs/YYYY-MM-DD/fly_connectome_seed_42
python scripts/evaluate.py outputs/YYYY-MM-DD/fly_connectome_seed_42 --rerun --episodes 10
```

### Biological learning mode

`three_factor` mode performs no backpropagation and creates no optimizer. It is motivated by neo-Hebbian three-factor rules in which local pre/postsynaptic activity creates an eligibility trace and a later modulatory signal gates plasticity ([Frémaux and Gerstner, 2016](#ref-fremaux-gerstner); [Gerstner et al., 2018](#ref-gerstner-eligibility)). Every
connectome edge maintains a decaying eligibility trace from its local pre- and
postsynaptic rate activity. A temporal-difference reward prediction error acts as the
third, dopamine-like factor. Descending-neuron-to-action synapses use the same local
rule with action surprise, the value readout uses a local delta rule, and homeostatic
bias changes keep node activity bounded. The anatomical edge set never changes.

Run the gradient-free controller with:

```bash
python scripts/train.py \
  experiment=flydoom_biological data=processed \
  model=connectome_three_factor vizdoom=basic \
  device=cuda seed=42
```

The current preset permits local plasticity on every edge in the selected subgraph so
it can run on the existing visual-to-descending graph. If dopaminergic neurons are
present, the signed prediction error is also injected into their recurrent state. If
none are present, training emits a warning and broadcasts the modulatory factor directly
to eligible synapses.

For a graph containing Kenyon cells and mushroom-body output neurons, restrict
plasticity to annotated KC→MBON edges:

```bash
training.plasticity_scope=mushroom_body
```

The mushroom-body option is motivated by compartmental organization of Kenyon cells,
dopaminergic neurons, and mushroom-body output neurons in fly associative learning
([Aso et al., 2014a](#ref-aso-architecture); [Aso et al., 2014b](#ref-aso-valence)).
FlyDoom's scalar temporal-difference signal and update equations are nevertheless a
biologically inspired computational hypothesis, not a molecular simulation or a claim
that its reward coefficients reproduce dopamine concentrations. PPO
([Schulman et al., 2017](#ref-ppo)), with generalized advantage estimation
([Schulman et al., 2016](#ref-gae)), remains the performance control.

### How much training?

Independent runs do not teach one another: each command starts a freshly initialized
policy. For the small `basic` aiming task, begin with one focused 3-million-step run:

```bash
python scripts/train.py \
  experiment=flydoom_learn data=processed model=connectome_rate vizdoom=basic \
  device=cuda seed=42
```

The `flydoom_learn` preset trains only `real_connectome`; it does not spend the same
budget on all five controls. Treat 3 million steps as a starting estimate, not a
guarantee. Check `success_rate`, `mean_kills`, and the reward curve after the run. If
the curve is still rising, try 5 million steps. If it is flat near zero, more runs are
unlikely to repair the configuration—inspect action fractions and reward components
first. Use three seeds while tuning and five to ten seeds for a final comparison; the
seed runs measure reliability rather than accumulating learning.

The three-factor learner is expected to be less sample-efficient than PPO. Start with a
short `training.total_steps=100000` validation, then use the 3-million-step biological
preset. Treat the result as an experiment: extend it only while success rate is still
improving.

## WebGL neural cinema

The primary activity viewer is a browser-based Three.js instrument panel. It avoids
WSL/X11 rendering problems and combines a large Doom replay, health/kill/ammo telemetry,
per-action readout logits, a live reward breakdown, a visual-input map, a scrolling
rate-activity raster, and the orbitable 3D connectome. Positive reward is cyan, damage
and other negative reward are red, and strong firing remains orange. The pinned
toolchain supports Node.js `18+`, including the Node `18.19.1` release commonly
installed by Ubuntu/WSL.

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

Older traces still open, but their new health, ammunition, readout-logit, and reward
component fields appear blank. Train once with the current code to record full dashboard
telemetry, then export that new `episode_trace.npz`.

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

## Citation and attribution

If you publish results made with FlyDoom, cite the project using [`CITATION.cff`](CITATION.cff), cite the specific external resources used, and report the exact `male-cns:v1.0` selection parameters and generated dataset fingerprint. Copy-ready BibTeX for the references below is in [`REFERENCES.bib`](REFERENCES.bib). At minimum:

- Cite the MaleCNS dataset and paper for every real-connectome result.
- Also cite neuPrint when the graph was acquired through Mode 1.
- Cite VizDoom, PPO and GAE for PPO experiments.
- Cite the three-factor and Drosophila mushroom-body papers for biological-learning experiments.
- Cite the scientific-software stack—PyTorch, NumPy, pandas, SciPy, Matplotlib,
  Gymnasium, NetworkX, PyVista, and Three.js—when those components materially
  contribute to a published method, analysis, or figure.

Processed graphs are derived data. Preserve their `.pt.json` sidecar, the training
run's `config.yaml`, `subgraph_metadata.json`, dataset fingerprint, random seed, and Git
commit with any archived results. See [`data/README.md`](data/README.md) for a
field-by-field provenance map.

## References

<a id="ref-malecns-data"></a>**MaleCNS consortium (2026).** *Male CNS Connectome v1.0* [data set]. FlyEM, HHMI Janelia Research Campus; University of Cambridge; MRC Laboratory of Molecular Biology; and Google Research. [Landing page](https://male-cns.janelia.org/) · [download and license](https://male-cns.janelia.org/download/) · dataset identifier `male-cns:v1.0`.

<a id="ref-malecns-paper"></a>**Berg, S., Beckett, I. R., Costa, M., Schlegel, P., Januszewski, M., et al. (2026).** Sexual dimorphism in the complete connectome of the *Drosophila* male central nervous system. *Cell*. [Published article](https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6) · [bioRxiv DOI: 10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999).

<a id="ref-neuprint"></a>**Plaza, S. M., Clements, J., Dolafi, T., Umayam, L., Neubarth, N. N., Scheffer, L. K., & Berg, S. (2022).** neuPrint: An open access tool for EM connectomics. *Frontiers in Neuroinformatics, 16*, 896292. [DOI: 10.3389/fninf.2022.896292](https://doi.org/10.3389/fninf.2022.896292).

<a id="ref-vizdoom"></a>**Kempka, M., Wydmuch, M., Runc, G., Toczek, J., & Jaśkowski, W. (2016).** ViZDoom: A Doom-based AI research platform for visual reinforcement learning. *IEEE Conference on Computational Intelligence and Games*, 341–348. [DOI: 10.1109/CIG.2016.7860433](https://doi.org/10.1109/CIG.2016.7860433).

<a id="ref-ppo"></a>**Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** Proximal policy optimization algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).

<a id="ref-gae"></a>**Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016).** High-dimensional continuous control using generalized advantage estimation. *International Conference on Learning Representations*. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438).

<a id="ref-fremaux-gerstner"></a>**Frémaux, N., & Gerstner, W. (2016).** Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. *Frontiers in Neural Circuits, 9*, 85. [DOI: 10.3389/fncir.2015.00085](https://doi.org/10.3389/fncir.2015.00085).

<a id="ref-gerstner-eligibility"></a>**Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., & Brea, J. (2018).** Eligibility traces and plasticity on behavioral time scales: Experimental support of neo-Hebbian three-factor learning rules. *Frontiers in Neural Circuits, 12*, 53. [DOI: 10.3389/fncir.2018.00053](https://doi.org/10.3389/fncir.2018.00053).

<a id="ref-aso-architecture"></a>**Aso, Y., Hattori, D., Yu, Y., Johnston, R. M., Iyer, N. A., et al. (2014a).** The neuronal architecture of the mushroom body provides a logic for associative learning. *eLife, 3*, e04577. [DOI: 10.7554/eLife.04577](https://doi.org/10.7554/eLife.04577).

<a id="ref-aso-valence"></a>**Aso, Y., Sitaraman, D., Ichinose, T., Kaun, K. R., Vogt, K., et al. (2014b).** Mushroom body output neurons encode valence and guide memory-based action selection in *Drosophila*. *eLife, 3*, e04580. [DOI: 10.7554/eLife.04580](https://doi.org/10.7554/eLife.04580).

<a id="ref-erdos-renyi"></a>**Erdős, P., & Rényi, A. (1959).** On random graphs I. *Publicationes Mathematicae, 6*, 290–297. [DOI: 10.5486/PMD.1959.6.3-4.12](https://doi.org/10.5486/PMD.1959.6.3-4.12).

<a id="ref-maslov-sneppen"></a>**Maslov, S., & Sneppen, K. (2002).** Specificity and stability in topology of protein networks. *Science, 296*(5569), 910–913. [DOI: 10.1126/science.1065103](https://doi.org/10.1126/science.1065103).

<a id="ref-pytorch"></a>**Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., et al. (2019).** PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*. [Paper](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html).

<a id="ref-numpy"></a>**Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., et al. (2020).** Array programming with NumPy. *Nature, 585*, 357–362. [DOI: 10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-pandas"></a>**McKinney, W. (2010).** Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. [DOI: 10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a).

<a id="ref-scipy"></a>**Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., et al. (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*, 261–272. [DOI: 10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

<a id="ref-matplotlib"></a>**Hunter, J. D. (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. [DOI: 10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

<a id="ref-gymnasium"></a>**Towers, M., Kwiatkowski, A., Terry, J. K., Balis, J. U., De Cola, G., et al. (2024).** Gymnasium: A standard interface for reinforcement learning environments. [arXiv:2407.17032](https://arxiv.org/abs/2407.17032).

<a id="ref-networkx"></a>**Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008).** Exploring network structure, dynamics, and function using NetworkX. *Proceedings of the 7th Python in Science Conference*, 11–15. [Paper](https://conference.scipy.org/proceedings/SciPy2008/paper_2/).

<a id="ref-pyvista"></a>**Sullivan, C. B., & Kaszynski, A. A. (2019).** PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). *Journal of Open Source Software, 4*(37), 1450. [DOI: 10.21105/joss.01450](https://doi.org/10.21105/joss.01450).

**Three.js contributors.** *Three.js: JavaScript 3D library* [software]. [Project website](https://threejs.org/) · [source repository](https://github.com/mrdoob/three.js).

## Quality checks

```bash
pytest
ruff check .
```

Tests cover canonical/Feather loading, ID mapping, COO/CSR sparsity, subgraph paths, matched graph controls, degree preservation, observation/action mapping, constrained sparse gradients, PPO updates, deterministic checkpoint evaluation, and trace serialization.

## Scientific interpretation

Treat topology choice, neuron-selection rules, weight threshold, signs, parameter budget, environment interactions, optimization budget, and seeds as controlled experimental variables. Report effect sizes and uncertainty. A null or negative result is a valid answer; this project does not assume that biological topology helps.
