#!/usr/bin/env python3
"""Train connectome, matched-graph, MLP, GRU, and LSTM policies with PPO."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from flydoom.config import load_config
from flydoom.data.annotations import transmitter_signs
from flydoom.data.controls import degree_preserving_rewire, erdos_renyi_matched
from flydoom.data.loaders import generate_mock_connectome, load_connectome
from flydoom.data.male_cns import load_male_cns_bulk, load_male_cns_neuprint
from flydoom.data.schema import ConnectomeGraph
from flydoom.data.subgraph import SubgraphSpec, extract_subgraph
from flydoom.env.actions import action_count
from flydoom.env.doom_env import make_environment
from flydoom.experiments.baselines import GRUPolicy, LSTMPolicy, MLPPolicy
from flydoom.models.connectome_network import ConnectomeRateNetwork
from flydoom.models.graph_policy import ConnectomePolicy
from flydoom.training.checkpoint import save_checkpoint
from flydoom.training.ppo import PPO, PPOConfig
from flydoom.training.trainer import evaluate, train
from flydoom.visualization.learning_curves import plot_learning_curves
from flydoom.visualization.recording import record_episode

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("flydoom.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fingerprint(graph: ConnectomeGraph) -> str:
    digest = hashlib.sha256()
    digest.update(
        pd.util.hash_pandas_object(graph.neurons.astype(str), index=True).values.tobytes()
    )
    digest.update(pd.util.hash_pandas_object(graph.edges.astype(str), index=True).values.tobytes())
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_graph(config: dict[str, Any]) -> ConnectomeGraph:
    data = config["data"]
    kind = data.get("kind", "mock")
    if kind == "mock":
        graph = generate_mock_connectome(
            int(data.get("node_count", 1_000)), int(data.get("seed", 17))
        )
    elif kind == "bulk":
        graph = load_male_cns_bulk(data.get("directory", "data/raw/male-cns-v1.0"))
    elif kind == "neuprint":
        graph = load_male_cns_neuprint(
            types=data.get("types"),
            classes=data.get("classes"),
            rois=data.get("regions"),
            body_ids=data.get("body_ids"),
            min_weight=int(data.get("min_synapses", 3)),
            max_neurons=int(data.get("max_neurons", 5_000)),
            num_hops=int(data.get("num_hops", 4)),
        )
    elif kind == "generic":
        graph = load_connectome(data["neurons"], data["edges"])
    elif kind == "processed":
        payload = torch.load(data["path"], map_location="cpu", weights_only=True)
        graph = ConnectomeGraph(pd.DataFrame(payload["neurons"]), pd.DataFrame(payload["edges"]))
    else:
        raise ValueError(f"Unknown data kind: {kind}")
    spec = SubgraphSpec(
        source_class=data.get("source_class"),
        target_class=data.get("target_class"),
        source_region=data.get("source_region"),
        target_region=data.get("target_region"),
        neuron_types=tuple(data.get("types") or ()) if kind != "neuprint" else (),
        neuron_classes=tuple(data.get("classes") or ()) if kind != "neuprint" else (),
        regions=tuple(data.get("regions") or ()) if kind != "neuprint" else (),
        seed_body_ids=tuple(data.get("body_ids") or ()),
        min_synapses=float(data.get("min_synapses", 1)),
        max_neurons=int(data.get("max_neurons", 5_000)),
        num_hops=int(data.get("num_hops", 4)),
        shortest_path_union=bool(data.get("shortest_path_union", True)),
        top_connected=data.get("top_connected"),
    )
    return extract_subgraph(graph, spec)


def build_connectome_policy(
    graph: ConnectomeGraph, config: dict[str, Any], device: torch.device
) -> ConnectomePolicy:
    edge_index, weights = graph.sparse_tensors()
    model = config["model"]
    signs = None
    if model.get("sign_constraints", False):
        source_nt = graph.neurons.set_index("body_id")["neurotransmitter"]
        signed_edges = graph.edges.copy()
        signed_edges["neurotransmitter"] = (
            signed_edges["pre_body_id"].map(source_nt).fillna("unknown")
        )
        signs = transmitter_signs(signed_edges)
    network = ConnectomeRateNetwork(
        graph.node_count,
        edge_index,
        weights,
        leak=float(model.get("leak", 0.5)),
        weight_scale=float(model.get("weight_scale", 0.05)),
        trainable_weights=model.get("training_mode", "trainable_internal") == "trainable_internal",
        normalize_incoming=bool(model.get("normalize_incoming", True)),
        transmitter_sign=signs,
        trainable_bias=model.get("training_mode", "trainable_internal") == "trainable_internal",
        activation=str(model.get("activation", "tanh")),
    )
    sensory = torch.tensor(
        graph.neurons.index[graph.neurons["is_sensory"].astype(bool)].tolist(), dtype=torch.long
    )
    descending = torch.tensor(
        graph.neurons.index[graph.neurons["is_descending"].astype(bool)].tolist(), dtype=torch.long
    )
    if not len(sensory) or not len(descending):
        raise ValueError(
            "The selected graph needs sensory and descending neurons. Set canonical boolean "
            "annotations or choose matching source/target populations."
        )
    return ConnectomePolicy(
        network,
        sensory,
        descending,
        action_count(),
        encoder=str(model.get("encoder", "cnn")),
        propagation_steps=int(model.get("propagation_steps", 5)),
        training_mode=str(model.get("training_mode", "trainable_internal")),
    ).to(device)


def parameter_count(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def ppo_config(training: dict[str, Any]) -> PPOConfig:
    names = set(PPOConfig.__dataclass_fields__)
    return PPOConfig(**{key: value for key, value in training.items() if key in names})


def main(arguments: list[str] | None = None) -> Path:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    defaults = (
        ("experiment", "flydoom_basic"),
        ("data", "mock_connectome"),
        ("model", "connectome_rate"),
    )
    for index, (group, name) in enumerate(defaults):
        if not any(argument.startswith(group + "=") for argument in arguments):
            arguments.insert(index, f"{group}={name}")
    config = load_config(arguments)
    seed = int(config.get("seed", 42))
    requested = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use device=cpu")
    device = torch.device(requested)
    set_seed(seed)
    graph = load_graph(config)
    LOGGER.info("Graph: %d neurons, %d real directed edges", graph.node_count, graph.edge_count)

    run_dir = (
        Path(config["output"]["root"]) / date.today().isoformat() / f"fly_connectome_seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config.update(
        seed=seed,
        device=str(device),
        git_commit=git_commit(),
        dataset_fingerprint=fingerprint(graph),
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {
        "neurons": graph.node_count,
        "edges": graph.edge_count,
        "fingerprint": fingerprint(graph),
    }
    (run_dir / "subgraph_metadata.json").write_text(json.dumps(metadata, indent=2))

    requested_variants = config["experiment"].get(
        "variants", ["real_connectome", "erdos_renyi", "degree_rewired", "mlp", "gru", "lstm"]
    )
    graph_variants: dict[str, ConnectomeGraph] = {"real_connectome": graph}
    if "erdos_renyi" in requested_variants:
        graph_variants["erdos_renyi"] = erdos_renyi_matched(graph, seed + 1)
    if "degree_rewired" in requested_variants:
        graph_variants["degree_rewired"] = degree_preserving_rewire(graph, seed + 2)
    all_metrics: list[pd.DataFrame] = []
    comparisons: list[dict[str, Any]] = []
    for name in requested_variants:
        set_seed(seed)
        variant = graph_variants.get(name)
        if variant is not None:
            policy = build_connectome_policy(variant, config, device)
        else:
            hidden = int(config["model"].get("baseline_hidden_size", 128))
            policies = {"mlp": MLPPolicy, "gru": GRUPolicy, "lstm": LSTMPolicy}
            if name not in policies:
                raise ValueError(f"Unknown experiment variant: {name}")
            policy = policies[name](hidden, action_count()).to(device)
        algorithm = PPO(policy, ppo_config(config["training"]))
        model_dir = run_dir / name
        model_dir.mkdir(exist_ok=True)
        env = make_environment(config["env"])
        rows = train(
            env,
            algorithm,
            total_steps=int(config["training"]["total_steps"]),
            rollout_steps=int(config["training"]["rollout_steps"]),
            seed=seed,
            device=device,
        )
        metrics = pd.DataFrame(rows)
        metrics.insert(0, "model", name)
        metrics["graph_neurons"] = variant.node_count if variant else 0
        metrics["graph_edges"] = variant.edge_count if variant else 0
        metrics["parameter_count"] = parameter_count(policy)
        metrics.to_csv(model_dir / "metrics.csv", index=False)
        all_metrics.append(metrics)
        save_checkpoint(
            model_dir / "checkpoint.pt",
            policy,
            algorithm.optimizer,
            {
                "seed": seed,
                "model": name,
                "environment_steps": config["training"]["total_steps"],
                "graph_neurons": variant.node_count if variant else 0,
                "graph_edges": variant.edge_count if variant else 0,
            },
        )
        evaluation_env = make_environment(config["env"])
        rewards = evaluate(
            evaluation_env,
            policy,
            int(config["experiment"]["evaluation_episodes"]),
            seed + 10_000,
            device,
        )
        evaluation = {"rewards": rewards, "mean_reward": float(np.mean(rewards))}
        (model_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        if name == "real_connectome" and config["output"].get("record_activity", True):
            trace_env = make_environment(config["env"])
            record_episode(trace_env, policy, graph, device=device, seed=seed + 20_000).save(
                model_dir / "episode_trace.npz"
            )
            trace_env.close()
        comparisons.append(
            {
                "model": name,
                "parameters": parameter_count(policy),
                "trainable_parameters": parameter_count(policy, True),
                "graph_neurons": variant.node_count if variant else 0,
                "graph_edges": variant.edge_count if variant else 0,
                "final_evaluation_reward": evaluation["mean_reward"],
                "training_reward_auc": float(
                    np.trapezoid(metrics.episodic_reward, metrics.environment_steps)
                ),
            }
        )
        env.close()
        evaluation_env.close()
        LOGGER.info("Completed %s: evaluation reward %.3f", name, evaluation["mean_reward"])
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(run_dir / "comparison.csv", index=False)
    plot_learning_curves(combined, run_dir / "learning_curves.png")
    LOGGER.info("Artifacts written to %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
