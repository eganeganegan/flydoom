#!/usr/bin/env python3
"""Train connectome and topology-matched control policies with PPO."""

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
from flydoom.data.controls import erdos_renyi_matched
from flydoom.data.loaders import generate_mock_connectome, load_connectome
from flydoom.data.schema import ConnectomeGraph
from flydoom.data.subgraph import SubgraphSpec, extract_subgraph
from flydoom.env.actions import action_count
from flydoom.env.doom_env import make_environment
from flydoom.models.connectome_network import ConnectomeRateNetwork
from flydoom.models.graph_policy import ConnectomePolicy
from flydoom.training.checkpoint import save_checkpoint
from flydoom.training.ppo import PPO, PPOConfig
from flydoom.training.trainer import evaluate, train
from flydoom.visualization.learning_curves import plot_learning_curves

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
    digest.update(pd.util.hash_pandas_object(graph.neurons, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(graph.edges, index=True).values.tobytes())
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_policy(
    graph: ConnectomeGraph, config: dict[str, Any], device: torch.device
) -> ConnectomePolicy:
    edge_index, weights = graph.sparse_tensors()
    model_config = config["model"]
    network = ConnectomeRateNetwork(
        graph.node_count,
        edge_index,
        weights,
        leak=float(model_config["leak"]),
        weight_scale=float(model_config["weight_scale"]),
        trainable_weights=bool(model_config["trainable_weights"]),
        normalize_incoming=bool(model_config["normalize_incoming"]),
    )
    sensory = torch.tensor(
        graph.neurons.index[graph.neurons["is_sensory"].astype(bool)].tolist(), dtype=torch.long
    )
    descending = torch.tensor(
        graph.neurons.index[graph.neurons["is_descending"].astype(bool)].tolist(), dtype=torch.long
    )
    return ConnectomePolicy(
        network,
        sensory,
        descending,
        action_count(),
        encoder=str(model_config["encoder"]),
        propagation_steps=int(model_config["propagation_steps"]),
    ).to(device)


def estimate_megabytes(graph: ConnectomeGraph, rollout_steps: int) -> float:
    graph_bytes = graph.edge_count * (2 * 8 + 4) + graph.node_count * 4
    rollout_bytes = rollout_steps * 42 * 42 * 3
    return (graph_bytes + rollout_bytes) / (1024**2)


def main(arguments: list[str] | None = None) -> Path:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if not any(argument.startswith("experiment=") for argument in arguments):
        arguments.insert(0, "experiment=flydoom_basic")
    if not any(argument.startswith("data=") for argument in arguments):
        arguments.insert(1, "data=mock_connectome")
    if not any(argument.startswith("model=") for argument in arguments):
        arguments.insert(2, "model=connectome_rate")
    config = load_config(arguments)
    seed = int(config.get("seed", 42))
    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use device=cpu")
    device = torch.device(requested_device)
    set_seed(seed)

    data_config = config["data"]
    if data_config.get("kind") != "mock":
        raise NotImplementedError("Use scripts/build_subgraph.py for real tables in this milestone")
    generated = generate_mock_connectome(int(data_config["node_count"]), int(data_config["seed"]))
    raw_dir = Path("data/raw/mock_connectome")
    raw_dir.mkdir(parents=True, exist_ok=True)
    generated.neurons.to_csv(raw_dir / "neurons.csv", index=False)
    generated.edges.to_csv(raw_dir / "synapses.csv", index=False)
    loaded = load_connectome(raw_dir / "neurons.csv", raw_dir / "synapses.csv")
    graph = extract_subgraph(
        loaded,
        SubgraphSpec(
            source_class="visual",
            target_class="descending",
            min_synapses=float(data_config["min_synapses"]),
            max_neurons=int(data_config["max_neurons"]),
            num_hops=int(data_config["num_hops"]),
        ),
    )

    estimate = estimate_megabytes(graph, int(config["training"]["rollout_steps"]))
    LOGGER.info(
        "Graph: %d neurons, %d edges, %.1f estimated MB (excluding autograd)",
        graph.node_count,
        graph.edge_count,
        estimate,
    )
    if estimate > 8_000:
        raise MemoryError("Estimated graph/rollout storage exceeds 8 GB; reduce the subgraph")

    run_dir = (
        Path(config["output"]["root"]) / date.today().isoformat() / f"fly_connectome_seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config["seed"] = seed
    config["device"] = str(device)
    config["git_commit"] = git_commit()
    config["dataset_fingerprint"] = fingerprint(graph)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "subgraph_metadata.json").write_text(
        json.dumps(
            {
                "neurons": graph.node_count,
                "edges": graph.edge_count,
                "fingerprint": fingerprint(graph),
            },
            indent=2,
        )
    )

    variants = {"fly_connectome": graph}
    if config["experiment"].get("compare_erdos_renyi", True):
        variants["erdos_renyi"] = erdos_renyi_matched(graph, seed + 1)
    all_metrics: list[pd.DataFrame] = []
    comparisons: list[dict[str, Any]] = []
    for name, variant in variants.items():
        set_seed(seed)
        policy = build_policy(variant, config, device)
        ppo_keys = {
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_range",
            "entropy_coef",
            "value_coef",
            "max_grad_norm",
            "epochs",
            "batch_size",
        }
        algorithm = PPO(
            policy,
            PPOConfig(
                **{key: value for key, value in config["training"].items() if key in ppo_keys}
            ),
        )
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
        metrics.to_csv(model_dir / "metrics.csv", index=False)
        if config["output"].get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "Install FlyDoom with the `tracking` extra for TensorBoard"
                ) from exc
            writer = SummaryWriter(model_dir / "tensorboard")
            for row in rows:
                step = int(row["environment_steps"])
                for key, value in row.items():
                    if key != "environment_steps" and isinstance(value, (int, float)):
                        writer.add_scalar(key, value, step)
            writer.close()
        all_metrics.append(metrics)
        save_checkpoint(
            model_dir / "checkpoint.pt",
            policy,
            algorithm.optimizer,
            {"seed": seed, "model": name, "environment_steps": config["training"]["total_steps"]},
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
        comparisons.append(
            {
                "model": name,
                "parameters": sum(parameter.numel() for parameter in policy.parameters()),
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
