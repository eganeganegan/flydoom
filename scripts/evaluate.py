#!/usr/bin/env python3
"""Inspect saved results or deterministically rerun checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from train import build_connectome_policy, load_graph

from flydoom.data.controls import degree_preserving_rewire, erdos_renyi_matched
from flydoom.env.doom_env import make_environment
from flydoom.experiments.baselines import GRUPolicy, LSTMPolicy, MLPPolicy
from flydoom.training.checkpoint import load_checkpoint
from flydoom.training.trainer import evaluate, reward_summary


def inspect(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.glob("*/evaluation.json")):
        evaluation = json.loads(path.read_text())
        rows.append({"model": path.parent.name, **evaluation})
    if not rows:
        raise SystemExit(f"No evaluation artifacts found under {run_dir}")
    print(pd.DataFrame(rows).to_string(index=False))


def rerun(run_dir: Path, episodes: int, device: torch.device, seed: int) -> None:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    base_graph = load_graph(config)
    hidden = int(config["model"].get("baseline_hidden_size", 128))
    for checkpoint in sorted(run_dir.glob("*/checkpoint.pt")):
        name = checkpoint.parent.name
        env = make_environment(config["env"])
        num_actions = int(env.action_space.n)
        if name == "real_connectome":
            graph = base_graph
        elif name == "erdos_renyi":
            graph = erdos_renyi_matched(base_graph, int(config["seed"]) + 1)
        elif name == "degree_rewired":
            graph = degree_preserving_rewire(base_graph, int(config["seed"]) + 2)
        else:
            graph = None
        if graph is not None:
            policy = build_connectome_policy(graph, config, device, num_actions)
        else:
            policies = {"mlp": MLPPolicy, "gru": GRUPolicy, "lstm": LSTMPolicy}
            policy = policies[name](hidden, num_actions).to(device)
        load_checkpoint(checkpoint, policy, map_location=device)
        rewards = evaluate(env, policy, episodes, seed, device)
        env.close()
        result = {**reward_summary(rewards), "seed": seed}
        (checkpoint.parent / "evaluation-rerun.json").write_text(json.dumps(result, indent=2))
        print(f"{name:20s} mean_reward={result['mean_reward']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.rerun:
        rerun(args.run_dir, args.episodes, torch.device(args.device), args.seed)
    else:
        inspect(args.run_dir)


if __name__ == "__main__":
    main()
