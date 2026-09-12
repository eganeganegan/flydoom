from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flydoom.data.controls import degree_preserving_rewire, erdos_renyi_matched
from flydoom.data.loaders import generate_mock_connectome, load_connectome
from flydoom.env.actions import DoomAction, action_count
from flydoom.env.doom_env import MockDoomEnv
from flydoom.env.observations import FlyInspiredVisualEncoder, SimpleCNNEncoder
from flydoom.models.connectome_network import ConnectomeRateNetwork
from flydoom.models.graph_policy import ConnectomePolicy
from flydoom.training.checkpoint import load_checkpoint, save_checkpoint
from flydoom.training.ppo import PPO, PPOConfig
from flydoom.training.rollout import RolloutBuffer
from flydoom.training.trainer import evaluate


def policy_for(node_count: int = 40) -> ConnectomePolicy:
    graph = generate_mock_connectome(node_count, seed=4)
    edge_index, weights = graph.sparse_tensors()
    sensory = torch.tensor(graph.neurons.index[graph.neurons.is_sensory].tolist())
    descending = torch.tensor(graph.neurons.index[graph.neurons.is_descending].tolist())
    return ConnectomePolicy(
        ConnectomeRateNetwork(node_count, edge_index, weights),
        sensory,
        descending,
        propagation_steps=5,
    )


def test_csv_loader(tmp_path: Path) -> None:
    graph = generate_mock_connectome(40, seed=1)
    neurons = tmp_path / "neurons.csv"
    edges = tmp_path / "edges.csv"
    graph.neurons.to_csv(neurons, index=False)
    graph.edges.to_csv(edges, index=False)
    loaded = load_connectome(neurons, edges)
    assert (loaded.node_count, loaded.edge_count) == (graph.node_count, graph.edge_count)


def test_matched_controls_preserve_requested_properties() -> None:
    graph = generate_mock_connectome(40, seed=2)
    random_graph = erdos_renyi_matched(graph, seed=3)
    assert (random_graph.node_count, random_graph.edge_count) == (
        graph.node_count,
        graph.edge_count,
    )
    rewired = degree_preserving_rewire(graph, seed=3, swaps_per_edge=1)
    original_out = graph.edges.pre_neuron_id.value_counts().sort_index()
    rewired_out = rewired.edges.pre_neuron_id.value_counts().sort_index()
    original_in = graph.edges.post_neuron_id.value_counts().sort_index()
    rewired_in = rewired.edges.post_neuron_id.value_counts().sort_index()
    pd.testing.assert_series_equal(original_out, rewired_out)
    pd.testing.assert_series_equal(original_in, rewired_in)


def test_observation_and_action_mapping() -> None:
    images = torch.zeros(2, 42, 42, 3, dtype=torch.uint8)
    assert SimpleCNNEncoder(16)(images).shape == (2, 16)
    assert FlyInspiredVisualEncoder()(images).shape == (2, 112)
    assert action_count() == 5
    assert DoomAction.SHOOT == 4
    policy = policy_for()
    assert policy.action_readout.in_features == len(policy.descending_indices)


def test_ppo_rollout_updates_policy() -> None:
    policy = policy_for()
    algorithm = PPO(policy, PPOConfig(epochs=1, batch_size=4))
    buffer = RolloutBuffer()
    for index in range(4):
        observation = torch.zeros(1, 42, 42, 3, dtype=torch.uint8)
        with torch.no_grad():
            action, log_prob, value = policy.act(observation)
        buffer.observations.append(observation)
        buffer.actions.append(action.squeeze(0))
        buffer.log_probs.append(log_prob.squeeze(0))
        buffer.values.append(value.squeeze(0))
        buffer.rewards.append(float(index == 3))
        buffer.dones.append(index == 3)
    before = policy.action_readout.weight.detach().clone()
    metrics = algorithm.update(buffer, torch.tensor(0.0))
    assert set(metrics) == {"policy_loss", "value_loss", "policy_entropy", "gradient_norm"}
    assert not torch.equal(before, policy.action_readout.weight)


def test_checkpoint_and_deterministic_evaluation(tmp_path: Path) -> None:
    policy = policy_for()
    optimizer = torch.optim.Adam(policy.parameters())
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, policy, optimizer, {"seed": 9})
    restored = policy_for()
    metadata = load_checkpoint(checkpoint, restored)
    assert metadata == {"seed": 9}
    first = evaluate(MockDoomEnv(max_steps=5), restored, 2, 10, torch.device("cpu"))
    second = evaluate(MockDoomEnv(max_steps=5), restored, 2, 10, torch.device("cpu"))
    np.testing.assert_allclose(first, second)
