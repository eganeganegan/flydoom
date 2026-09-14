from __future__ import annotations

import numpy as np
import torch

from flydoom.data.loaders import generate_mock_connectome
from flydoom.models.connectome_network import ConnectomeRateNetwork
from flydoom.models.graph_policy import ConnectomePolicy
from flydoom.training.three_factor import (
    ThreeFactorConfig,
    ThreeFactorLearner,
    train_three_factor,
)


class TinyVisualEnv:
    action_names = ("no_op", "left", "right", "shoot")

    def __init__(self) -> None:
        self.action_space = type("ActionSpace", (), {"n": 4})()
        self.steps = 0

    def reset(self, *, seed: int | None = None):
        self.steps = 0
        return np.zeros((42, 42, 3), dtype=np.uint8), {}

    def step(self, action: int):
        self.steps += 1
        done = self.steps == 4
        reward = 1.0 if action == 3 else -0.01
        components = {"kill": reward} if reward > 0 else {"living": reward}
        info = {
            "kills": float(reward > 0),
            "hits": float(reward > 0),
            "health": 100.0,
            "ammo": 100.0 - self.steps,
            "damage_taken": 0.0,
            "deaths": 0.0,
            "survival_time": self.steps,
            "reward_components": components,
        }
        return np.zeros((42, 42, 3), dtype=np.uint8), reward, done, False, info


def biological_policy() -> ConnectomePolicy:
    graph = generate_mock_connectome(40, seed=4)
    edge_index, weights = graph.sparse_tensors()
    sensory = torch.tensor(graph.neurons.index[graph.neurons.is_sensory].tolist())
    descending = torch.tensor(graph.neurons.index[graph.neurons.is_descending].tolist())
    return ConnectomePolicy(
        ConnectomeRateNetwork(40, edge_index, weights),
        sensory,
        descending,
        action_count=4,
        encoder="fly",
        training_mode="three_factor",
    )


def test_three_factor_learning_updates_locally_without_gradients() -> None:
    policy = biological_policy()
    learner = ThreeFactorLearner(
        policy,
        ThreeFactorConfig(
            learning_rate=0.01,
            readout_learning_rate=0.01,
            reward_scale=1.0,
        ),
    )
    observation = torch.randint(0, 256, (1, 42, 42, 3), dtype=torch.uint8)
    state = policy.initial_state(1, torch.device("cpu"))
    action, logits, value, activity, _ = learner.act(observation, state)
    before_edges = policy.network.edge_weight.clone()
    before_readout = policy.action_readout.weight.clone()

    metrics = learner.update(
        activity,
        logits,
        value,
        action,
        reward=1.0,
        next_value=torch.tensor(0.0),
        done=True,
    )

    assert metrics["dopamine_prediction_error"] != 0
    assert not torch.equal(before_edges, policy.network.edge_weight)
    assert not torch.equal(before_readout, policy.action_readout.weight)
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert all(not parameter.requires_grad for parameter in policy.parameters())


def test_three_factor_training_loop_records_local_learning_metrics() -> None:
    policy = biological_policy()
    learner = ThreeFactorLearner(policy, ThreeFactorConfig(reward_scale=1.0))

    rows = train_three_factor(
        TinyVisualEnv(),
        learner,
        total_steps=8,
        report_interval=4,
        seed=7,
        device=torch.device("cpu"),
    )

    assert rows[-1]["environment_steps"] == 8
    assert rows[-1]["plastic_edges"] == policy.network.edge_weight.numel()
    assert "dopamine_prediction_error" in rows[-1]
