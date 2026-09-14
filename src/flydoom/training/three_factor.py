"""Online dopamine-modulated local plasticity for connectome policies."""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.distributions import Categorical

from flydoom.env.rewards import REWARD_COMPONENT_NAMES
from flydoom.models.graph_policy import ConnectomePolicy

if TYPE_CHECKING:
    import gymnasium as gym


def observation_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(observation, device=device).unsqueeze(0)


@dataclass(frozen=True, slots=True)
class ThreeFactorConfig:
    """Parameters for eligibility × dopamine plasticity."""

    learning_rate: float = 1e-4
    readout_learning_rate: float = 2e-3
    value_learning_rate: float = 5e-3
    gamma: float = 0.99
    eligibility_decay: float = 0.95
    reward_scale: float = 0.1
    homeostasis_rate: float = 1e-5
    target_activity: float = 0.10
    weight_decay: float = 1e-6
    max_weight: float = 1.0
    max_readout_weight: float = 3.0
    temperature: float = 1.0
    dopamine_state_gain: float = 0.25

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ThreeFactorConfig:
        names = {field.name for field in fields(cls)}
        return cls(**{name: values[name] for name in names if name in values})

    def __post_init__(self) -> None:
        if not 0 <= self.eligibility_decay <= 1:
            raise ValueError("eligibility_decay must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.max_weight <= 0 or self.max_readout_weight <= 0:
            raise ValueError("weight bounds must be positive")


class ThreeFactorLearner:
    """Apply local eligibility-trace plasticity without gradients or an optimizer."""

    def __init__(
        self,
        policy: ConnectomePolicy,
        config: ThreeFactorConfig,
        edge_plasticity_mask: torch.Tensor | None = None,
        dopamine_indices: torch.Tensor | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        edge_weight = policy.network.edge_weight
        self.edge_eligibility = torch.zeros_like(edge_weight)
        self.action_eligibility = torch.zeros_like(policy.action_readout.weight)
        self.action_bias_eligibility = torch.zeros_like(policy.action_readout.bias)
        self.initial_edge_weight = edge_weight.detach().clone()
        if edge_plasticity_mask is None:
            edge_plasticity_mask = torch.ones_like(edge_weight, dtype=torch.bool)
        if edge_plasticity_mask.shape != edge_weight.shape:
            raise ValueError("edge plasticity mask must match the connectome edge count")
        self.edge_plasticity_mask = edge_plasticity_mask.to(edge_weight.device, dtype=torch.bool)
        self.dopamine_indices = (
            torch.empty(0, dtype=torch.long, device=edge_weight.device)
            if dopamine_indices is None
            else dopamine_indices.to(edge_weight.device, dtype=torch.long)
        )

    @property
    def plastic_edge_count(self) -> int:
        return int(self.edge_plasticity_mask.sum().item())

    @property
    def dopamine_neuron_count(self) -> int:
        return len(self.dopamine_indices)

    def reset_episode(self) -> None:
        self.edge_eligibility.zero_()
        self.action_eligibility.zero_()
        self.action_bias_eligibility.zero_()

    @torch.no_grad()
    def act(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, next_state = self.policy.forward_with_activity(observation, state)
        distribution = Categorical(logits=logits / self.config.temperature)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, logits, value, next_state, distribution.entropy()

    @torch.no_grad()
    def update(
        self,
        activity: torch.Tensor,
        logits: torch.Tensor,
        value: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_value: torch.Tensor,
        done: bool,
    ) -> dict[str, float]:
        """Update synapses from local activity and a scalar temporal-difference signal."""
        config = self.config
        scalar_value = value.squeeze()
        scalar_next = next_value.squeeze()
        dopamine = (
            reward * config.reward_scale
            + config.gamma * scalar_next * (1.0 - float(done))
            - scalar_value
        )
        dopamine = dopamine.clamp(-5.0, 5.0)
        node_activity = activity.squeeze(0)
        if self.dopamine_neuron_count:
            node_activity[self.dopamine_indices] = torch.tanh(
                node_activity[self.dopamine_indices]
                + config.dopamine_state_gain * dopamine
            )
        source, target = self.policy.network.edge_index
        hebbian = node_activity[source] * (
            node_activity[target] - node_activity[target].mean()
        )
        self.edge_eligibility.mul_(config.eligibility_decay).add_(hebbian)
        self.edge_eligibility.clamp_(-5.0, 5.0)

        edge_change = config.learning_rate * dopamine * self.edge_eligibility
        edge_change = edge_change * self.edge_plasticity_mask
        signs = self.policy.network.transmitter_sign
        raw_change = torch.where(signs == 0, edge_change, signs * edge_change)
        edge_weight = self.policy.network.edge_weight
        edge_weight.add_(raw_change)
        edge_weight.add_(
            config.weight_decay * (self.initial_edge_weight - edge_weight)
            * self.edge_plasticity_mask
        )
        signed = signs != 0
        edge_weight[signed].clamp_(0.0, config.max_weight)
        edge_weight[~signed].clamp_(-config.max_weight, config.max_weight)

        descending = node_activity[self.policy.descending_indices]
        probabilities = torch.softmax(logits.squeeze(0) / config.temperature, dim=-1)
        action_surprise = -probabilities
        action_surprise[int(action.item())] += 1.0
        local_action_trace = torch.outer(action_surprise, descending)
        self.action_eligibility.mul_(config.eligibility_decay).add_(local_action_trace)
        self.action_bias_eligibility.mul_(config.eligibility_decay).add_(action_surprise)
        self.policy.action_readout.weight.add_(
            config.readout_learning_rate * dopamine * self.action_eligibility
        ).clamp_(-config.max_readout_weight, config.max_readout_weight)
        self.policy.action_readout.bias.add_(
            config.readout_learning_rate * dopamine * self.action_bias_eligibility
        ).clamp_(-config.max_readout_weight, config.max_readout_weight)

        self.policy.value_readout.weight.add_(
            config.value_learning_rate * dopamine * descending.unsqueeze(0)
        ).clamp_(-config.max_readout_weight, config.max_readout_weight)
        self.policy.value_readout.bias.add_(config.value_learning_rate * dopamine).clamp_(
            -config.max_readout_weight, config.max_readout_weight
        )
        homeostatic_drive = config.target_activity - node_activity.abs()
        self.policy.network.bias.add_(config.homeostasis_rate * homeostatic_drive).clamp_(
            -0.5, 0.5
        )
        return {
            "dopamine_prediction_error": float(dopamine),
            "mean_abs_eligibility": float(
                self.edge_eligibility[self.edge_plasticity_mask].abs().mean()
            ),
            "mean_abs_activity": float(node_activity.abs().mean()),
            "local_edge_update_norm": float(raw_change.norm()),
            "value_prediction_error_squared": float(dopamine.square()),
        }


def _mean_metrics(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([sample[name] for sample in samples]))
        for name in samples[0]
    }


def train_three_factor(
    env: gym.Env[np.ndarray, int],
    learner: ThreeFactorLearner,
    *,
    total_steps: int,
    report_interval: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Train online with local plasticity and return PPO-compatible experiment rows."""
    observation, _ = env.reset(seed=seed)
    state = learner.policy.initial_state(1, device)
    learner.reset_episode()
    episode_reward = 0.0
    episode_length = 0
    episode_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, float]] = []
    action_counts = np.zeros(int(env.action_space.n), dtype=np.int64)
    started = time.perf_counter()
    final_info: dict[str, Any] = {}

    for completed_steps in range(1, total_steps + 1):
        tensor = observation_tensor(observation, device)
        action, logits, value, next_state, entropy = learner.act(tensor, state)
        next_observation, reward, terminated, truncated, final_info = env.step(
            int(action.item())
        )
        done = terminated or truncated
        if done:
            next_value = value.new_zeros(())
        else:
            with torch.no_grad():
                next_tensor = observation_tensor(next_observation, device)
                _, next_value, _ = learner.policy.forward_with_activity(
                    next_tensor, next_state
                )
        update_metrics = learner.update(
            next_state, logits, value, action, float(reward), next_value, done
        )
        update_metrics["policy_entropy"] = float(entropy.mean())
        samples.append(update_metrics)
        action_counts[int(action.item())] += 1
        episode_reward += float(reward)
        episode_length += 1
        for name, component in final_info.get("reward_components", {}).items():
            if name in episode_components:
                episode_components[name] += float(component)
        observation = next_observation
        state = next_state

        if done:
            rows.append(
                {
                    "environment_steps": completed_steps,
                    "episodic_reward": episode_reward,
                    "episode_length": episode_length,
                    "kills": final_info.get("kills", 0),
                    "hits": final_info.get("hits", 0),
                    "health": final_info.get("health", np.nan),
                    "ammo": final_info.get("ammo", np.nan),
                    "damage_taken": final_info.get("damage_taken", 0),
                    "deaths": final_info.get("deaths", 0),
                    "survival_time": final_info.get("survival_time", episode_length),
                    "distance_travelled": final_info.get("distance_travelled", 0.0),
                    "episode_dopamine": episode_reward,
                    **{
                        f"reward_component_{name}": component
                        for name, component in episode_components.items()
                    },
                }
            )
            observation, _ = env.reset()
            state = learner.policy.initial_state(1, device)
            learner.reset_episode()
            episode_reward = 0.0
            episode_length = 0
            episode_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}

        if completed_steps % report_interval == 0 or completed_steps == total_steps:
            if not rows or rows[-1]["environment_steps"] != completed_steps:
                rows.append(
                    {
                        "environment_steps": completed_steps,
                        "episodic_reward": episode_reward,
                        "episode_length": episode_length,
                        "kills": final_info.get("kills", 0),
                        "hits": final_info.get("hits", 0),
                        "health": final_info.get("health", np.nan),
                        "ammo": final_info.get("ammo", np.nan),
                        "damage_taken": final_info.get("damage_taken", 0),
                        "deaths": final_info.get("deaths", 0),
                        "survival_time": final_info.get("survival_time", episode_length),
                        "distance_travelled": final_info.get("distance_travelled", 0.0),
                        "episode_dopamine": episode_reward,
                        **{
                            f"reward_component_{name}": component
                            for name, component in episode_components.items()
                        },
                    }
                )
            rows[-1].update(_mean_metrics(samples))
            names = getattr(
                env,
                "action_names",
                tuple(f"action_{index}" for index in range(len(action_counts))),
            )
            for index, count in enumerate(action_counts):
                rows[-1][f"action_fraction_{names[index]}"] = float(
                    count / max(action_counts.sum(), 1)
                )
            elapsed = max(time.perf_counter() - started, 1e-6)
            rows[-1]["fps"] = completed_steps / elapsed
            rows[-1]["plastic_edges"] = learner.plastic_edge_count
            rows[-1]["dopamine_neurons"] = learner.dopamine_neuron_count
            samples.clear()
            action_counts.fill(0)
    return rows
