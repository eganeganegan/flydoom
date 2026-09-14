"""Environment interaction and experiment metric collection."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from flydoom.env.rewards import REWARD_COMPONENT_NAMES
from flydoom.training.ppo import PPO
from flydoom.training.rollout import RecurrentState, RolloutBuffer


def observation_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(observation, device=device).unsqueeze(0)


def reward_summary(rewards: list[float]) -> dict[str, Any]:
    """Summarize evaluation rewards without hiding episode-level variance."""
    values = np.asarray(rewards, dtype=float)
    if not len(values):
        raise ValueError("At least one evaluation reward is required")
    return {
        "rewards": rewards,
        "mean_reward": float(values.mean()),
        "median_reward": float(np.median(values)),
        "std_reward": float(values.std()),
        "min_reward": float(values.min()),
        "max_reward": float(values.max()),
    }


def evaluation_summary(episodes: list[dict[str, float]]) -> dict[str, Any]:
    """Summarize reward and task outcomes for a deterministic evaluation set."""
    if not episodes:
        raise ValueError("At least one evaluation episode is required")
    summary = reward_summary([episode["reward"] for episode in episodes])
    for name in ("kills", "hits", "damage_taken", "deaths", "survival_time"):
        summary[f"mean_{name}"] = float(np.mean([episode[name] for episode in episodes]))
    summary["success_rate"] = float(np.mean([episode["kills"] > 0 for episode in episodes]))
    summary["episodes"] = episodes
    return summary


def _detach_state(state: RecurrentState) -> RecurrentState:
    if isinstance(state, tuple):
        return state[0].detach(), state[1].detach()
    return state.detach()


def train(
    env: gym.Env[np.ndarray, int],
    algorithm: PPO,
    *,
    total_steps: int,
    rollout_steps: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    observation, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_length = 0
    episode_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    completed_steps = 0
    recurrent = hasattr(algorithm.policy, "initial_state")
    state = algorithm.policy.initial_state(1, device) if recurrent else None
    while completed_steps < total_steps:
        buffer = RolloutBuffer()
        action_counts = np.zeros(int(env.action_space.n), dtype=np.int64)
        for _ in range(min(rollout_steps, total_steps - completed_steps)):
            tensor = observation_tensor(observation, device)
            with torch.no_grad():
                if state is None:
                    action, log_prob, value = algorithm.policy.act(tensor)
                else:
                    buffer.recurrent_states.append(_detach_state(state))
                    action, log_prob, value, state = algorithm.policy.act_recurrent(tensor, state)
            next_observation, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated
            buffer.observations.append(tensor)
            buffer.actions.append(action.squeeze(0))
            action_counts[int(action.item())] += 1
            buffer.rewards.append(float(reward) * algorithm.config.reward_scale)
            buffer.dones.append(done)
            buffer.log_probs.append(log_prob.squeeze(0))
            buffer.values.append(value.squeeze(0))
            completed_steps += 1
            episode_reward += float(reward)
            episode_length += 1
            for name, value in info.get("reward_components", {}).items():
                if name in episode_components:
                    episode_components[name] += float(value)
            observation = next_observation
            if done:
                rows.append(
                    {
                        "environment_steps": completed_steps,
                        "episodic_reward": episode_reward,
                        "episode_length": episode_length,
                        "kills": info.get("kills", 0),
                        "hits": info.get("hits", 0),
                        "health": info.get("health", np.nan),
                        "ammo": info.get("ammo", np.nan),
                        "damage_taken": info.get("damage_taken", 0),
                        "deaths": info.get("deaths", 0),
                        "survival_time": info.get("survival_time", episode_length),
                        "distance_travelled": info.get("distance_travelled", 0.0),
                        "episode_dopamine": episode_reward,
                        **{
                            f"reward_component_{name}": value
                            for name, value in episode_components.items()
                        },
                    }
                )
                observation, _ = env.reset()
                if recurrent:
                    state = algorithm.policy.initial_state(1, device)
                episode_reward = 0.0
                episode_length = 0
                episode_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        with torch.no_grad():
            tensor = observation_tensor(observation, device)
            if state is None:
                _, last_value = algorithm.policy(tensor)
            else:
                _, last_value, _ = algorithm.policy.forward_recurrent(tensor, state)
        losses = algorithm.update(buffer, last_value.squeeze(0))
        elapsed = max(time.perf_counter() - started, 1e-6)
        if not rows or rows[-1]["environment_steps"] != completed_steps:
            rows.append(
                {
                    "environment_steps": completed_steps,
                    "episodic_reward": episode_reward,
                    "episode_length": episode_length,
                    "kills": 0,
                    "hits": info.get("hits", 0),
                    "health": info.get("health", np.nan),
                    "ammo": info.get("ammo", np.nan),
                    "damage_taken": info.get("damage_taken", 0),
                    "deaths": info.get("deaths", 0),
                    "survival_time": episode_length,
                    "distance_travelled": 0.0,
                    "episode_dopamine": episode_reward,
                    **{
                        f"reward_component_{name}": value
                        for name, value in episode_components.items()
                    },
                }
            )
        rows[-1].update(losses)
        action_names = getattr(
            env, "action_names", tuple(f"action_{index}" for index in range(len(action_counts)))
        )
        for index, count in enumerate(action_counts):
            rows[-1][f"action_fraction_{action_names[index]}"] = float(count / action_counts.sum())
        rows[-1]["fps"] = completed_steps / elapsed
    return rows


def evaluate_detailed(
    env: gym.Env[np.ndarray, int],
    policy: torch.nn.Module,
    episodes: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        recurrent = hasattr(policy, "initial_state")
        state = policy.initial_state(1, device) if recurrent else None
        done = False
        total = 0.0
        final_info: dict[str, Any] = {}
        steps = 0
        while not done:
            with torch.no_grad():
                tensor = observation_tensor(observation, device)
                if state is None:
                    action, _, _ = policy.act(tensor, deterministic=True)
                else:
                    action, _, _, state = policy.act_recurrent(
                        tensor, state, deterministic=True
                    )
            observation, reward, terminated, truncated, final_info = env.step(
                int(action.item())
            )
            done = terminated or truncated
            total += float(reward)
            steps += 1
        results.append(
            {
                "reward": total,
                "kills": float(final_info.get("kills", 0.0)),
                "hits": float(final_info.get("hits", 0.0)),
                "damage_taken": float(final_info.get("damage_taken", 0.0)),
                "deaths": float(final_info.get("deaths", 0.0)),
                "survival_time": float(final_info.get("survival_time", steps)),
            }
        )
    return results


def evaluate(
    env: gym.Env[np.ndarray, int],
    policy: torch.nn.Module,
    episodes: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    """Compatibility wrapper returning only episode reward totals."""
    return [
        episode["reward"]
        for episode in evaluate_detailed(env, policy, episodes, seed, device)
    ]
