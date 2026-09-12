"""Environment interaction and experiment metric collection."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from flydoom.training.ppo import PPO
from flydoom.training.rollout import RolloutBuffer


def observation_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(observation, device=device).unsqueeze(0)


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
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    completed_steps = 0
    while completed_steps < total_steps:
        buffer = RolloutBuffer()
        for _ in range(min(rollout_steps, total_steps - completed_steps)):
            tensor = observation_tensor(observation, device)
            with torch.no_grad():
                action, log_prob, value = algorithm.policy.act(tensor)
            next_observation, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated
            buffer.observations.append(tensor)
            buffer.actions.append(action.squeeze(0))
            buffer.rewards.append(float(reward))
            buffer.dones.append(done)
            buffer.log_probs.append(log_prob.squeeze(0))
            buffer.values.append(value.squeeze(0))
            completed_steps += 1
            episode_reward += float(reward)
            episode_length += 1
            observation = next_observation
            if done:
                rows.append(
                    {
                        "environment_steps": completed_steps,
                        "episodic_reward": episode_reward,
                        "episode_length": episode_length,
                        "kills": info.get("kills", 0),
                        "survival_time": info.get("survival_time", episode_length),
                        "distance_travelled": info.get("distance_travelled", 0.0),
                    }
                )
                observation, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0
        with torch.no_grad():
            _, last_value = algorithm.policy(observation_tensor(observation, device))
        losses = algorithm.update(buffer, last_value.squeeze(0))
        elapsed = max(time.perf_counter() - started, 1e-6)
        if not rows or rows[-1]["environment_steps"] != completed_steps:
            rows.append(
                {
                    "environment_steps": completed_steps,
                    "episodic_reward": episode_reward,
                    "episode_length": episode_length,
                    "kills": 0,
                    "survival_time": episode_length,
                    "distance_travelled": 0.0,
                }
            )
        rows[-1].update(losses)
        rows[-1]["fps"] = completed_steps / elapsed
    return rows


def evaluate(
    env: gym.Env[np.ndarray, int],
    policy: torch.nn.Module,
    episodes: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    rewards: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                action, _, _ = policy.act(
                    observation_tensor(observation, device), deterministic=True
                )
            observation, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            total += float(reward)
        rewards.append(total)
    return rewards
