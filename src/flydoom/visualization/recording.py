"""Record controller state in lockstep with observations and environment outcomes."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from flydoom.data.schema import ConnectomeGraph
from flydoom.training.trainer import observation_tensor
from flydoom.visualization.layout import graph_positions
from flydoom.visualization.trace import EpisodeTrace


def record_episode(
    env: gym.Env[np.ndarray, int],
    policy: torch.nn.Module,
    graph: ConnectomeGraph,
    *,
    device: torch.device,
    seed: int,
    fps: float = 20.0,
) -> EpisodeTrace:
    if not hasattr(policy, "forward_with_activity"):
        raise TypeError("Activity recording requires a connectome policy")
    observation, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    activity: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    state = policy.initial_state(1, device)
    done = False
    while not done:
        tensor = observation_tensor(observation, device)
        with torch.no_grad():
            logits, _, state = policy.forward_with_activity(tensor, state)
            action = int(logits.argmax(dim=-1).item())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        frames.append(np.asarray(observation).copy())
        activity.append(state.squeeze(0).detach().cpu().numpy())
        actions.append(action)
        rewards.append(float(reward))
        observation = next_observation
        done = terminated or truncated
    neurons = graph.neurons
    return EpisodeTrace(
        np.asarray(frames),
        np.asarray(activity),
        np.asarray(actions),
        np.asarray(rewards),
        graph_positions(graph),
        neurons["body_id"].astype(str).to_numpy(),
        neurons["region"].fillna("unknown").astype(str).to_numpy(),
        neurons["type"].fillna("unknown").astype(str).to_numpy(),
        fps,
        np.asarray(getattr(env, "action_names", ())),
    )
