"""Gymnasium boundary for a lightweight fallback task and optional VizDoom."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from flydoom.env.actions import DoomAction, action_count


class MockDoomEnv(gym.Env[np.ndarray, int]):
    """Fast visual aiming task used for tests and installation-free smoke runs."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size: int = 42, max_steps: int = 96) -> None:
        self.size = size
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(0, 255, (size, size, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(action_count())
        self.heading = 0.0
        self.target = 0.0
        self.steps = 0
        self.kills = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.heading = float(self.np_random.uniform(-1.0, 1.0))
        self.target = float(self.np_random.uniform(-1.0, 1.0))
        self.steps = 0
        self.kills = 0
        return self._frame(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        reward = -0.002
        if action == DoomAction.TURN_LEFT:
            self.heading -= 0.12
        elif action == DoomAction.TURN_RIGHT:
            self.heading += 0.12
        elif action == DoomAction.MOVE_FORWARD:
            reward += 0.01
        elif action == DoomAction.SHOOT:
            error = abs(self.target - self.heading)
            if error < 0.12:
                reward += 1.0
                self.kills += 1
                self.target = float(self.np_random.uniform(-1.0, 1.0))
            else:
                reward -= 0.03
        self.heading = float(np.clip(self.heading, -1.2, 1.2))
        return self._frame(), reward, False, self.steps >= self.max_steps, self._info()

    def _frame(self) -> np.ndarray:
        frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        frame[:] = np.linspace(20, 60, self.size, dtype=np.uint8)[:, None, None]
        relative = np.clip((self.target - self.heading + 1.2) / 2.4, 0.0, 1.0)
        column = int(relative * (self.size - 1))
        frame[self.size // 3 : 2 * self.size // 3, max(0, column - 1) : column + 2] = (230, 45, 35)
        frame[-4:, :] = (40, 90, 40)
        return frame

    def _info(self) -> dict[str, Any]:
        return {"kills": self.kills, "survival_time": self.steps, "distance_travelled": 0.0}

    def render(self) -> np.ndarray:
        return self._frame()


def make_environment(config: dict[str, Any]) -> gym.Env[np.ndarray, int]:
    """Create the configured environment, falling back only when explicitly requested."""
    backend = config.get("backend", "mock")
    if backend == "mock":
        return MockDoomEnv(config.get("size", 42), config.get("max_episode_steps", 96))
    if backend != "vizdoom":
        raise ValueError(f"Unknown environment backend: {backend}")
    try:
        from vizdoom import gymnasium_wrapper  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "VizDoom is not installed. Run `pip install -e '.[doom]'` or set env.backend=mock."
        ) from exc
    scenario = config.get("scenario", "VizdoomBasic-v0")
    return gym.make(scenario, render_mode="rgb_array")
