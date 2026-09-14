"""Gymnasium boundary for a lightweight fallback task and optional VizDoom."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from flydoom.env.actions import DoomAction, action_count
from flydoom.env.rewards import REWARD_COMPONENT_NAMES, RewardSignals, RewardWeights

SCENARIO_CONTROLS = {
    "basic": (
        ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"),
        ("no_op", "move_left", "move_right", "shoot"),
    ),
    "defend_the_center": (
        ("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
        ("no_op", "turn_left", "turn_right", "shoot"),
    ),
}
DEFAULT_CONTROLS = (
    ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "ATTACK"),
    ("no_op", "move_forward", "turn_left", "turn_right", "shoot"),
)


class MockDoomEnv(gym.Env[np.ndarray, int]):
    """Fast visual aiming task used for tests and installation-free smoke runs."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size: int = 42, max_steps: int = 96) -> None:
        self.size = size
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(0, 255, (size, size, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(action_count())
        self.action_names = tuple(action.name.lower() for action in DoomAction)
        self.heading = 0.0
        self.target = 0.0
        self.steps = 0
        self.kills = 0
        self.hits = 0
        self.ammo = 100
        self.episode_dopamine = 0.0
        self.reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.heading = float(self.np_random.uniform(-1.0, 1.0))
        self.target = float(self.np_random.uniform(-1.0, 1.0))
        self.steps = 0
        self.kills = 0
        self.hits = 0
        self.ammo = 100
        self.episode_dopamine = 0.0
        self.reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
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
            self.ammo = max(self.ammo - 1, 0)
            error = abs(self.target - self.heading)
            if error < 0.12:
                reward += 1.0
                self.kills += 1
                self.hits += 1
                self.target = float(self.np_random.uniform(-1.0, 1.0))
            else:
                reward -= 0.03
        self.heading = float(np.clip(self.heading, -1.2, 1.2))
        self.reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        self.reward_components["native"] = reward
        self.episode_dopamine += reward
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
        return {
            "health": 100.0,
            "ammo": float(self.ammo),
            "kills": float(self.kills),
            "hits": float(self.hits),
            "damage_taken": 0.0,
            "deaths": 0.0,
            "survival_time": self.steps,
            "distance_travelled": 0.0,
            "dopamine": float(sum(self.reward_components.values())),
            "episode_dopamine": self.episode_dopamine,
            "reward_components": self.reward_components.copy(),
        }

    def render(self) -> np.ndarray:
        return self._frame()


class VizDoomEnv(gym.Env[np.ndarray, int]):
    """Small discrete-action boundary around VizDoom's native API."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        scenario: str = "basic",
        size: int = 84,
        max_steps: int = 2_100,
        frame_skip: int = 4,
        render_mode: str = "rgb_array",
        reward_weights: RewardWeights | None = None,
    ) -> None:
        try:
            import vizdoom as vzd
        except ImportError as exc:
            raise RuntimeError("Install VizDoom: pip install -e '.[doom]'") from exc
        scenario = {
            "VizdoomBasic-v0": "basic",
            "VizdoomDefendCenter-v0": "defend_the_center",
        }.get(scenario, scenario)
        config_path = Path(vzd.scenarios_path) / f"{scenario}.cfg"
        if not config_path.is_file():
            raise ValueError(f"Unknown or unavailable VizDoom scenario: {scenario}")
        self.vzd = vzd
        self.game = vzd.DoomGame()
        self.game.load_config(str(config_path))
        button_names, self.action_names = SCENARIO_CONTROLS.get(scenario, DEFAULT_CONTROLS)
        self.buttons = tuple(getattr(vzd.Button, name) for name in button_names)
        self.game.clear_available_buttons()
        for button in self.buttons:
            self.game.add_available_button(button)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_window_visible(render_mode == "human")
        self.game.init()
        self.size = size
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.reward_weights = reward_weights or RewardWeights()
        self.steps = 0
        self.episode_dopamine = 0.0
        self.game_stats: dict[str, float] = {}
        self.last_frame = np.zeros((size, size, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(0, 255, (size, size, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(len(self.action_names))

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        y = np.linspace(0, frame.shape[0] - 1, self.size).astype(int)
        x = np.linspace(0, frame.shape[1] - 1, self.size).astype(int)
        return frame[y][:, x]

    def _observation(self) -> np.ndarray:
        state = self.game.get_state()
        if state is not None:
            self.last_frame = self._resize(np.asarray(state.screen_buffer))
        return self.last_frame.copy()

    def _game_variable(self, name: str, default: float = 0.0) -> float:
        variable = getattr(self.vzd.GameVariable, name, None)
        if variable is None:
            return default
        try:
            return float(self.game.get_game_variable(variable))
        except RuntimeError:
            return default

    def _game_stats(self) -> dict[str, float]:
        return {
            "health": self._game_variable("HEALTH", 100.0),
            "ammo": self._game_variable("AMMO2", 0.0),
            "kills": self._game_variable("KILLCOUNT"),
            "hits": self._game_variable("HITCOUNT"),
            "damage_taken": self._game_variable("DAMAGE_TAKEN"),
            "deaths": self._game_variable("DEATHCOUNT"),
        }

    def _info(
        self,
        stats: dict[str, float] | None = None,
        components: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        stats = stats or self._game_stats()
        components = components or {}
        return {
            **stats,
            "survival_time": self.game.get_episode_time(),
            "distance_travelled": 0.0,
            "dopamine": float(sum(components.values())),
            "episode_dopamine": self.episode_dopamine,
            "reward_components": components,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.steps = 0
        self.episode_dopamine = 0.0
        self.game_stats = self._game_stats()
        return self._observation(), self._info(self.game_stats)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        command = [False] * len(self.buttons)
        if action:
            command[action - 1] = True
        native_reward = float(self.game.make_action(command, self.frame_skip))
        self.steps += 1
        terminated = self.game.is_episode_finished()
        truncated = self.steps >= self.max_steps and not terminated
        current = self._game_stats()
        previous = self.game_stats
        deaths = max(current["deaths"] - previous["deaths"], 0.0)
        if terminated and deaths == 0 and current["health"] <= 0 < previous["health"]:
            deaths = 1.0
        signals = RewardSignals(
            native=native_reward,
            kills=max(current["kills"] - previous["kills"], 0.0),
            hits=max(current["hits"] - previous["hits"], 0.0),
            damage_taken=max(current["damage_taken"] - previous["damage_taken"], 0.0),
            deaths=deaths,
            ammo_spent=max(previous["ammo"] - current["ammo"], 0.0),
            health_gained=max(current["health"] - previous["health"], 0.0),
        )
        reward, components = self.reward_weights.score(signals)
        self.episode_dopamine += reward
        self.game_stats = current
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            self._info(current, components),
        )

    def render(self) -> np.ndarray:
        return self.last_frame.copy()

    def close(self) -> None:
        self.game.close()


def make_environment(config: dict[str, Any]) -> gym.Env[np.ndarray, int]:
    """Create the configured environment, falling back only when explicitly requested."""
    backend = config.get("backend", "mock")
    if backend == "mock":
        return MockDoomEnv(config.get("size", 42), config.get("max_episode_steps", 96))
    if backend != "vizdoom":
        raise ValueError(f"Unknown environment backend: {backend}")
    return VizDoomEnv(
        scenario=config.get("scenario", "basic"),
        size=int(config.get("size", 84)),
        max_steps=int(config.get("max_episode_steps", 2_100)),
        frame_skip=int(config.get("frame_skip", 4)),
        render_mode=str(config.get("render_mode", "rgb_array")),
        reward_weights=RewardWeights.from_mapping(config.get("reward")),
    )
