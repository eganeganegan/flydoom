"""Portable synchronized episode traces used by the activity player."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flydoom.env.actions import ACTION_NAMES


@dataclass(slots=True)
class EpisodeTrace:
    frames: np.ndarray
    activity: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    positions: np.ndarray
    body_ids: np.ndarray
    regions: np.ndarray
    types: np.ndarray
    fps: float = 20.0
    action_names: np.ndarray | None = None
    learning_mode: str = "unknown"
    health: np.ndarray | None = None
    ammo: np.ndarray | None = None
    kills: np.ndarray | None = None
    action_values: np.ndarray | None = None
    reward_components: np.ndarray | None = None
    reward_component_names: np.ndarray | None = None

    def __post_init__(self) -> None:
        steps = len(self.frames)
        if self.activity.ndim != 2 or len(self.activity) != steps:
            raise ValueError("activity must have shape [steps, neurons]")
        if len(self.actions) != steps or len(self.rewards) != steps:
            raise ValueError("frames, actions, rewards, and activity must be synchronized")
        neurons = self.activity.shape[1]
        if self.positions.shape != (neurons, 3):
            raise ValueError("positions must have shape [neurons, 3]")
        if any(len(values) != neurons for values in (self.body_ids, self.regions, self.types)):
            raise ValueError("node metadata must contain one item per activity column")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.action_names is None:
            self.action_names = np.asarray(ACTION_NAMES)
        else:
            self.action_names = np.asarray(self.action_names).astype(str)
        self.learning_mode = str(self.learning_mode)
        for name in ("health", "ammo", "kills"):
            values = getattr(self, name)
            if values is None:
                setattr(self, name, np.full(steps, np.nan, dtype=np.float32))
            else:
                values = np.asarray(values, dtype=np.float32)
                if len(values) != steps:
                    raise ValueError(f"{name} must contain one value per step")
                setattr(self, name, values)
        if self.action_values is None:
            self.action_values = np.zeros((steps, len(self.action_names)), dtype=np.float32)
        else:
            self.action_values = np.asarray(self.action_values, dtype=np.float32)
            if self.action_values.shape != (steps, len(self.action_names)):
                raise ValueError("action_values must have shape [steps, actions]")
        if self.reward_components is None:
            self.reward_components = np.zeros((steps, 0), dtype=np.float32)
        else:
            self.reward_components = np.asarray(self.reward_components, dtype=np.float32)
            if self.reward_components.ndim != 2 or len(self.reward_components) != steps:
                raise ValueError("reward_components must have shape [steps, components]")
        if self.reward_component_names is None:
            self.reward_component_names = np.asarray([], dtype=str)
        else:
            self.reward_component_names = np.asarray(self.reward_component_names).astype(str)
        if self.reward_components.shape[1] != len(self.reward_component_names):
            raise ValueError("reward component names must match reward component columns")

    def action_name(self, action: int) -> str:
        """Return the recorded environment-specific name for an action index."""
        assert self.action_names is not None
        if 0 <= action < len(self.action_names):
            return str(self.action_names[action])
        return str(action)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            frames=self.frames.astype(np.uint8),
            activity=self.activity.astype(np.float32),
            actions=self.actions.astype(np.int16),
            rewards=self.rewards.astype(np.float32),
            positions=self.positions.astype(np.float32),
            body_ids=self.body_ids.astype(str),
            regions=self.regions.astype(str),
            types=self.types.astype(str),
            fps=np.asarray(self.fps, dtype=np.float32),
            action_names=self.action_names.astype(str),
            learning_mode=np.asarray(self.learning_mode),
            health=self.health.astype(np.float32),
            ammo=self.ammo.astype(np.float32),
            kills=self.kills.astype(np.float32),
            action_values=self.action_values.astype(np.float32),
            reward_components=self.reward_components.astype(np.float32),
            reward_component_names=self.reward_component_names.astype(str),
        )

    @classmethod
    def load(cls, path: str | Path) -> EpisodeTrace:
        with np.load(path, allow_pickle=False) as payload:
            values = {
                name: payload[name]
                for name in (
                    "frames",
                    "activity",
                    "actions",
                    "rewards",
                    "positions",
                    "body_ids",
                    "regions",
                    "types",
                )
            }
            return cls(
                **values,
                fps=float(payload["fps"]),
                action_names=payload["action_names"] if "action_names" in payload else None,
                learning_mode=(
                    str(payload["learning_mode"]) if "learning_mode" in payload else "unknown"
                ),
                health=payload["health"] if "health" in payload else None,
                ammo=payload["ammo"] if "ammo" in payload else None,
                kills=payload["kills"] if "kills" in payload else None,
                action_values=payload["action_values"] if "action_values" in payload else None,
                reward_components=(
                    payload["reward_components"] if "reward_components" in payload else None
                ),
                reward_component_names=(
                    payload["reward_component_names"]
                    if "reward_component_names" in payload
                    else None
                ),
            )
