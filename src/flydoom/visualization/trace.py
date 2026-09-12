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
            )
