"""Portable synchronized episode traces used by the activity player."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        )

    @classmethod
    def load(cls, path: str | Path) -> EpisodeTrace:
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                **{
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
                },
                fps=float(payload["fps"]),
            )
