"""Configurable, inspectable reward accounting for Doom environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

REWARD_COMPONENT_NAMES = (
    "native",
    "living",
    "kill",
    "hit",
    "damage_taken",
    "death",
    "ammo_spent",
    "health_gained",
)


@dataclass(frozen=True, slots=True)
class RewardSignals:
    """Events observed over one agent step."""

    native: float = 0.0
    kills: float = 0.0
    hits: float = 0.0
    damage_taken: float = 0.0
    deaths: float = 0.0
    ammo_spent: float = 0.0
    health_gained: float = 0.0


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Weights for a dopamine-like scalar reinforcement signal.

    The defaults replace scenario-native rewards with transparent shaping. Values are
    deliberately expressed in readable, pre-``training.reward_scale`` units.
    """

    native_scale: float = 0.0
    living: float = -0.01
    kill: float = 10.0
    hit: float = 0.25
    damage_taken: float = -0.05
    death: float = -5.0
    ammo_spent: float = -0.02
    health_gained: float = 0.02

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> RewardWeights:
        if not values:
            return cls()
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(valid))
        if unknown:
            raise ValueError(f"Unknown reward weights: {', '.join(unknown)}")
        return cls(**{name: float(value) for name, value in values.items()})

    def components(self, signals: RewardSignals) -> dict[str, float]:
        """Return every weighted contribution, including zero-valued events."""
        return {
            "native": self.native_scale * signals.native,
            "living": self.living,
            "kill": self.kill * signals.kills,
            "hit": self.hit * signals.hits,
            "damage_taken": self.damage_taken * signals.damage_taken,
            "death": self.death * signals.deaths,
            "ammo_spent": self.ammo_spent * signals.ammo_spent,
            "health_gained": self.health_gained * signals.health_gained,
        }

    def score(self, signals: RewardSignals) -> tuple[float, dict[str, float]]:
        components = self.components(signals)
        return float(sum(components.values())), components
