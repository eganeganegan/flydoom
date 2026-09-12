"""Configurable reward accounting shared by environment adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardWeights:
    living: float = -0.002
    kill: float = 1.0
    missed_shot: float = -0.03
    forward_motion: float = 0.01

    def total(
        self, *, killed: bool = False, missed: bool = False, moved_forward: bool = False
    ) -> float:
        return (
            self.living
            + self.kill * killed
            + self.missed_shot * missed
            + self.forward_motion * moved_forward
        )
