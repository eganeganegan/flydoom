from __future__ import annotations

import numpy as np

from flydoom.env.rewards import RewardSignals, RewardWeights


def test_dopamine_reward_is_event_based_and_inspectable() -> None:
    score, components = RewardWeights().score(
        RewardSignals(
            native=100.0,
            kills=1.0,
            hits=1.0,
            damage_taken=10.0,
            ammo_spent=1.0,
        )
    )

    assert components["native"] == 0.0
    assert components["kill"] == 10.0
    assert components["damage_taken"] == -0.5
    assert np.isclose(score, 9.72)
