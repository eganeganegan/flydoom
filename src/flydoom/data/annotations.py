"""Explicit, configurable interpretation of biological annotations."""

from __future__ import annotations

import pandas as pd
import torch

DEFAULT_TRANSMITTER_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": 0.0,
    "dopamine": 0.0,
    "serotonin": 0.0,
    "octopamine": 0.0,
}


def transmitter_signs(edges: pd.DataFrame, mapping: dict[str, float] | None = None) -> torch.Tensor:
    """Map annotations to signs; unknown/context-dependent transmitters remain unsigned."""
    signs = mapping or DEFAULT_TRANSMITTER_SIGN
    return torch.tensor(
        [signs.get(str(value).lower(), 0.0) for value in edges["neurotransmitter"]],
        dtype=torch.float32,
    )
