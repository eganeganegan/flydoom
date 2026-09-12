"""Effect-focused summaries for multi-seed learning experiments."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_mean_ci(
    values: np.ndarray, confidence: float = 0.95, samples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(np.quantile(means, [alpha, 1.0 - alpha]))


def compare_samples(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Report robust test results and rank-biserial effect size, not p-values alone."""
    statistic, p_value = stats.mannwhitneyu(first, second, alternative="two-sided")
    effect = 2.0 * statistic / (len(first) * len(second)) - 1.0
    return {
        "first_mean": float(np.mean(first)),
        "second_mean": float(np.mean(second)),
        "mann_whitney_u": float(statistic),
        "p_value": float(p_value),
        "rank_biserial_effect": float(effect),
    }
