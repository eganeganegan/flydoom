"""Learning-curve plots with an explicit model comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_learning_curves(metrics: pd.DataFrame, output: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for model, group in metrics.groupby("model"):
        ordered = group.sort_values("environment_steps")
        smoothed = ordered["episodic_reward"].rolling(10, min_periods=1).mean()
        axis.plot(ordered["environment_steps"], smoothed, label=model, linewidth=2)
    axis.set(xlabel="Environment steps", ylabel="Episode reward", title="FlyDoom learning")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)
