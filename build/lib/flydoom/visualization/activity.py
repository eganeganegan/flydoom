"""Region-level activity heatmaps for controller diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_region_activity(activity: np.ndarray, region_names: list[str], output: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    image = axis.imshow(activity.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axis.set(xlabel="Controller step", ylabel="Region", yticks=range(len(region_names)))
    axis.set_yticklabels(region_names)
    figure.colorbar(image, ax=axis, label="Mean activation")
    figure.savefig(output, dpi=180)
    plt.close(figure)
