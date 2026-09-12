"""Sparse graph summaries suitable for large or sampled subnetworks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from flydoom.data.schema import ConnectomeGraph  # noqa: E402


def plot_degree_distribution(graph: ConnectomeGraph, output: str | Path) -> None:
    degree = graph.edges.pre_body_id.value_counts().add(
        graph.edges.post_body_id.value_counts(), fill_value=0
    )
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.hist(degree, bins=50, color="#287271", alpha=0.85)
    axis.set(xlabel="Total degree", ylabel="Neurons", title="Connectome degree distribution")
    figure.savefig(output, dpi=180)
    plt.close(figure)
