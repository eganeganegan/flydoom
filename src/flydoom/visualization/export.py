"""Presentation/social composite image and MP4 export."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from flydoom.env.actions import ACTION_NAMES  # noqa: E402
from flydoom.visualization.player import render_brain_image  # noqa: E402
from flydoom.visualization.trace import EpisodeTrace  # noqa: E402

ASPECTS = {"16:9": (1280, 720), "9:16": (720, 1280)}


def compose_frame(
    trace: EpisodeTrace,
    index: int,
    aspect: str = "16:9",
    skeleton_dir: str | Path | None = None,
) -> np.ndarray:
    """Compose a synchronized Doom/brain/status frame."""
    if aspect not in ASPECTS:
        raise ValueError("aspect must be '16:9' or '9:16'")
    width, height = ASPECTS[aspect]
    brain_height = int(height * (0.58 if aspect == "16:9" else 0.55))
    brain = render_brain_image(trace, index, size=(width, brain_height), skeleton_dir=skeleton_dir)
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#020b0d")
    ratios = (0.40, 0.52, 0.08) if aspect == "16:9" else (0.32, 0.60, 0.08)
    grid = figure.add_gridspec(3, 1, height_ratios=ratios, hspace=0.025)
    for row, image in enumerate((trace.frames[index], brain)):
        axis = figure.add_subplot(grid[row])
        axis.imshow(image)
        axis.axis("off")
    footer = figure.add_subplot(grid[2])
    footer.set_facecolor("#020b0d")
    footer.axis("off")
    action = int(trace.actions[index])
    action_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else str(action)
    footer.text(
        0.02,
        0.5,
        f"t={index / trace.fps:6.2f}s     action={action_name}     "
        f"reward={trace.rewards[index]:+.3f}",
        color="#d5fcff",
        fontsize=13,
        family="monospace",
        va="center",
    )
    figure.canvas.draw()
    result = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return result


def export_png(
    trace: EpisodeTrace,
    output: str | Path,
    index: int = 0,
    aspect: str = "16:9",
    skeleton_dir: str | Path | None = None,
) -> None:
    plt.imsave(output, compose_frame(trace, index, aspect, skeleton_dir))


def export_mp4(
    trace: EpisodeTrace,
    output: str | Path,
    aspect: str = "16:9",
    skeleton_dir: str | Path | None = None,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Install the visualization extra: pip install -e '.[viz]'") from exc
    with imageio.get_writer(output, fps=trace.fps, codec="libx264", quality=8) as writer:
        for index in range(len(trace.frames)):
            writer.append_data(compose_frame(trace, index, aspect, skeleton_dir))
