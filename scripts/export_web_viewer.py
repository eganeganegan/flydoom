#!/usr/bin/env python3
"""Export a FlyDoom trace and graph for the WebGL activity viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from flydoom.data.schema import ConnectomeGraph
from flydoom.visualization.trace import EpisodeTrace
from flydoom.visualization.web_export import export_web_session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="episode_trace.npz from training")
    parser.add_argument(
        "--graph", type=Path, default=Path("data/processed/visual-descending.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("viewer/public/session"))
    parser.add_argument("--skeleton-dir", type=Path)
    parser.add_argument("--max-skeleton-segments", type=int, default=2_000_000)
    parser.add_argument("--title", default="FlyDoom neural activity")
    args = parser.parse_args()
    payload = torch.load(args.graph, map_location="cpu", weights_only=True)
    graph = ConnectomeGraph(pd.DataFrame(payload["neurons"]), pd.DataFrame(payload["edges"]))
    manifest = export_web_session(
        EpisodeTrace.load(args.trace),
        graph,
        args.output,
        skeleton_dir=args.skeleton_dir,
        max_skeleton_segments=args.max_skeleton_segments,
        title=args.title,
    )
    print(f"Web viewer session written to {manifest}")


if __name__ == "__main__":
    main()
