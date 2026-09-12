#!/usr/bin/env python3
"""Open the interactive synchronized FlyDoom activity player."""

from __future__ import annotations

import argparse

from flydoom.visualization.player import ActivityPlayer
from flydoom.visualization.trace import EpisodeTrace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="episode .npz written by training/evaluation")
    parser.add_argument("--region")
    parser.add_argument("--type", dest="type_")
    parser.add_argument("--skeleton-dir")
    args = parser.parse_args()
    ActivityPlayer(
        EpisodeTrace.load(args.trace),
        region=args.region,
        type_=args.type_,
        skeleton_dir=args.skeleton_dir,
    ).show()


if __name__ == "__main__":
    main()
