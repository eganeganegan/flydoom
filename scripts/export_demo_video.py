#!/usr/bin/env python3
"""Export synchronized 16:9 or 9:16 activity media."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--aspect", choices=("16:9", "9:16"), default="16:9")
    parser.add_argument("--frame", type=int, default=0, help="frame used for PNG export")
    parser.add_argument("--skeleton-dir")
    args = parser.parse_args()
    from flydoom.visualization.export import export_mp4, export_png
    from flydoom.visualization.trace import EpisodeTrace

    trace = EpisodeTrace.load(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".png":
        export_png(trace, args.output, args.frame, args.aspect, args.skeleton_dir)
    elif args.output.suffix.lower() == ".mp4":
        export_mp4(trace, args.output, args.aspect, args.skeleton_dir)
    else:
        raise SystemExit("Output extension must be .png or .mp4")


if __name__ == "__main__":
    main()
