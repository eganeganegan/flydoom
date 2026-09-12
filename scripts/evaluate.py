#!/usr/bin/env python3
"""Inspect evaluation artifacts produced by a training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.run_dir.glob("*/evaluation.json")):
        evaluation = json.loads(path.read_text())
        rows.append({"model": path.parent.name, **evaluation})
    if not rows:
        raise SystemExit(f"No evaluation artifacts found under {args.run_dir}")
    print(pd.DataFrame(rows).to_string(index=False))
