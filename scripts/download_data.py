#!/usr/bin/env python3
"""List or download official HHMI Janelia Male CNS v1.0 Feather files."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from flydoom.data.male_cns import BULK_ROOT, OPTIONAL_FILES, REQUIRED_FILES

HTTPS_ROOT = BULK_ROOT.replace("gs://", "https://storage.googleapis.com/", 1)


def download(name: str, output: Path, *, force: bool = False) -> None:
    target = output / name
    if target.exists() and not force:
        print(f"exists: {target}")
        return
    partial = target.with_suffix(target.suffix + ".part")
    print(f"downloading {HTTPS_ROOT}{name}")
    try:
        with urllib.request.urlopen(HTTPS_ROOT + name) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", choices=("none", "required", "all"), default="none")
    parser.add_argument("--output", type=Path, default=Path("data/raw/male-cns-v1.0"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(f"Official bucket: {BULK_ROOT}")
    print("Required files:")
    for name in REQUIRED_FILES:
        print(f"  {name}")
    print("Optional richer files:")
    for name in OPTIONAL_FILES:
        print(f"  {name}")
    if args.download == "none":
        print("Use --download required (several GB) or copy with gcloud storage cp.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    selected = REQUIRED_FILES + (OPTIONAL_FILES if args.download == "all" else ())
    for name in selected:
        download(name, args.output, force=args.force)


if __name__ == "__main__":
    main()
