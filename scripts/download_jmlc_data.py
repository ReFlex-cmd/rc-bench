#!/usr/bin/env python3
"""Download the pinned JMLC UCI dataset into the ignored raw-data directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rc_bench.data.download import DatasetDownloadError, download_dataset


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "jmlc" / "dataset_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and SHA-256 verify the UCI Individual Household Electric "
            "Power Consumption dataset."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"pinned dataset manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"ignored raw-data directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="per-request network timeout in seconds (default: 120)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        raw_path = download_dataset(
            args.manifest,
            args.output_dir,
            timeout_seconds=args.timeout_seconds,
        )
    except (DatasetDownloadError, ValueError) as exc:
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Verified raw dataset: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
