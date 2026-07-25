#!/usr/bin/env python
"""Run the JMLC real-data experiment matrix from a template config.

    poetry run python scripts/run_jmlc_matrix.py \
        --config configs/jmlc/smoke.yaml \
        --output reports/jmlc_2026/smoke

The config is a single valid ExperimentSpec (see configs/jmlc/smoke.yaml); the
canonical 14-cell matrix is expanded from it. RunRecords are written under
<output>/runs and a summary to <output>/summary.json. Exit code is non-zero if
any cell failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rc_bench.runners.jmlc_matrix import run_matrix  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the JMLC experiment matrix")
    parser.add_argument("--config", required=True, help="template ExperimentSpec YAML")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="do not save per-cell prediction artifacts",
    )
    args = parser.parse_args()

    summary = run_matrix(
        args.config,
        args.output,
        save_predictions=not args.no_predictions,
    )

    n_fail = sum(1 for e in summary if e["status"] == "FAILED")
    n_ok = len(summary) - n_fail
    print(f"\nJMLC matrix: {n_ok} completed, {n_fail} failed, {len(summary)} cells")
    for e in summary:
        line = f"  {e['family']:<9} {e['model']:<20} h{e['horizon']:<2} {e['status']}"
        if e.get("nrmse_std") is not None:
            line += f"  NRMSE_std={e['nrmse_std']:.4f}"
        if e["status"] == "FAILED":
            line += f"  {e.get('error', '')}"
        print(line)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
