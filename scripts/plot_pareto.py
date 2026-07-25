#!/usr/bin/env python
"""Render the quality-latency and quality-memory Pareto figures — PLOT-001.

    poetry run python scripts/plot_pareto.py \
        --bundle reports/jmlc_2026 --output reports/jmlc_2026/plots

Reads the published table (`aggregates/matrix_table.json`) and the resource
profiles (`profiles/summary.json`) and joins them on the config hash, so a
figure can only be produced from a profile measured on the configuration that
was actually published. Run scripts/validate_evidence.py first if either input
may be stale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rc_bench.reporting.pareto import generate_pareto_plots  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the JMLC Pareto figures")
    parser.add_argument("--bundle", default="reports/jmlc_2026", help="evidence bundle directory")
    parser.add_argument(
        "--output",
        default=None,
        help="output directory for the figures (default: <bundle>/plots)",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path(args.bundle) / "plots"
    try:
        written = generate_pareto_plots(args.bundle, output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot render Pareto figures: {exc}", file=sys.stderr)
        return 1

    for cost, path in written.items():
        print(f"wrote {path}  ({cost})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
