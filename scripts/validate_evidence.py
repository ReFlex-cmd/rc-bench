#!/usr/bin/env python
"""Validate (and regenerate) the JMLC evidence bundle — EVID-001.

    poetry run python scripts/validate_evidence.py reports/jmlc_2026
    poetry run python scripts/validate_evidence.py reports/jmlc_2026 --write

Without ``--write`` nothing is modified: the bundle is checked and every problem
is printed, exit code 1 if any were found. This is the form `verify.sh release`
runs, so a bundle whose published table has drifted from its raw RunRecords —
or whose cells disagree on the dataset, the evaluation context or the HPO
budget — cannot pass the release gate.

``--write`` regenerates ``aggregates/matrix_table.{json,csv}`` from the raw
records first, then validates. Use it after a matrix run; never to make a
failing check go away.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rc_bench.reporting.evidence import (  # noqa: E402
    build_rows,
    load_cells,
    validate_bundle,
    write_aggregates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the JMLC evidence bundle")
    parser.add_argument("bundle", help="bundle directory, e.g. reports/jmlc_2026")
    parser.add_argument(
        "--runs-subdir",
        default="fair/runs",
        help="matrix RunRecord directory inside the bundle (default: fair/runs)",
    )
    parser.add_argument(
        "--expected-cells",
        type=int,
        default=14,
        help="number of matrix cells the bundle must publish (default: 14)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the aggregate table from the raw records before validating",
    )
    parser.add_argument(
        "--no-profiles",
        action="store_true",
        help="skip the resource-profile checks (bundle is not release-ready)",
    )
    args = parser.parse_args()

    bundle = Path(args.bundle)
    runs_dir = bundle / args.runs_subdir

    if args.write:
        if not runs_dir.is_dir():
            print(f"cannot regenerate aggregates: missing {runs_dir}", file=sys.stderr)
            return 1
        rows = build_rows(load_cells(runs_dir), runs_dir)
        written = write_aggregates(rows, bundle / "aggregates")
        print(f"wrote {written['json']}")
        print(f"wrote {written['csv']}")

    problems = validate_bundle(
        bundle,
        runs_subdir=args.runs_subdir,
        expected_cells=args.expected_cells,
        require_profiles=not args.no_profiles,
    )

    if problems:
        print(f"\nEvidence bundle {bundle}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"Evidence bundle {bundle}: OK ({args.expected_cells} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
