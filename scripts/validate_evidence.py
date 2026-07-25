#!/usr/bin/env python
"""Validate (and regenerate) the JMLC evidence bundle — EVID-001.

    poetry run python scripts/validate_evidence.py reports/jmlc_2026
    poetry run python scripts/validate_evidence.py reports/jmlc_2026 --write

Without ``--write`` nothing is modified: the bundle is checked and every problem
is printed, exit code 1 if any were found. This is the form `verify.sh release`
runs, so a bundle whose published table has drifted from its raw RunRecords —
or whose cells disagree on the dataset, the evaluation context or the HPO
budget — cannot pass the release gate.

Every protocol mode the bundle publishes (``fair/runs``, ``best_effort/runs``)
is checked separately, against its own aggregate table; ``--modes`` narrows the
set. Only the fair matrix must carry resource profiles — best-effort changes the
hyper-parameters, so reusing the fair profile for it would be a forgery.

``--write`` regenerates ``aggregates/matrix_table*.{json,csv}`` from the raw
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
    discover_modes,
    load_cells,
    validate_bundle,
    validate_bundle_modes,
    write_aggregates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the JMLC evidence bundle")
    parser.add_argument("bundle", help="bundle directory, e.g. reports/jmlc_2026")
    parser.add_argument(
        "--runs-subdir",
        default=None,
        help=(
            "validate a single RunRecord directory instead of every mode "
            "(e.g. fair/runs); mutually exclusive with --modes. A '<mode>/runs' "
            "path is still checked against that mode"
        ),
    )
    parser.add_argument(
        "--modes",
        default=None,
        help=(
            "comma-separated protocol modes to validate (default: every mode "
            "that has a runs/ directory in the bundle)"
        ),
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

    if args.runs_subdir is not None and args.modes is not None:
        parser.error("--runs-subdir validates one directory; it cannot be combined with --modes")

    bundle = Path(args.bundle)

    # Один явный каталог — узкий режим для отладки и для бандлов со старой
    # раскладкой: гейт при этом ничего не знает о режимах и профили требует
    # безусловно.
    if args.runs_subdir is not None:
        runs_dirs = [bundle / args.runs_subdir]
    else:
        modes = (
            [mode.strip() for mode in args.modes.split(",") if mode.strip()]
            if args.modes is not None
            else discover_modes(bundle)
        )
        if not modes:
            print(f"no protocol modes found under {bundle}", file=sys.stderr)
            return 1
        runs_dirs = [bundle / mode / "runs" for mode in modes]

    if args.write:
        for runs_dir in runs_dirs:
            if not runs_dir.is_dir():
                print(f"cannot regenerate aggregates: missing {runs_dir}", file=sys.stderr)
                return 1
            rows = build_rows(load_cells(runs_dir), runs_dir)
            written = write_aggregates(rows, bundle / "aggregates")
            print(f"wrote {written['json']}")
            print(f"wrote {written['csv']}")

    if args.runs_subdir is not None:
        # Даже в узкой форме каталог вида "<mode>/runs" называет режим, и
        # сверку записей с этим именем терять незачем: иначе документированная
        # отладочная форма оказывается обходным путём вокруг проверки.
        head = args.runs_subdir.strip("/").split("/")[0]
        problems = validate_bundle(
            bundle,
            runs_subdir=args.runs_subdir,
            expected_cells=args.expected_cells,
            require_profiles=not args.no_profiles,
            expected_mode=head if head in ("fair", "best_effort") else None,
        )
        scope = args.runs_subdir
    else:
        problems = validate_bundle_modes(
            bundle,
            modes=modes,
            expected_cells=args.expected_cells,
            require_profiles=not args.no_profiles,
        )
        scope = ", ".join(modes)

    if problems:
        print(f"\nEvidence bundle {bundle}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"Evidence bundle {bundle}: OK ({scope}; {args.expected_cells} cells each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
