#!/usr/bin/env python
"""Profile latency/memory/size for the JMLC experiment-matrix cells (PROF-003).

Separate pass from the matrix itself: reads the already-written RunRecords
under ``--runs``, rebuilds and fits each cell's exact resolved configuration,
and measures single-step inference latency, isolated peak RSS, and
serialized-model/working-state sizes.

    poetry run python scripts/run_profiling.py \\
        --config configs/jmlc/smoke.yaml \\
        --runs reports/jmlc_2026/smoke/runs \\
        --output reports/jmlc_2026/profiles \\
        --hardware-output reports/jmlc_2026/hardware_profile.json

Writes one sanitized JSON per cell to ``--output`` plus a ``summary.json``,
and a sanitized hardware/software profile to ``--hardware-output``. Exit code
is non-zero if any cell failed to profile (never silently dropped).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rc_bench.profiling.model_profiles import run_profiling_pass  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile latency/memory/size for JMLC experiment-matrix cells"
    )
    parser.add_argument("--config", required=True, help="template ExperimentSpec YAML")
    parser.add_argument("--runs", required=True, help="directory of RunRecord JSONs to profile")
    parser.add_argument("--output", required=True, help="output directory for per-cell profiles")
    parser.add_argument("--hardware-output", required=True, help="output path for the hardware profile")
    parser.add_argument("--n-steps", type=int, default=1000, help="measured inference steps per cell")
    parser.add_argument("--warmup", type=int, default=100, help="untimed warmup steps per cell")
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="optional filter: run-file names or stems (e.g. reservoir_esn_h1)",
    )
    args = parser.parse_args()

    summary = run_profiling_pass(
        config_path=args.config,
        runs_dir=args.runs,
        output_dir=args.output,
        hardware_output_path=args.hardware_output,
        n_steps=args.n_steps,
        warmup=args.warmup,
        cells=args.cells,
    )

    n_fail = sum(1 for e in summary if e["status"] == "FAILED")
    n_ok = len(summary) - n_fail
    print(f"\nJMLC profiling: {n_ok} completed, {n_fail} failed, {len(summary)} cells")
    for e in summary:
        line = f"  {e['family']:<9} {e['model']:<20} h{e['horizon']:<2} {e['status']}"
        if e["status"] == "completed":
            line += f"  p50={e['p50_ns']:.0f}ns p95={e['p95_ns']:.0f}ns"
        else:
            line += f"  {e.get('error', '')}"
        print(line)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
