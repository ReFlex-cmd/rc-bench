"""Stage 4 smoke tests: HPO with 5 trials + multi_seed with 2 seeds per (model, task).

Per ТЗ §4.1: target time per cell < 10 min, just to verify nothing crashes
and to get rough NRMSE order-of-magnitudes.

Cell selection follows the priority matrix in ТЗ §4.2:
    P0: ESN, Leaky ESN  → all 4 tasks
    P1: Deep ESN        → all 4 tasks
    P1: LSM, FHN        → NARMA-10, Mackey-Glass
    P2: Logistic        → NARMA-10, Mackey-Glass
    P3: QRC             → NARMA-10
Lorenz-63 runs in closed_loop mode (decision Q3) for ESN/Leaky/Deep ESN only.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.schema import (
    DatasetSpec, ExperimentSpec, ProtocolSpec,
    ReadoutSpec, ReservoirSpec,
)
from rc_bench.runners.pipeline import run_pipeline


# (model, task, priority) — priority kept for reporting only
CELLS = [
    # P0
    ("esn",       "narma10",      "P0"),
    ("esn",       "narma30",      "P0"),
    ("esn",       "mackey_glass", "P0"),
    ("esn",       "lorenz63",     "P0"),
    ("leaky_esn", "narma10",      "P0"),
    ("leaky_esn", "narma30",      "P0"),
    ("leaky_esn", "mackey_glass", "P0"),
    ("leaky_esn", "lorenz63",     "P0"),
    # P1
    ("deep_esn",  "narma10",      "P1"),
    ("deep_esn",  "narma30",      "P1"),
    ("deep_esn",  "mackey_glass", "P1"),
    ("deep_esn",  "lorenz63",     "P1"),
    ("lsm",       "narma10",      "P1"),
    ("lsm",       "mackey_glass", "P1"),
    ("fhn",       "narma10",      "P1"),
    ("fhn",       "mackey_glass", "P1"),
    # P2
    ("logistic",  "narma10",      "P2"),
    ("logistic",  "mackey_glass", "P2"),
    # P3
    ("qrc",       "narma10",      "P3"),
]

DATA_LENGTH = {
    "narma10":      1000,
    "narma30":      1500,
    "mackey_glass": 2000,
    "lorenz63":     2000,
}

SMOKE_HPO_TRIALS = 5
SMOKE_N_SEEDS    = 2
SMOKE_WASHOUT    = 100


def _make_spec(model: str, task: str) -> ExperimentSpec:
    forecasting_mode = "closed_loop" if task == "lorenz63" else "one_step"
    length = DATA_LENGTH[task]
    return ExperimentSpec(
        dataset=DatasetSpec(name=task, length=length, seed=42),
        reservoir=ReservoirSpec(type=model, params={}),
        protocol=ProtocolSpec(
            washout=SMOKE_WASHOUT,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode=forecasting_mode,
            use_hpo=True,
            hpo_budget=SMOKE_HPO_TRIALS,
            n_seeds=SMOKE_N_SEEDS,
        ),
        readout=ReadoutSpec(alpha_grid=[1.0]),  # overridden by HPO
        seed=42,
    )


def _run_one(model: str, task: str, priority: str) -> dict:
    t0 = time.perf_counter()
    spec = _make_spec(model, task)
    data = get_data_for_experiment(task, length=DATA_LENGTH[task], seed=42)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_pipeline(data, spec)
        dt = time.perf_counter() - t0

        ms = result.multi_seed_result
        if ms is None:
            return {
                "model": model, "task": task, "priority": priority,
                "status": "no_multi_seed", "elapsed_sec": dt,
            }
        diag = result.hpo_diagnostics or {}
        return {
            "model": model, "task": task, "priority": priority,
            "status": "ok",
            "nrmse_range_mean": ms.mean.nrmse_range,
            "nrmse_range_std":  ms.std.nrmse_range,
            "val_nrmse_mean":   ms.mean.val_nrmse_range,
            "n_completed_trials": diag.get("n_completed", -1),
            "n_pruned_trials":    diag.get("n_pruned", -1),
            "elapsed_sec":      dt,
        }
    except Exception as exc:
        dt = time.perf_counter() - t0
        return {
            "model": model, "task": task, "priority": priority,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=4),
            "elapsed_sec": dt,
        }


def main():
    out_dir = REPO / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_results.json"

    results = []
    print(f"Running {len(CELLS)} smoke cells (HPO={SMOKE_HPO_TRIALS} trials, "
          f"seeds={SMOKE_N_SEEDS})...\n", flush=True)
    print(f"{'#':>3} {'priority':<3} {'model':<10} {'task':<14} "
          f"{'status':<10} {'nrmse_mean':>10} {'time_s':>8}", flush=True)
    print("-" * 75, flush=True)

    for i, (model, task, priority) in enumerate(CELLS, 1):
        rec = _run_one(model, task, priority)
        results.append(rec)
        nrmse = rec.get("nrmse_range_mean")
        nrmse_str = f"{nrmse:.4f}" if isinstance(nrmse, float) else "—"
        print(f"{i:>3} {priority:<3} {model:<10} {task:<14} "
              f"{rec['status']:<10} {nrmse_str:>10} {rec['elapsed_sec']:>8.1f}",
              flush=True)
        # Persist after each cell so partial progress is preserved
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "FAILED")
    print("\n" + "=" * 75, flush=True)
    print(f"DONE: {n_ok} ok, {n_fail} failed, {len(results)} total. "
          f"Saved → {out_path.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
