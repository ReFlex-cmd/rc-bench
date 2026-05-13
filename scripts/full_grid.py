"""Stage 4.2 full grid: HPO 100×10 (P0/P1), 50×5 (P2/P3) per ТЗ §4.2.

Saves a RunRecord (with full reproducibility metadata) per cell to
``reports/runs/<model>__<task>.json`` and the aggregate table to
``reports/full_results.json``.
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
from rc_bench.reporting.run_record import RunRecord, save_run_record
from rc_bench.runners.pipeline import run_pipeline


# (model, task, priority, hpo_budget, n_seeds)
CELLS = [
    # P0: ESN, Leaky ESN — full grid
    ("esn",       "narma10",      "P0", 100, 10),
    ("esn",       "narma30",      "P0", 100, 10),
    ("esn",       "mackey_glass", "P0", 100, 10),
    ("esn",       "lorenz63",     "P0", 100, 10),
    ("leaky_esn", "narma10",      "P0", 100, 10),
    ("leaky_esn", "narma30",      "P0", 100, 10),
    ("leaky_esn", "mackey_glass", "P0", 100, 10),
    ("leaky_esn", "lorenz63",     "P0", 100, 10),
    # P1: Deep ESN — full grid; LSM, FHN — no NARMA-30, no Lorenz-63
    ("deep_esn",  "narma10",      "P1", 100, 10),
    ("deep_esn",  "narma30",      "P1", 100, 10),
    ("deep_esn",  "mackey_glass", "P1", 100, 10),
    ("deep_esn",  "lorenz63",     "P1", 100, 10),
    ("lsm",       "narma10",      "P1", 100, 10),
    ("lsm",       "mackey_glass", "P1", 100, 10),
    ("fhn",       "narma10",      "P1", 100, 10),
    ("fhn",       "mackey_glass", "P1", 100, 10),
    # P2: Logistic — reduced budget
    ("logistic",  "narma10",      "P2", 50,  5),
    ("logistic",  "mackey_glass", "P2", 50,  5),
    # P3: QRC — minimum
    ("qrc",       "narma10",      "P3", 50,  5),
]

DATA_LENGTH = {
    "narma10":      2000,
    "narma30":      3000,
    "mackey_glass": 5000,
    "lorenz63":     5000,
}

WASHOUT = 200


def _make_spec(model: str, task: str, hpo_budget: int, n_seeds: int) -> ExperimentSpec:
    forecasting_mode = "closed_loop" if task == "lorenz63" else "one_step"
    length = DATA_LENGTH[task]
    return ExperimentSpec(
        dataset=DatasetSpec(name=task, length=length, seed=42),
        reservoir=ReservoirSpec(type=model, params={}),
        protocol=ProtocolSpec(
            washout=WASHOUT,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode=forecasting_mode,
            use_hpo=True,
            hpo_budget=hpo_budget,
            n_seeds=n_seeds,
        ),
        readout=ReadoutSpec(alpha_grid=[1.0]),  # overridden by HPO
        seed=42,
    )


def _run_one(model: str, task: str, priority: str, hpo_budget: int, n_seeds: int,
             runs_dir: Path) -> dict:
    t0 = time.perf_counter()
    spec = _make_spec(model, task, hpo_budget, n_seeds)
    data = get_data_for_experiment(task, length=DATA_LENGTH[task], seed=42)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_spec = run_pipeline(data, spec)
        dt = time.perf_counter() - t0

        # Persist full RunRecord with reproducibility metadata
        record = RunRecord.make(spec, result_spec)
        save_run_record(record, runs_dir / f"{model}__{task}.json")

        ms = result_spec.multi_seed_result
        diag = result_spec.hpo_diagnostics or {}
        if ms is None:
            return {
                "model": model, "task": task, "priority": priority,
                "status": "no_multi_seed", "elapsed_sec": dt,
            }
        return {
            "model": model, "task": task, "priority": priority,
            "status": "ok",
            "hpo_budget": hpo_budget, "n_seeds": n_seeds,
            "nrmse_range_mean": ms.mean.nrmse_range,
            "nrmse_range_std":  ms.std.nrmse_range,
            "val_nrmse_mean":   ms.mean.val_nrmse_range,
            "prediction_horizon_mean": ms.mean.prediction_horizon,
            "n_completed_trials":      diag.get("n_completed", -1),
            "n_pruned_trials":         diag.get("n_pruned", -1),
            "best_trial":              diag.get("best_trial_number", -1),
            "elapsed_sec": dt,
        }
    except Exception as exc:
        dt = time.perf_counter() - t0
        return {
            "model": model, "task": task, "priority": priority,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=5),
            "elapsed_sec": dt,
        }


def main():
    out_dir = REPO / "reports"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full_results.json"

    results = []
    print(f"Running {len(CELLS)} full-grid cells...\n", flush=True)
    print(f"{'#':>3} {'pri':<3} {'model':<10} {'task':<14} "
          f"{'status':<10} {'NRMSE±std':>22} {'pruned':>7} {'time_s':>9}", flush=True)
    print("-" * 88, flush=True)

    t_start = time.perf_counter()
    for i, (model, task, priority, budget, seeds) in enumerate(CELLS, 1):
        rec = _run_one(model, task, priority, budget, seeds, runs_dir)
        results.append(rec)
        if rec["status"] == "ok":
            nrmse_str = f"{rec['nrmse_range_mean']:.4f} ± {rec['nrmse_range_std']:.4f}"
            pruned_str = f"{rec['n_pruned_trials']}/{rec['n_completed_trials']+rec['n_pruned_trials']}"
        else:
            nrmse_str = "—"
            pruned_str = "—"
        print(f"{i:>3} {priority:<3} {model:<10} {task:<14} "
              f"{rec['status']:<10} {nrmse_str:>22} {pruned_str:>7} {rec['elapsed_sec']:>9.1f}",
              flush=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    t_total = time.perf_counter() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "FAILED")
    print("\n" + "=" * 88, flush=True)
    print(f"DONE in {t_total/60:.1f} min: {n_ok} ok, {n_fail} failed, "
          f"{len(results)} total. Saved → {out_path.relative_to(REPO)}", flush=True)
    print(f"Per-cell records → {runs_dir.relative_to(REPO)}/", flush=True)


if __name__ == "__main__":
    main()
