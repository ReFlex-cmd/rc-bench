"""Re-run only Lorenz-63 cells with the closed_loop_predict fix applied.

HPO is repeated (deterministic with seed=42 → same best_params), but the
test-time closed-loop inference now uses the corrected forecast loop.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.schema import (
    DatasetSpec, ExperimentSpec, ProtocolSpec,
    ReadoutSpec, ReservoirSpec,
)
from rc_bench.reporting.run_record import RunRecord, save_run_record
from rc_bench.runners.pipeline import run_pipeline


CELLS = [("esn", 100, 10), ("leaky_esn", 100, 10), ("deep_esn", 100, 10)]


def main():
    runs_dir = REPO / "reports" / "runs"
    full_results = json.loads((REPO / "reports" / "full_results.json").read_text())

    print("Re-running Lorenz-63 cells with closed_loop fix...")
    for model, budget, seeds in CELLS:
        spec = ExperimentSpec(
            dataset=DatasetSpec(name="lorenz63", length=5000, seed=42),
            reservoir=ReservoirSpec(type=model, params={}),
            protocol=ProtocolSpec(
                washout=200, train_frac=0.6, val_frac=0.2,
                forecasting_mode="closed_loop",
                use_hpo=True, hpo_budget=budget, n_seeds=seeds,
            ),
            readout=ReadoutSpec(alpha_grid=[1.0]),
            seed=42,
        )
        data = get_data_for_experiment("lorenz63", length=5000, seed=42)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_spec = run_pipeline(data, spec)
        dt = time.perf_counter() - t0

        record = RunRecord.make(spec, result_spec)
        save_run_record(record, runs_dir / f"{model}__lorenz63.json")

        ms = result_spec.multi_seed_result
        nrmse = ms.mean.nrmse_range
        std = ms.std.nrmse_range
        ph = ms.mean.prediction_horizon
        ph_std = ms.std.prediction_horizon
        print(f"  {model:<10} NRMSE={nrmse:.4f} ± {std:.4f}   "
              f"horizon={ph:.1f} ± {ph_std:.1f} steps "
              f"= {ph*0.1*0.906:.2f} ± {ph_std*0.1*0.906:.2f} λ_max·t   ({dt:.1f}s)")

        # Also update the aggregate full_results.json
        for r in full_results:
            if r.get("model") == model and r.get("task") == "lorenz63":
                r["nrmse_range_mean"] = float(nrmse)
                r["nrmse_range_std"] = float(std)
                r["val_nrmse_mean"] = float(ms.mean.val_nrmse_range)
                r["prediction_horizon_mean"] = float(ph)
                r["elapsed_sec"] = dt
                break

    (REPO / "reports" / "full_results.json").write_text(
        json.dumps(full_results, indent=2, ensure_ascii=False)
    )
    print("\nfull_results.json updated.")


if __name__ == "__main__":
    main()
