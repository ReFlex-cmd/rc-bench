"""JMLC real-data experiment matrix (DEC-002/DEC-013/DEC-014).

The canonical matrix is 7 models x 2 horizons = 14 cells: three deterministic
baselines (persistence, seasonal persistence, Ridge AR) and four reservoir
models (ESN, Leaky ESN, LSM, Logistic) on horizons 1 and 24.

A matrix config file is a single valid ``ExperimentSpec`` YAML used as the
shared template (dataset, protocol, readout, seed). ``build_cell_spec`` derives
each cell's spec from it, so `rcbench validate-spec <config>` stays meaningful
while `run_matrix` expands the full grid.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.schema import (
    BaselineSpec,
    DatasetSpec,
    ExperimentSpec,
    ProtocolSpec,
    ReadoutSpec,
    ReservoirSpec,
)
from rc_bench.reporting.run_record import RunRecord, save_run_record
from rc_bench.runners.pipeline import run_pipeline

# (family, model)
BASELINE_MODELS: Tuple[str, ...] = ("persistence", "seasonal_persistence", "ridge_ar")
RESERVOIR_MODELS: Tuple[str, ...] = ("esn", "leaky_esn", "lsm", "logistic")
HORIZONS: Tuple[int, ...] = (1, 24)

CANONICAL_CELLS: Tuple[Tuple[str, str], ...] = tuple(
    [("baseline", m) for m in BASELINE_MODELS]
    + [("reservoir", m) for m in RESERVOIR_MODELS]
)


def load_template(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate the matrix template (a single ExperimentSpec)."""
    raw = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("matrix config must be a mapping (a single ExperimentSpec)")
    ExperimentSpec.model_validate(raw)  # fail fast on an invalid template
    return raw


def build_cell_spec(
    template: Dict[str, Any],
    family: str,
    model: str,
    horizon: int,
) -> ExperimentSpec:
    """Derive one matrix cell's ExperimentSpec from the shared template."""
    dataset = dict(template["dataset"])
    readout = dict(template.get("readout", {"alpha_grid": [0.001, 0.01, 0.1, 1.0, 10.0]}))
    protocol = dict(template["protocol"])
    protocol["forecasting_mode"] = "fixed_horizon"
    protocol["horizon"] = horizon

    if family == "baseline":
        protocol = {**protocol, "n_seeds": 0, "use_hpo": False}
        return ExperimentSpec(
            dataset=DatasetSpec(**dataset),
            baseline=BaselineSpec(type=model),
            protocol=ProtocolSpec(**protocol),
            readout=ReadoutSpec(**readout),
            seed=None,
        )
    if family == "reservoir":
        return ExperimentSpec(
            dataset=DatasetSpec(**dataset),
            reservoir=ReservoirSpec(type=model, params={}),
            protocol=ProtocolSpec(**protocol),
            readout=ReadoutSpec(**readout),
            seed=template.get("seed", 42),
        )
    raise ValueError(f"unknown model family {family!r}")


def _metrics_summary(result: Any) -> Dict[str, Any]:
    msr = result.multi_seed_result
    metrics = msr.mean if msr is not None else result.metrics
    if metrics is None:
        return {}
    return {
        "nrmse_std": metrics.nrmse_std,
        "nrmse_range": metrics.nrmse_range,
        "mae": metrics.mae,
        "mase": metrics.mase,
        "mae_skill": metrics.mae_skill,
    }


def run_matrix(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    cells: Optional[List[Tuple[str, str]]] = None,
    horizons: Optional[Tuple[int, ...]] = None,
    save_predictions: bool = True,
) -> List[Dict[str, Any]]:
    """Run the JMLC matrix defined by *config_path*; write runs + summary.

    Failed cells are recorded (never dropped) so the report stays honest.
    """
    template = load_template(config_path)
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"

    dataset = template["dataset"]
    protocol = template["protocol"]
    data = get_data_for_experiment(
        dataset["name"],
        length=dataset.get("length", 2000),
        train_frac=protocol.get("train_frac", 0.6),
        val_frac=protocol.get("val_frac", 0.2),
    )

    cells = list(cells) if cells is not None else list(CANONICAL_CELLS)
    horizons = horizons if horizons is not None else HORIZONS

    summary: List[Dict[str, Any]] = []
    for family, model in cells:
        for horizon in horizons:
            entry: Dict[str, Any] = {
                "family": family,
                "model": model,
                "horizon": horizon,
            }
            t0 = time.perf_counter()
            try:
                spec = build_cell_spec(template, family, model, horizon)
                result = run_pipeline(
                    data,
                    spec,
                    artifact_dir=artifacts_dir,
                    save_predictions=save_predictions,
                )
                record = RunRecord.make(spec, result)
                save_run_record(record, runs_dir / f"{family}_{model}_h{horizon}.json")
                entry.update(
                    status=result.status,
                    config_hash=result.config_hash,
                    elapsed_sec=time.perf_counter() - t0,
                    **_metrics_summary(result),
                )
            except Exception as exc:  # noqa: BLE001 — record, never drop
                entry.update(
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                    trace=traceback.format_exc(limit=5),
                    elapsed_sec=time.perf_counter() - t0,
                )
            summary.append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary
