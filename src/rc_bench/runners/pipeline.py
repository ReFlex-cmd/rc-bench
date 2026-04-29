"""
Unified experiment pipeline: HPO → multi-seed → ResultSpec.

Both CLI ``rcbench run`` and the Celery worker call this entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from rc_bench.core.reservoirs.registry import get_reservoir
from rc_bench.core.schema import ExperimentSpec, ResultSpec
from rc_bench.runners.experiment_runner import run_experiment
from rc_bench.runners.multi_seed import run_multi_seed


def run_pipeline(
    data: Dict[str, Any],
    spec: ExperimentSpec,
    artifact_dir: Optional[Path] = None,
) -> ResultSpec:
    """Execute the full pipeline described by *spec*.

    Steps
    -----
    1. Optional HPO (spec.protocol.use_hpo):
       Optuna finds the best reservoir + readout hyperparameters;
       the spec is updated in-place for the evaluation phase.
    2. Evaluation:
       - n_seeds == 1  → single run_experiment, stores MetricsResult
       - n_seeds  > 1  → run_multi_seed, stores MultiSeedResult
    3. Optional artifacts: if artifact_dir is given, saves predictions.npz
       (single-seed only) and populates ResultSpec.artifact_paths.
    """
    hpo_best_params = None

    if spec.protocol.use_hpo:
        from rc_bench.hpo.tuner import apply_hpo_params, run_hpo

        hpo_best_params, _ = run_hpo(
            spec,
            data,
            n_trials=spec.protocol.hpo_budget,
            seed=spec.seed,
        )
        spec = apply_hpo_params(spec, hpo_best_params)

    n_seeds = spec.protocol.n_seeds

    if n_seeds > 1:
        multi_seed_result = run_multi_seed(data, spec, n_seeds)
        return ResultSpec(
            status="completed",
            config_hash=spec.config_hash(),
            metrics=None,               # use multi_seed_result.mean for reporting
            multi_seed_result=multi_seed_result,
            hpo_best_params=hpo_best_params,
        )
    else:
        reservoir_config = {"seed": spec.seed, **spec.reservoir.params}
        reservoir = get_reservoir(spec.reservoir.type, reservoir_config)
        result = run_experiment(data, spec, reservoir)

        artifact_paths: Dict[str, str] = {}
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            npz_path = artifact_dir / f"{spec.config_hash()}_predictions.npz"
            np.savez(
                npz_path,
                y_test=result["y_test"],
                y_pred=result["preds"],
            )
            artifact_paths["predictions"] = str(npz_path)

        return ResultSpec(
            status="completed",
            config_hash=spec.config_hash(),
            metrics=result["metrics"],
            hpo_best_params=hpo_best_params,
            artifact_paths=artifact_paths,
        )
