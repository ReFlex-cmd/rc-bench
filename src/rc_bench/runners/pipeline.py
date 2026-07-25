"""
Unified experiment pipeline: HPO → multi-seed → ResultSpec.

Both CLI ``rcbench run`` and the Celery worker call this entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from rc_bench.core.reservoirs.registry import get_reservoir
from rc_bench.core.schema import ExperimentSpec, ResultSpec, SelectionResult
from rc_bench.runners.baseline_runner import run_baseline
from rc_bench.runners.experiment_runner import run_experiment
from rc_bench.runners.multi_seed import run_multi_seed


def _save_predictions_artifact(
    artifact_dir: Path,
    config_hash: str,
    seed: int,
    result: Dict[str, Any],
) -> str:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    npz_path = artifact_dir / f"{config_hash}_seed{seed}_predictions.npz"
    np.savez(
        npz_path,
        y_test=result["y_test"],
        y_pred=result["preds"],
        seed=np.asarray(seed, dtype=np.int64),
    )
    return str(npz_path)


def _reservoir_selection(
    resolved_spec: ExperimentSpec,
    hpo_best_params: Optional[Dict[str, Any]],
) -> SelectionResult:
    if resolved_spec.protocol.use_hpo:
        return SelectionResult(
            method="optuna",
            metric=resolved_spec.protocol.selection_metric,
            selected_params=hpo_best_params or {},
        )
    return SelectionResult(method="none")


def _run_baseline_pipeline(
    data: Dict[str, Any],
    spec: ExperimentSpec,
    artifact_dir: Optional[Path],
    save_predictions: bool,
) -> ResultSpec:
    """Deterministic single-result path for BaselineSpec experiments (DEC-014).

    Baselines never run Optuna HPO or multi-seed (enforced by ExperimentSpec),
    so the frozen and resolved specs are identical.
    """
    resolved_spec = spec.model_copy(deep=True)
    config_hash = resolved_spec.config_hash()
    result = run_baseline(data, resolved_spec)

    artifact_paths: Dict[str, str] = {}
    if save_predictions:
        assert artifact_dir is not None
        artifact_dir.mkdir(parents=True, exist_ok=True)
        npz_path = artifact_dir / f"{config_hash}_deterministic_predictions.npz"
        np.savez(
            npz_path,
            y_test=result["y_test"],
            y_pred=result["preds"],
        )
        artifact_paths["predictions"] = str(npz_path)

    return ResultSpec(
        status="completed",
        config_hash=config_hash,
        frozen_config_hash=config_hash,
        resolved_spec=resolved_spec,
        metrics=result["metrics"],
        model_family="baseline",
        deterministic=True,
        evaluated_seeds=[],
        selection=result["selection"],
        evaluation=result["evaluation"],
        artifact_paths=artifact_paths,
    )


def run_pipeline(
    data: Dict[str, Any],
    spec: ExperimentSpec,
    artifact_dir: Optional[Path] = None,
    save_predictions: bool = False,
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
    3. Optional demo artifact: only when ``save_predictions`` is true, saves
       one predictions file in ``artifact_dir`` and populates
       ResultSpec.artifact_paths. Multi-seed runs save only the first seed.
    """
    if save_predictions and artifact_dir is None:
        raise ValueError("artifact_dir is required when save_predictions=True")

    if spec.model_family == "baseline":
        return _run_baseline_pipeline(data, spec, artifact_dir, save_predictions)

    frozen_spec = spec.model_copy(deep=True)
    resolved_spec = frozen_spec.model_copy(deep=True)
    hpo_best_params = None
    hpo_convergence = None
    hpo_diagnostics = None

    if frozen_spec.protocol.use_hpo:
        from rc_bench.hpo.tuner import apply_hpo_params, run_hpo

        hpo = run_hpo(
            frozen_spec,
            data,
            n_trials=frozen_spec.protocol.hpo_budget,
            seed=frozen_spec.seed,
        )
        hpo_best_params = hpo.best_params
        hpo_convergence = hpo.convergence
        hpo_diagnostics = hpo.diagnostics
        resolved_spec = apply_hpo_params(frozen_spec, hpo_best_params)

    frozen_config_hash = frozen_spec.config_hash()
    resolved_config_hash = resolved_spec.config_hash()
    n_seeds = resolved_spec.protocol.n_seeds

    if n_seeds > 1:
        artifact_paths: Dict[str, str] = {}
        # The evaluation context is a property of the data and protocol, so it
        # is identical across seeds; the first seed's copy represents the run.
        first_seed: Dict[str, Any] = {}

        def observe_first_seed(seed: int, result: Dict[str, Any]) -> None:
            first_seed["evaluation"] = result.get("evaluation")
            if save_predictions:
                assert artifact_dir is not None
                artifact_paths["predictions"] = _save_predictions_artifact(
                    artifact_dir,
                    resolved_config_hash,
                    seed,
                    result,
                )

        multi_seed_result = run_multi_seed(
            data,
            resolved_spec,
            n_seeds,
            representative_callback=observe_first_seed,
        )
        return ResultSpec(
            status="completed",
            config_hash=resolved_config_hash,
            frozen_config_hash=frozen_config_hash,
            resolved_spec=resolved_spec,
            metrics=None,               # use multi_seed_result.mean for reporting
            multi_seed_result=multi_seed_result,
            hpo_best_params=hpo_best_params,
            hpo_convergence=hpo_convergence,
            hpo_diagnostics=hpo_diagnostics,
            model_family="reservoir",
            deterministic=False,
            evaluated_seeds=list(multi_seed_result.seeds),
            selection=_reservoir_selection(resolved_spec, hpo_best_params),
            evaluation=first_seed.get("evaluation"),
            artifact_paths=artifact_paths,
        )
    else:
        reservoir_config = {
            "seed": resolved_spec.seed,
            **resolved_spec.reservoir.params,
        }
        reservoir = get_reservoir(
            resolved_spec.reservoir.type,
            reservoir_config,
        )
        result = run_experiment(data, resolved_spec, reservoir)

        artifact_paths: Dict[str, str] = {}
        if save_predictions:
            assert artifact_dir is not None
            artifact_paths["predictions"] = _save_predictions_artifact(
                artifact_dir,
                resolved_config_hash,
                resolved_spec.seed,
                result,
            )

        return ResultSpec(
            status="completed",
            config_hash=resolved_config_hash,
            frozen_config_hash=frozen_config_hash,
            resolved_spec=resolved_spec,
            metrics=result["metrics"],
            hpo_best_params=hpo_best_params,
            hpo_convergence=hpo_convergence,
            hpo_diagnostics=hpo_diagnostics,
            model_family="reservoir",
            deterministic=False,
            evaluated_seeds=[resolved_spec.seed],
            selection=_reservoir_selection(resolved_spec, hpo_best_params),
            evaluation=result["evaluation"],
            artifact_paths=artifact_paths,
        )
