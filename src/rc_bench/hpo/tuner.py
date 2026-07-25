"""
Optuna-based hyperparameter optimisation for RC-Bench.

Pipeline (Variant A):
    hpo = run_hpo(spec, data, n_trials, seed)
    final_spec = apply_hpo_params(spec, hpo.best_params)
    result = run_experiment(data, final_spec, reservoir)

``run_hpo`` returns an ``HPOResult`` carrying:
    - best_params      : structured params dict (reservoir_params + readout_alpha)
    - best_score       : final best val_nrmse_range
    - convergence      : best-so-far val_nrmse_range after each completed trial
                         (used for the hpo_convergence_<model>_<task>.png plots,
                         see ТЗ §5.2 / audit/03 §3.7.3)
    - diagnostics      : counters {n_trials, n_completed, n_pruned, n_failed,
                                   best_trial_number}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import optuna
from pydantic import BaseModel, Field

from rc_bench.core.schema import ExperimentSpec, ReadoutSpec
from rc_bench.hpo.search_spaces import params_from_trial, suggest_params
from rc_bench.runners.experiment_runner import (
    ExperimentDataError,
    evaluate_train_validation,
)
from rc_bench.core.reservoirs.registry import get_reservoir

# Silence Optuna's own logging; rc-bench uses warnings/print for UX
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


class HPOResult(BaseModel):
    best_params: Dict[str, Any]
    best_score: float
    convergence: List[float] = Field(default_factory=list)
    diagnostics: Dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_hpo(
    spec: ExperimentSpec,
    data: Dict[str, Any],
    n_trials: int,
    seed: Optional[int] = None,
) -> HPOResult:
    """Search for the best hyperparameters using Optuna TPE + MedianPruner.

    Returns
    -------
    HPOResult — see module docstring.
    """
    if seed is None:
        seed = spec.seed

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # Stateful trackers populated by the callback below.
    convergence: List[float] = []
    counters = {"n_completed": 0, "n_pruned": 0, "n_failed": 0}

    def _callback(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            counters["n_completed"] += 1
            convergence.append(float(study_.best_value))
            logger.debug(
                "Trial %d | val_nrmse_range=%.5f | best=%.5f",
                trial.number, trial.value, study_.best_value,
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            counters["n_pruned"] += 1
        elif trial.state == optuna.trial.TrialState.FAIL:
            counters["n_failed"] += 1

    study.optimize(
        lambda trial: _objective(trial, spec, data),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_callback],
    )

    best_params = params_from_trial(study.best_trial.params, spec.reservoir.type)
    diagnostics = {
        "n_trials":          n_trials,
        "n_completed":       counters["n_completed"],
        "n_pruned":          counters["n_pruned"],
        "n_failed":          counters["n_failed"],
        "best_trial_number": int(study.best_trial.number),
    }
    return HPOResult(
        best_params=best_params,
        best_score=float(study.best_trial.value),
        convergence=convergence,
        diagnostics=diagnostics,
    )


def apply_hpo_params(
    spec: ExperimentSpec,
    best_params: Dict[str, Any],
) -> ExperimentSpec:
    """Return a new ExperimentSpec with HPO-found params merged in."""
    reservoir_overrides = best_params.get("reservoir_params", {})
    readout_alpha = best_params.get("readout_alpha")

    new_reservoir = spec.reservoir.model_copy(
        update={"params": {**spec.reservoir.params, **reservoir_overrides}}
    )
    updates: Dict[str, Any] = {"reservoir": new_reservoir}
    if readout_alpha is not None:
        updates["readout"] = ReadoutSpec(alpha_grid=[readout_alpha])

    return spec.model_copy(update=updates)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _objective(
    trial: optuna.Trial,
    spec: ExperimentSpec,
    data: Dict[str, Any],
) -> float:
    params = suggest_params(trial, spec.reservoir.type)

    reservoir_config = {"seed": spec.seed, **spec.reservoir.params, **params["reservoir_params"]}
    reservoir = get_reservoir(spec.reservoir.type, reservoir_config)

    trial_spec = spec.model_copy(
        update={"readout": ReadoutSpec(alpha_grid=[params["readout_alpha"]])}
    )

    try:
        result = evaluate_train_validation(data, trial_spec, reservoir)
    except ExperimentDataError:
        # Broken data/mask contracts cannot be repaired by another trial.
        raise
    except Exception as exc:
        # Bad hyperparameter combination — report a large score and prune
        logger.debug("Trial %d failed: %s", trial.number, exc)
        raise optuna.TrialPruned() from exc

    # Feasibility gate: skip degenerate reservoirs (e.g. LSM that never spikes,
    # FHN locked in a fixed point). Without this, HPO can converge to a
    # "constant predictor" basin that gives an attractive but useless val_nrmse.
    # See audit/_diag_seed.py.
    states_std = result.get("reservoir_states_std", 1.0)
    if states_std < 1e-6:
        logger.debug("Trial %d pruned: degenerate reservoir (states_std=%.2e)",
                     trial.number, states_std)
        raise optuna.TrialPruned()

    # Minimise the configured selection metric (NRMSE_range legacy default,
    # NRMSE_std for JMLC per DEC-013); falls back to range for older results.
    val_score = float(result.get("val_score", result["val_nrmse_range"]))

    # Report intermediate value so MedianPruner can act on subsequent trials
    trial.report(val_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_score
