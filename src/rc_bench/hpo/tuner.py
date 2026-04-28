"""
Optuna-based hyperparameter optimisation for RC-Bench.

Pipeline (Variant A):
    best_params, best_score = run_hpo(spec, data, n_trials, seed)
    final_spec = apply_hpo_params(spec, best_params)
    result = run_experiment(data, final_spec, reservoir)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import optuna

from rc_bench.core.reservoirs.registry import get_reservoir
from rc_bench.core.schema import ExperimentSpec, ReadoutSpec
from rc_bench.hpo.search_spaces import params_from_trial, suggest_params
from rc_bench.runners.experiment_runner import run_experiment

# Silence Optuna's own logging; rc-bench uses warnings/print for UX
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_hpo(
    spec: ExperimentSpec,
    data: Dict[str, Any],
    n_trials: int,
    seed: int | None = None,
) -> Tuple[Dict[str, Any], float]:
    """Search for the best hyperparameters using Optuna TPE + MedianPruner.

    Parameters
    ----------
    spec     : base ExperimentSpec (not mutated)
    data     : pre-split data dict from get_data_for_experiment()
    n_trials : number of Optuna trials
    seed     : RNG seed for the Optuna sampler (defaults to spec.seed)

    Returns
    -------
    best_params : dict with keys "reservoir_params" and "readout_alpha"
    best_score  : best val_nrmse_range achieved
    """
    if seed is None:
        seed = spec.seed

    sampler = optuna.samplers.TPESampler(seed=seed)
    # MedianPruner: startup_trials complete without pruning, then prune
    # trials whose reported value exceeds the median of completed trials.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        lambda trial: _objective(trial, spec, data),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_log_callback],
    )

    best_params = params_from_trial(study.best_trial.params, spec.reservoir.type)
    return best_params, float(study.best_trial.value)


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
        result = run_experiment(data, trial_spec, reservoir)
    except Exception as exc:
        # Bad hyperparameter combination — report a large score and prune
        logger.debug("Trial %d failed: %s", trial.number, exc)
        raise optuna.TrialPruned() from exc

    val_score = float(result["metrics"].val_nrmse_range)

    # Report intermediate value so MedianPruner can act on subsequent trials
    trial.report(val_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_score


def _log_callback(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        logger.debug(
            "Trial %d | val_nrmse_range=%.5f | best=%.5f",
            trial.number,
            trial.value,
            study.best_value,
        )
