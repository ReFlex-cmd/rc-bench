"""Deterministic baseline runner for the JMLC real-data matrix.

Consumes the same ``data`` dict as :func:`run_experiment` (unshifted per-split
values plus observed-target masks) and produces a single deterministic result
for persistence, seasonal persistence or Ridge AR. Target alignment is shared
with the reservoir runner (``aligned_target_positions``) so every model is
evaluated on exactly the same observed target timestamps (DEC-013).
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from rc_bench.core.baselines import (
    RIDGE_AR_LAGS,
    ar_lag_features,
    persistence_forecast,
    seasonal_persistence_forecast,
    select_and_fit_ridge_ar,
)
from rc_bench.core.metrics import (
    mae,
    mae_skill,
    mase,
    mse,
    nrmse_range,
    nrmse_std,
    nrmse_var,
    prediction_horizon,
    rmse,
    seasonal_naive_mae_scale,
)
from rc_bench.core.schema import (
    EvaluationContext,
    ExperimentSpec,
    MetricsResult,
    SelectionCandidate,
    SelectionResult,
)
from rc_bench.protocol.forecasting import aligned_target_positions, target_offset
from rc_bench.runners.experiment_runner import (
    ExperimentDataError,
    _target_observed_mask,
    _validate_target_mask_group,
)


def _split_targets(
    data: Mapping[str, np.ndarray],
    split: str,
    washout: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (values, observed_mask, observed_target_positions) for a split."""
    values = np.asarray(data[f"y_{split}"], dtype=float)
    mask = _target_observed_mask(data, split, values)
    positions = aligned_target_positions(mask, washout, horizon, "fixed_horizon")
    if positions.size == 0:
        raise ExperimentDataError(
            f"{split} has zero observed targets after washout/forecast alignment"
        )
    return values, mask, positions


def run_baseline(
    data: Mapping[str, np.ndarray],
    spec: ExperimentSpec,
) -> Dict[str, Any]:
    if spec.model_family != "baseline" or spec.baseline is None:
        raise ExperimentDataError("run_baseline requires a baseline experiment spec")
    if spec.protocol.forecasting_mode != "fixed_horizon":
        raise ExperimentDataError(
            "baselines are evaluated in fixed_horizon mode only (DEC-013)"
        )
    horizon = spec.protocol.horizon
    washout = spec.protocol.washout
    season = spec.protocol.seasonal_period
    if season is None:
        raise ExperimentDataError(
            "baseline runs require protocol.seasonal_period (e.g. 24)"
        )

    _validate_target_mask_group(data, ("train", "val", "test"))
    train_values, train_mask, train_pos = _split_targets(
        data, "train", washout, horizon
    )
    val_values, _, val_pos = _split_targets(data, "val", washout, horizon)
    test_values, _, test_pos = _split_targets(data, "test", washout, horizon)

    y_val_true = val_values[val_pos]
    y_test_true = test_values[test_pos]
    btype = spec.baseline.type

    tracemalloc.start()
    try:
        t_train_start = time.perf_counter()
        if btype == "persistence":
            val_pred = persistence_forecast(val_values, val_pos, horizon)
            selection = SelectionResult(method="none")
        elif btype == "seasonal_persistence":
            val_pred = seasonal_persistence_forecast(val_values, val_pos, season)
            selection = SelectionResult(method="none")
        elif btype == "ridge_ar":
            X_train = ar_lag_features(train_values, train_pos, horizon, RIDGE_AR_LAGS)
            X_val = ar_lag_features(val_values, val_pos, horizon, RIDGE_AR_LAGS)
            X_test = ar_lag_features(test_values, test_pos, horizon, RIDGE_AR_LAGS)
            fit = select_and_fit_ridge_ar(
                X_train,
                train_values[train_pos],
                X_val,
                y_val_true,
                X_test,
                spec.readout.alpha_grid,
            )
            val_pred = fit.val_predictions
            selection = SelectionResult(
                method="fixed_grid",
                metric="nrmse_std",
                candidates=[
                    SelectionCandidate(params={"alpha": alpha}, score=score)
                    for alpha, score in fit.candidates
                ],
                selected_params={"alpha": fit.alpha},
            )
        else:  # pragma: no cover - guarded by BaselineSpec Literal
            raise ExperimentDataError(f"unknown baseline type {btype!r}")
        t_train_end = time.perf_counter()

        t_infer_start = time.perf_counter()
        if btype == "persistence":
            test_pred = persistence_forecast(test_values, test_pos, horizon)
        elif btype == "seasonal_persistence":
            test_pred = seasonal_persistence_forecast(test_values, test_pos, season)
        else:
            test_pred = fit.final_model.predict(fit.scaler.transform(X_test))
        t_infer_end = time.perf_counter()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Baseline-relative quality (MET-001 / DEC-013)
    seasonal_test_pred = seasonal_persistence_forecast(test_values, test_pos, season)
    seasonal_test_mae = mae(y_test_true, seasonal_test_pred)
    mase_scale, n_scale_terms = seasonal_naive_mae_scale(
        train_values, season, observed_mask=train_mask
    )
    model_test_mae = mae(y_test_true, test_pred)

    metrics = MetricsResult(
        rmse=rmse(y_test_true, test_pred),
        nrmse_range=nrmse_range(y_test_true, test_pred),
        nrmse_std=nrmse_std(y_test_true, test_pred),
        nrmse_var=nrmse_var(y_test_true, test_pred),
        mae=model_test_mae,
        mse=mse(y_test_true, test_pred),
        prediction_horizon=prediction_horizon(y_test_true, test_pred),
        val_nrmse_range=nrmse_range(y_val_true, val_pred),
        val_nrmse_std=nrmse_std(y_val_true, val_pred),
        mase=mase(y_test_true, test_pred, mase_scale),
        mae_skill=mae_skill(model_test_mae, seasonal_test_mae),
        train_time=t_train_end - t_train_start,
        inference_latency=t_infer_end - t_infer_start,
        peak_memory=float(peak_bytes),
    )

    evaluation = EvaluationContext(
        target_start_index=target_offset(washout, horizon, "fixed_horizon"),
        n_test_targets=int(test_pos.size),
        seasonal_period=season,
        mase_scale=mase_scale,
        seasonal_test_mae=seasonal_test_mae,
        n_mase_scale_terms=n_scale_terms,
    )

    return {
        "metrics": metrics,
        "selection": selection,
        "evaluation": evaluation,
        "preds": test_pred,
        "y_test": y_test_true,
        "model_family": "baseline",
        "deterministic": True,
        "evaluated_seeds": [],
        "best_alpha": fit.alpha if btype == "ridge_ar" else None,
    }
