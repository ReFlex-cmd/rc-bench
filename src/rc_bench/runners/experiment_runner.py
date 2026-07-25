import time
import tracemalloc
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from rc_bench.core.reservoirs.base import BaseReservoir
from rc_bench.core.schema import EvaluationContext, ExperimentSpec, MetricsResult
from rc_bench.core.metrics import (
    mae,
    mae_skill,
    mase,
    mse,
    rmse,
    nrmse_range,
    nrmse_std,
    nrmse_var,
    prediction_horizon,
    seasonal_naive_mae_scale,
)
from rc_bench.core.baselines import seasonal_persistence_forecast
from rc_bench.protocol.forecasting import (
    build_one_step_targets,
    build_fixed_horizon_targets,
    closed_loop_predict,
    aligned_target_positions,
    target_offset,
)
from rc_bench.readout.ridge import select_alpha, RidgeReadout


class ExperimentDataError(ValueError):
    """Raised when experiment data cannot satisfy the evaluation protocol."""


@dataclass(frozen=True)
class _TrainValidationEvaluation:
    H_train: np.ndarray
    y_train: np.ndarray
    H_val: np.ndarray
    y_val: np.ndarray
    alpha_info: Dict[str, float]
    scaler: StandardScaler | None
    reservoir_states_std: float


def _fit_input_scaler(
    X_train: np.ndarray,
    scaler_name: str,
) -> Tuple[np.ndarray, StandardScaler | None]:
    if scaler_name.lower() == "zscore":
        scaler = StandardScaler()
        return scaler.fit_transform(X_train), scaler
    return X_train, None


def _transform_inputs(
    X: np.ndarray,
    scaler: StandardScaler | None,
) -> np.ndarray:
    return scaler.transform(X) if scaler is not None else X


def _scale_inputs(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train_s, scaler = _fit_input_scaler(X_train, scaler_name)
    return (
        X_train_s,
        _transform_inputs(X_val, scaler),
        _transform_inputs(X_test, scaler),
    )


def _target_observed_mask(
    data: Mapping[str, np.ndarray],
    split: str,
    y: np.ndarray,
) -> np.ndarray:
    key = f"target_observed_mask_{split}"
    if key not in data:
        return np.ones(len(y), dtype=np.bool_)

    mask = np.asarray(data[key])
    if mask.dtype != np.dtype(np.bool_):
        raise ExperimentDataError(
            f"{key} must have boolean dtype, got {mask.dtype}"
        )
    if mask.shape != (len(y),):
        raise ExperimentDataError(
            f"{key} shape must be {(len(y),)}, got {mask.shape}"
        )
    return mask


def _validate_target_mask_group(
    data: Mapping[str, np.ndarray],
    splits: Tuple[str, ...],
) -> None:
    keys = tuple(f"target_observed_mask_{split}" for split in splits)
    present = tuple(key in data for key in keys)
    if not any(present) or all(present):
        return

    if len(splits) == 2:
        group_name = f"{splits[0]} and {splits[1]}"
    else:
        group_name = f"{', '.join(splits[:-1])}, and {splits[-1]}"
    missing = ", ".join(
        split for split, is_present in zip(splits, present) if not is_present
    )
    raise ExperimentDataError(
        f"target observed masks must provide {group_name} together; "
        f"missing: {missing}"
    )


def _aligned_target_mask(
    mask: np.ndarray,
    spec: ExperimentSpec,
) -> np.ndarray:
    offset = target_offset(
        spec.protocol.washout,
        spec.protocol.horizon,
        spec.protocol.forecasting_mode,
    )
    return mask[offset:]


def _build_observed_training_pairs(
    H_full: np.ndarray,
    y: np.ndarray,
    observed_mask: np.ndarray,
    spec: ExperimentSpec,
    split: str,
) -> Tuple[np.ndarray, np.ndarray]:
    washout = spec.protocol.washout
    H = H_full[washout:]
    y_eff = y[washout:]
    mode = spec.protocol.forecasting_mode

    if mode == "one_step" or mode == "closed_loop":
        H_aligned, y_aligned = build_one_step_targets(H, y_eff)
    elif mode == "fixed_horizon":
        try:
            H_aligned, y_aligned = build_fixed_horizon_targets(
                H,
                y_eff,
                spec.protocol.horizon,
            )
        except ValueError as exc:
            raise ExperimentDataError(
                f"{split} cannot satisfy fixed-horizon alignment: {exc}"
            ) from exc
    else:
        raise ValueError(f"Unknown forecasting_mode: {mode!r}")

    aligned_mask = _aligned_target_mask(observed_mask, spec)
    if not (
        len(H_aligned) == len(y_aligned) == len(aligned_mask)
    ):
        raise ExperimentDataError(
            f"{split} aligned states, targets, and observed-target mask "
            "must have equal lengths"
        )
    if not np.any(aligned_mask):
        raise ExperimentDataError(
            f"{split} has zero observed targets after "
            "washout/forecast alignment"
        )
    return H_aligned[aligned_mask], y_aligned[aligned_mask]


def _prepare_train_validation(
    data: Mapping[str, np.ndarray],
    spec: ExperimentSpec,
    reservoir: BaseReservoir,
) -> _TrainValidationEvaluation:
    _validate_target_mask_group(data, ("train", "val"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    train_mask = _target_observed_mask(data, "train", y_train)
    val_mask = _target_observed_mask(data, "val", y_val)

    scaler_name = spec.reservoir.params.get(
        "scaler",
        reservoir.DEFAULT_SCALER,
    )
    X_train_s, scaler = _fit_input_scaler(X_train, scaler_name)
    X_val_s = _transform_inputs(X_val, scaler)

    H_train_full = reservoir.transform(X_train_s)
    _check_sanity(reservoir, H_train_full)
    H_val_full = reservoir.transform(X_val_s)

    H_train, y_train_observed = _build_observed_training_pairs(
        H_train_full,
        y_train,
        train_mask,
        spec,
        "train",
    )
    H_val, y_val_observed = _build_observed_training_pairs(
        H_val_full,
        y_val,
        val_mask,
        spec,
        "val",
    )
    alpha_info = select_alpha(
        H_train,
        y_train_observed,
        H_val,
        y_val_observed,
        spec.readout.alpha_grid,
        metric=spec.protocol.selection_metric,
    )

    return _TrainValidationEvaluation(
        H_train=H_train,
        y_train=y_train_observed,
        H_val=H_val,
        y_val=y_val_observed,
        alpha_info=alpha_info,
        scaler=scaler,
        reservoir_states_std=float(np.asarray(H_train_full).std()),
    )


def evaluate_train_validation(
    data: Mapping[str, np.ndarray],
    spec: ExperimentSpec,
    reservoir: BaseReservoir,
) -> Dict[str, float]:
    """Evaluate one HPO candidate using train and validation data only."""

    prepared = _prepare_train_validation(data, spec, reservoir)
    alpha_info = prepared.alpha_info
    return {
        "best_alpha": float(alpha_info["alpha"]),
        "val_nrmse_range": float(alpha_info["val_nrmse"]),
        "val_nrmse_std": alpha_info.get("val_nrmse_std"),
        # Selection metric value the HPO objective minimises (DEC-013).
        "val_score": float(alpha_info.get("val_score", alpha_info["val_nrmse"])),
        "reservoir_states_std": prepared.reservoir_states_std,
    }


def run_experiment(
    data: Mapping[str, np.ndarray],
    spec: ExperimentSpec,
    reservoir: BaseReservoir,
) -> Dict[str, Any]:
    washout: int = spec.protocol.washout
    mode: str = spec.protocol.forecasting_mode

    _validate_target_mask_group(data, ("train", "val", "test"))
    tracemalloc.start()
    try:
        t_train_start = time.perf_counter()
        prepared = _prepare_train_validation(data, spec, reservoir)
        best_alpha = prepared.alpha_info["alpha"]

        H_tv = np.vstack([prepared.H_train, prepared.H_val])
        y_tv = np.concatenate([prepared.y_train, prepared.y_val])
        readout = RidgeReadout(best_alpha)
        readout.fit(H_tv, y_tv)
        t_train_end = time.perf_counter()

        # Test data is deliberately accessed only after model selection and fit.
        X_test, y_test = data["X_test"], data["y_test"]
        test_mask = _target_observed_mask(data, "test", y_test)
        X_test_s = _transform_inputs(X_test, prepared.scaler)
        y_test_eff = y_test[washout:]

        t_infer_start = time.perf_counter()
        if mode == "one_step":
            H_test_full = reservoir.transform(X_test_s)
            H_test = H_test_full[washout:]
            H_te, y_te_full = build_one_step_targets(H_test, y_test_eff)
            y_test_pred_full = readout.predict(H_te)
        elif mode == "fixed_horizon":
            H_test_full = reservoir.transform(X_test_s)
            H_test = H_test_full[washout:]
            try:
                H_te, y_te_full = build_fixed_horizon_targets(
                    H_test,
                    y_test_eff,
                    spec.protocol.horizon,
                )
            except ValueError as exc:
                raise ExperimentDataError(
                    f"test cannot satisfy fixed-horizon alignment: {exc}"
                ) from exc
            y_test_pred_full = readout.predict(H_te)
        elif mode == "closed_loop":
            # Warm up on the washout window of the test set.
            X_test_seed = (
                X_test_s[:washout] if washout > 0 else X_test_s[:1]
            )
            y_te_full = y_test_eff
            y_test_pred_full = closed_loop_predict(
                readout,
                reservoir,
                X_test_seed,
                len(y_te_full),
            )
        else:
            raise ValueError(f"Unknown forecasting_mode: {mode!r}")

        aligned_test_mask = _aligned_target_mask(test_mask, spec)
        if not (
            len(y_te_full)
            == len(y_test_pred_full)
            == len(aligned_test_mask)
        ):
            raise ExperimentDataError(
                "test aligned targets, predictions, and observed-target mask "
                "must have equal lengths"
            )
        if not np.any(aligned_test_mask):
            raise ExperimentDataError(
                "test has zero observed targets after "
                "washout/forecast alignment"
            )
        y_te = y_te_full[aligned_test_mask]
        y_test_pred = y_test_pred_full[aligned_test_mask]

        t_infer_end = time.perf_counter()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # JMLC baseline-relative metrics (DEC-013): computed for reservoirs too, on
    # the SAME test target set, so the fair table compares like with like.
    mase_value = None
    mae_skill_value = None
    season = None
    mase_scale = None
    seasonal_test_mae = None
    n_scale_terms = None
    if spec.protocol.seasonal_period is not None and mode == "fixed_horizon":
        season = spec.protocol.seasonal_period
        positions = aligned_target_positions(
            test_mask, washout, spec.protocol.horizon, mode
        )
        seasonal_ref = seasonal_persistence_forecast(
            np.asarray(data["y_test"], dtype=float), positions, season
        )
        seasonal_test_mae = mae(y_te, seasonal_ref)
        train_values = np.asarray(data["y_train"], dtype=float)
        train_mask = _target_observed_mask(data, "train", train_values)
        mase_scale, n_scale_terms = seasonal_naive_mae_scale(
            train_values, season, observed_mask=train_mask
        )
        mase_value = mase(y_te, y_test_pred, mase_scale)
        mae_skill_value = mae_skill(mae(y_te, y_test_pred), seasonal_test_mae)

    # Same context the baseline runner records (DEC-013), so a reader of the
    # evidence bundle can confirm both families scored the same target set and
    # shared the same MASE scale without reloading the predictions.
    evaluation = EvaluationContext(
        target_start_index=target_offset(washout, spec.protocol.horizon, mode),
        n_test_targets=int(y_te.size),
        seasonal_period=season,
        mase_scale=mase_scale,
        seasonal_test_mae=seasonal_test_mae,
        n_mase_scale_terms=n_scale_terms,
    )

    metrics = MetricsResult(
        rmse=rmse(y_te, y_test_pred),
        nrmse_range=nrmse_range(y_te, y_test_pred),
        nrmse_std=nrmse_std(y_te, y_test_pred),
        nrmse_var=nrmse_var(y_te, y_test_pred),
        mae=mae(y_te, y_test_pred),
        mse=mse(y_te, y_test_pred),
        prediction_horizon=prediction_horizon(y_te, y_test_pred),
        val_nrmse_range=prepared.alpha_info["val_nrmse"],
        val_nrmse_std=prepared.alpha_info.get("val_nrmse_std"),
        mase=mase_value,
        mae_skill=mae_skill_value,
        train_time=t_train_end - t_train_start,
        inference_latency=t_infer_end - t_infer_start,
        peak_memory=peak_bytes,
    )

    return {
        "metrics": metrics,
        "evaluation": evaluation,
        "best_alpha": best_alpha,
        "preds": y_test_pred,
        "y_test": y_te,
        "reservoir_states_std": prepared.reservoir_states_std,
    }


def _check_sanity(reservoir: BaseReservoir, H: np.ndarray) -> None:
    for check_name, passed in reservoir.sanity_check(H).items():
        if not passed:
            warnings.warn(
                f"{type(reservoir).__name__} failed sanity check: {check_name}",
                RuntimeWarning,
                stacklevel=3,
            )
