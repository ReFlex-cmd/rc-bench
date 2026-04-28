import time
import tracemalloc
import warnings
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from rc_bench.core.reservoirs.base import BaseReservoir
from rc_bench.core.schema import ExperimentSpec, MetricsResult
from rc_bench.core.metrics import (
    mae,
    mse,
    rmse,
    nrmse_range,
    nrmse_std,
    nrmse_var,
    prediction_horizon,
)
from rc_bench.protocol.state_collector import StateCollector
from rc_bench.protocol.forecasting import (
    build_one_step_targets,
    build_fixed_horizon_targets,
    closed_loop_predict,
)
from rc_bench.readout.ridge import select_alpha, RidgeReadout


def _scale_inputs(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaler_name.lower() == "zscore":
        scaler = StandardScaler()
        return (
            scaler.fit_transform(X_train),
            scaler.transform(X_val),
            scaler.transform(X_test),
        )
    return X_train, X_val, X_test


def run_experiment(
    data: Dict[str, np.ndarray],
    spec: ExperimentSpec,
    reservoir: BaseReservoir,
) -> Dict[str, Any]:
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    washout: int = spec.protocol.washout
    alpha_grid = spec.readout.alpha_grid
    mode: str = spec.protocol.forecasting_mode
    scaler_name: str = spec.reservoir.params.get("scaler", reservoir.DEFAULT_SCALER)

    X_train_s, X_val_s, X_test_s = _scale_inputs(X_train, X_val, X_test, scaler_name)

    tracemalloc.start()
    t_train_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Collect reservoir states
    # ------------------------------------------------------------------
    collector = StateCollector(reservoir, washout)

    H_train_full = reservoir.transform(X_train_s)
    _check_sanity(reservoir, H_train_full)
    H_val_full = reservoir.transform(X_val_s)

    H_train = H_train_full[washout:]
    H_val = H_val_full[washout:]
    y_train_eff = collector.trim(y_train)
    y_val_eff = collector.trim(y_val)

    # ------------------------------------------------------------------
    # Build (H, y) pairs for the chosen forecasting mode
    # ------------------------------------------------------------------
    if mode == "one_step":
        H_tr, y_tr = build_one_step_targets(H_train, y_train_eff)
        H_v,  y_v  = build_one_step_targets(H_val,   y_val_eff)
    elif mode == "fixed_horizon":
        h = spec.protocol.horizon
        H_tr, y_tr = build_fixed_horizon_targets(H_train, y_train_eff, h)
        H_v,  y_v  = build_fixed_horizon_targets(H_val,   y_val_eff,   h)
    elif mode == "closed_loop":
        # Train teacher-forced (same as one_step)
        H_tr, y_tr = build_one_step_targets(H_train, y_train_eff)
        H_v,  y_v  = build_one_step_targets(H_val,   y_val_eff)
    else:
        raise ValueError(f"Unknown forecasting_mode: {mode!r}")

    # ------------------------------------------------------------------
    # Select alpha on validation, fit on train+val
    # ------------------------------------------------------------------
    alpha_info = select_alpha(H_tr, y_tr, H_v, y_v, alpha_grid)
    best_alpha = alpha_info["alpha"]

    H_tv = np.vstack([H_tr, H_v])
    y_tv = np.concatenate([y_tr, y_v])
    readout = RidgeReadout(best_alpha)
    readout.fit(H_tv, y_tv)
    t_train_end = time.perf_counter()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    t_infer_start = time.perf_counter()

    y_test_eff = collector.trim(y_test)

    if mode == "one_step":
        H_test_full = reservoir.transform(X_test_s)
        H_test = H_test_full[washout:]
        H_te, y_te = build_one_step_targets(H_test, y_test_eff)
        y_test_pred = readout.predict(H_te)

    elif mode == "fixed_horizon":
        h = spec.protocol.horizon
        H_test_full = reservoir.transform(X_test_s)
        H_test = H_test_full[washout:]
        H_te, y_te = build_fixed_horizon_targets(H_test, y_test_eff, h)
        y_test_pred = readout.predict(H_te)

    elif mode == "closed_loop":
        # Warm up on the washout window of the test set
        X_test_seed = X_test_s[:washout] if washout > 0 else X_test_s[:1]
        y_te = y_test_eff
        y_test_pred = closed_loop_predict(readout, reservoir, X_test_seed, len(y_te))

    t_infer_end = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = MetricsResult(
        rmse=rmse(y_te, y_test_pred),
        nrmse_range=nrmse_range(y_te, y_test_pred),
        nrmse_std=nrmse_std(y_te, y_test_pred),
        nrmse_var=nrmse_var(y_te, y_test_pred),
        mae=mae(y_te, y_test_pred),
        mse=mse(y_te, y_test_pred),
        prediction_horizon=prediction_horizon(y_te, y_test_pred),
        val_nrmse_range=alpha_info["val_nrmse"],
        train_time=t_train_end - t_train_start,
        inference_latency=t_infer_end - t_infer_start,
        peak_memory=peak_bytes,
    )

    return {
        "metrics": metrics,
        "best_alpha": best_alpha,
        "preds": y_test_pred,
        "y_test": y_te,
    }


def _check_sanity(reservoir: BaseReservoir, H: np.ndarray) -> None:
    for check_name, passed in reservoir.sanity_check(H).items():
        if not passed:
            warnings.warn(
                f"{type(reservoir).__name__} failed sanity check: {check_name}",
                RuntimeWarning,
                stacklevel=3,
            )
