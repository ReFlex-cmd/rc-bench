"""JMLC per-cell profiling pass (PROF-003).

Wires the generic ``rc_bench.profiling`` utilities (latency, memory, hardware)
into an evidence-producing pass over the JMLC experiment-matrix RunRecords:
for every ``<runs>/<family>_<model>_h<horizon>.json`` cell, this module

1. rebuilds and fits the exact model described by the RunRecord's
   ``resolved_spec`` (the already-HPO-resolved configuration that was
   actually evaluated) — fitting happens OUTSIDE the timed region;
2. defines a single-sample inference ``step_fn`` for that model family and
   measures it with :func:`rc_bench.profiling.latency.measure_latency`;
3. measures isolated peak RSS of a fresh build+fit+inference-loop with
   :func:`rc_bench.profiling.memory.measure_peak_rss_subprocess`, and reports
   the "honesty" delta against the parent's current RSS (the child inherits
   the parent's residency via ``fork()``);
4. records serialized-model and working-state sizes.

A cell that cannot be profiled (unknown/mismatched spec, a model that fails
to build or fit, a reservoir type without ``step()``, ...) is recorded with
``status: "FAILED"`` and an ``error`` string — never silently dropped, never
allowed to crash the rest of the pass (mirrors
``rc_bench.runners.jmlc_matrix.run_matrix``).

Deliberately does not import from ``rc_bench.runners.experiment_runner``,
``rc_bench.runners.baseline_runner`` or ``rc_bench.runners.jmlc_matrix``: this
module re-implements the small pieces of their fit/alignment logic it needs
by calling the same PUBLIC functions those runners call
(``select_alpha``, ``build_fixed_horizon_targets``, ``ar_lag_features``,
``select_and_fit_ridge_ar``, ...), so it has no dependency on those modules'
internal (possibly concurrently-changing) helpers.
"""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from rc_bench.core.baselines import RIDGE_AR_LAGS, SEASONAL_PERIOD, ar_lag_features, select_and_fit_ridge_ar
from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.reservoirs.base import BaseReservoir
from rc_bench.core.reservoirs.registry import get_reservoir
from rc_bench.core.schema import EnergyResult, ExperimentSpec
from rc_bench.profiling.activity import (
    count_step_operations,
    spiking_activity,
    state_sparsity,
)
from rc_bench.profiling.energy import energy_backend_status, measure_energy
from rc_bench.profiling.hardware import get_hardware_profile
from rc_bench.profiling.latency import measure_latency
from rc_bench.profiling.memory import (
    current_rss_bytes,
    measure_peak_rss_subprocess,
    serialized_model_bytes,
    working_state_bytes,
)
from rc_bench.protocol.forecasting import (
    aligned_target_positions,
    build_fixed_horizon_targets,
    build_one_step_targets,
)
from rc_bench.readout.ridge import RidgeReadout, select_alpha
from rc_bench.reporting.run_record import RunRecord, load_run_record

# src/rc_bench/profiling/model_profiles.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]

_FILENAME_RE = re.compile(r"^(?P<family>[a-z0-9]+)_(?P<model>[a-z0-9_]+)_h(?P<horizon>\d+)\.json$")


class ProfilingCellError(RuntimeError):
    """Raised when a single matrix cell cannot be profiled.

    Always caught by the per-cell driver and turned into a ``status: "FAILED"``
    summary/artifact entry — never propagates out of :func:`run_profiling_pass`.
    """


# ---------------------------------------------------------------------------
# Step-function builders (one single-sample inference step; no fitting inside)
# ---------------------------------------------------------------------------


def build_reservoir_step_fn(
    reservoir: BaseReservoir,
    readout: RidgeReadout,
    X_scaled: np.ndarray,
) -> Callable[[], float]:
    """One reservoir inference step: ``reservoir.step(x_t) -> h``, then
    ``readout.predict(h[None, :]) -> scalar``.

    Inputs cycle over ``X_scaled`` (already scaled the same way the model was
    trained) with a wrapping index, so the step function can run indefinitely
    regardless of ``n_steps``. Call ``reservoir.reset_state()`` before timing
    starts; the returned callable never resets state itself, so
    ``measure_latency``'s warmup calls are real steps that advance it.

    The returned callable exposes ``.last_state["h"]`` (the most recent state
    vector) so callers can size the working state without an extra step
    outside the timed region.
    """
    X_scaled = np.asarray(X_scaled)
    n = len(X_scaled)
    if n == 0:
        raise ValueError("X_scaled must be non-empty to build a reservoir step function")
    idx = {"i": 0}
    last_state: Dict[str, Any] = {"h": None}

    def step_fn() -> float:
        x_t = X_scaled[idx["i"] % n]
        h = reservoir.step(np.asarray(x_t).reshape(1, -1))
        last_state["h"] = h
        pred = readout.predict(np.asarray(h).reshape(1, -1))
        idx["i"] += 1
        return float(np.asarray(pred).reshape(-1)[0])

    step_fn.last_state = last_state  # type: ignore[attr-defined]
    return step_fn


def build_reservoir_compute_step_fn(
    reservoir: BaseReservoir,
    readout: RidgeReadout,
    X_scaled: np.ndarray,
) -> Callable[[], float]:
    """Deployment-representative reservoir step: state update + ``coef @ h + b``.

    Identical to :func:`build_reservoir_step_fn` except that the readout is
    evaluated as plain arithmetic instead of through ``sklearn``'s per-sample
    ``predict``, whose input validation costs tens of microseconds and would
    otherwise dominate — and distort — every cross-family latency comparison.
    The predictions of the two paths agree to floating-point tolerance; see
    ``tests/test_model_profiles.py``.
    """
    X_scaled = np.asarray(X_scaled)
    n = len(X_scaled)
    if n == 0:
        raise ValueError("X_scaled must be non-empty to build a reservoir step function")
    coef, intercept = readout.coefficients()
    idx = {"i": 0}
    last_state: Dict[str, Any] = {"h": None}

    def step_fn() -> float:
        x_t = X_scaled[idx["i"] % n]
        h = reservoir.step(np.asarray(x_t).reshape(1, -1))
        last_state["h"] = h
        idx["i"] += 1
        return float(coef @ np.asarray(h).reshape(-1) + intercept)

    step_fn.last_state = last_state  # type: ignore[attr-defined]
    return step_fn


def build_ridge_ar_compute_step_fn(
    values: np.ndarray,
    tau: int,
    horizon: int,
    scaler: StandardScaler,
    model: Ridge,
    n_lags: int = RIDGE_AR_LAGS,
) -> Callable[[], float]:
    """Deployment-representative Ridge-AR step.

    Same ring buffer and same fitted parameters as
    :func:`build_ridge_ar_step_fn`, with ``scaler.transform`` and
    ``model.predict`` replaced by the arithmetic they perform
    (``(row - mean) / scale``, then ``coef @ row + intercept``). Without this,
    Ridge AR pays sklearn's per-call overhead twice and measures as the slowest
    model in the matrix — a property of the API, not of AR(24).
    """
    values = np.asarray(values, dtype=float)
    if n_lags < 1:
        raise ValueError("n_lags must be a positive integer")
    window = horizon + n_lags - 1
    if tau < window:
        raise ValueError("tau must be >= horizon + n_lags - 1: insufficient causal history")
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")

    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])

    buf: "deque[float]" = deque(values[tau - window : tau], maxlen=window)
    idx = {"i": tau % n}

    def step_fn() -> float:
        origin_window = list(buf)[:n_lags]
        row = np.asarray(origin_window[::-1], dtype=float)
        row_scaled = (row - mean) / scale
        new_val = values[idx["i"] % n]
        buf.append(new_val)
        idx["i"] += 1
        return float(coef @ row_scaled + intercept)

    step_fn.buffer = buf  # type: ignore[attr-defined]
    return step_fn


def _build_lag_step_fn(values: np.ndarray, tau: int, lag: int) -> Callable[[], float]:
    """Shared ring-buffer mechanics for persistence / seasonal persistence.

    ``y_hat = history[-lag]``, then push the new observation. The buffer is a
    fixed-size deque of the last ``lag`` observed values (oldest first); its
    leftmost (oldest) entry is exactly ``lag`` steps behind the value about to
    be pushed, which is what "history[-lag]" means for a causal ring buffer.
    """
    values = np.asarray(values, dtype=float)
    if lag < 1:
        raise ValueError("lag must be a positive integer")
    if tau < lag:
        raise ValueError("tau must be >= lag: insufficient causal history")
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    buf: "deque[float]" = deque(values[tau - lag : tau], maxlen=lag)
    idx = {"i": tau % n}

    def step_fn() -> float:
        y_hat = buf[0]
        new_val = values[idx["i"] % n]
        buf.append(new_val)
        idx["i"] += 1
        return float(y_hat)

    step_fn.buffer = buf  # type: ignore[attr-defined]
    return step_fn


def build_persistence_step_fn(values: np.ndarray, tau: int, horizon: int) -> Callable[[], float]:
    """One persistence step: read ``history[-horizon]``, then push the new
    observation. One step = one lookup + one buffer update."""
    return _build_lag_step_fn(values, tau, horizon)


def build_seasonal_persistence_step_fn(values: np.ndarray, tau: int, season: int) -> Callable[[], float]:
    """One seasonal-persistence step: read ``history[-season]``, then push the
    new observation."""
    return _build_lag_step_fn(values, tau, season)


def build_ridge_ar_step_fn(
    values: np.ndarray,
    tau: int,
    horizon: int,
    scaler: StandardScaler,
    model: Ridge,
    n_lags: int = RIDGE_AR_LAGS,
) -> Callable[[], float]:
    """One Ridge-AR step: assemble the ``n_lags``-lag feature row at the
    forecast origin (``ar_lag_features`` semantics: most recent first) from
    the causal ring buffer, ``scaler.transform``, ``model.predict``, then push
    the new observation.

    The buffer holds ``horizon + n_lags - 1`` values: at any point its OLDEST
    ``n_lags`` entries are exactly the ``ar_lag_features`` window for the
    value about to be pushed (target position ``tau``, origin ``tau -
    horizon``) — see ``tests/test_model_profiles.py`` for the exact-match
    check against ``ar_lag_features``.
    """
    values = np.asarray(values, dtype=float)
    if n_lags < 1:
        raise ValueError("n_lags must be a positive integer")
    window = horizon + n_lags - 1
    if tau < window:
        raise ValueError(
            "tau must be >= horizon + n_lags - 1: insufficient causal history"
        )
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty")
    buf: "deque[float]" = deque(values[tau - window : tau], maxlen=window)
    idx = {"i": tau % n}

    def step_fn() -> float:
        origin_window = list(buf)[:n_lags]  # oldest n_lags entries, oldest first
        row = np.asarray(origin_window[::-1], dtype=float).reshape(1, -1)  # most recent first
        row_scaled = scaler.transform(row)
        pred = model.predict(row_scaled)
        new_val = values[idx["i"] % n]
        buf.append(new_val)
        idx["i"] += 1
        return float(np.asarray(pred).reshape(-1)[0])

    step_fn.buffer = buf  # type: ignore[attr-defined]
    return step_fn


# ---------------------------------------------------------------------------
# Fitting (mirrors experiment_runner/baseline_runner protocol; public API only)
# ---------------------------------------------------------------------------


def _observed_mask(data: Mapping[str, Any], split: str, length: int) -> np.ndarray:
    key = f"target_observed_mask_{split}"
    if key in data:
        return np.asarray(data[key], dtype=bool)
    return np.ones(length, dtype=bool)


def _split_positions(
    data: Mapping[str, Any], split: str, washout: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data[f"y_{split}"], dtype=float)
    mask = _observed_mask(data, split, len(values))
    positions = aligned_target_positions(mask, washout, horizon, "fixed_horizon")
    if positions.size == 0:
        raise ProfilingCellError(
            f"{split} has zero observed targets after washout/horizon alignment"
        )
    return values, positions


def _fit_reservoir_readout(
    data: Mapping[str, Any],
    spec: ExperimentSpec,
    reservoir: BaseReservoir,
) -> Tuple[RidgeReadout, Optional[StandardScaler], np.ndarray]:
    """Fit a readout for ``reservoir`` on train+val, mirroring
    ``experiment_runner``'s protocol (train-only scaler, washout trim,
    horizon alignment, alpha selection on val, refit on train+val).

    Returns ``(readout, scaler, X_test_scaled)``.
    """
    washout = spec.protocol.washout
    mode = spec.protocol.forecasting_mode
    horizon = spec.protocol.horizon

    scaler_name = spec.reservoir.params.get("scaler", reservoir.DEFAULT_SCALER)
    X_train = np.asarray(data["X_train"])
    X_val = np.asarray(data["X_val"])
    X_test = np.asarray(data["X_test"])

    scaler: Optional[StandardScaler] = None
    if str(scaler_name).lower() == "zscore":
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

    H_train_full = reservoir.transform(X_train_s)
    H_val_full = reservoir.transform(X_val_s)

    y_train = np.asarray(data["y_train"], dtype=float)[washout:]
    y_val = np.asarray(data["y_val"], dtype=float)[washout:]
    H_train = H_train_full[washout:]
    H_val = H_val_full[washout:]

    if mode == "fixed_horizon":
        H_train, y_train = build_fixed_horizon_targets(H_train, y_train, horizon)
        H_val, y_val = build_fixed_horizon_targets(H_val, y_val, horizon)
    else:  # one_step / closed_loop: both train teacher-forced (identity alignment)
        H_train, y_train = build_one_step_targets(H_train, y_train)
        H_val, y_val = build_one_step_targets(H_val, y_val)

    alpha_info = select_alpha(
        H_train, y_train, H_val, y_val,
        spec.readout.alpha_grid,
        metric=spec.protocol.selection_metric,
    )
    best_alpha = alpha_info["alpha"]

    H_tv = np.vstack([H_train, H_val])
    y_tv = np.concatenate([y_train, y_val])
    readout = RidgeReadout(best_alpha)
    readout.fit(H_tv, y_tv)

    return readout, scaler, X_test_s


def _fit_ridge_ar(data: Mapping[str, Any], spec: ExperimentSpec):
    """Fit Ridge AR on train+val, mirroring ``baseline_runner``'s protocol.

    Returns the ``RidgeARFit`` (exposing ``.scaler`` and ``.final_model``).
    """
    horizon = spec.protocol.horizon
    washout = spec.protocol.washout
    n_lags = RIDGE_AR_LAGS

    train_values, train_pos = _split_positions(data, "train", washout, horizon)
    val_values, val_pos = _split_positions(data, "val", washout, horizon)
    test_values, test_pos = _split_positions(data, "test", washout, horizon)

    X_train = ar_lag_features(train_values, train_pos, horizon, n_lags)
    X_val = ar_lag_features(val_values, val_pos, horizon, n_lags)
    X_test = ar_lag_features(test_values, test_pos, horizon, n_lags)

    return select_and_fit_ridge_ar(
        X_train, train_values[train_pos],
        X_val, val_values[val_pos],
        X_test, spec.readout.alpha_grid,
    )


# ---------------------------------------------------------------------------
# Per-cell profiling
# ---------------------------------------------------------------------------


def _protocol_block(
    warmup: int,
    n_steps: int,
    step_description: str,
    deployable_description: str,
) -> Dict[str, Any]:
    return {
        "warmup_steps": warmup,
        "n_steps": n_steps,
        "single_threaded": True,
        "timer": "perf_counter_ns",
        "step_definition": step_description,
        "deployable_step_definition": deployable_description,
        "fit_excluded_from_timing": True,
    }


# Persistence-family steps never call into sklearn, so their two measurements
# describe the same code; saying so is clearer than pretending they differ.
_SAME_AS_IMPLEMENTED = (
    "identical to step_definition: this model performs no framework call, so "
    "the as-implemented and deployable paths are the same code"
)


def _compute_sizes(model_obj: Any, state_obj: Any) -> Dict[str, Any]:
    try:
        model_bytes: Optional[int] = serialized_model_bytes(model_obj)
        serialization_error: Optional[str] = None
    except Exception as exc:  # noqa: BLE001 — record, never crash the cell
        model_bytes = None
        serialization_error = f"{type(exc).__name__}: {exc}"

    result: Dict[str, Any] = {
        "serialized_model_bytes": model_bytes,
        "working_state_bytes": working_state_bytes(state_obj),
    }
    if serialization_error is not None:
        result["serialization_error"] = serialization_error
    return result


def _measure_memory(build_and_run: Callable[[], Any]) -> Dict[str, Any]:
    parent_rss, _method = current_rss_bytes()
    profile = measure_peak_rss_subprocess(build_and_run)
    delta = profile.peak_rss_bytes - parent_rss
    return {
        "peak_rss_bytes": profile.peak_rss_bytes,
        "method": profile.method,
        "platform": profile.platform,
        "parent_rss_at_fork_bytes": parent_rss,
        "peak_rss_delta_bytes": delta,
    }


#: Окно одного энергетического измерения, с. Счётчик RAPL обновляется с
#: периодом порядка миллисекунд, поэтому окно должно быть на порядки длиннее;
#: две секунды дают запас и на медленный шаг, и на шумного соседа по машине.
DEFAULT_ENERGY_WINDOW_S = 2.0

_ACTIVITY_NOTE = (
    "analytic/observed proxies; never converted into energy (DEC-007) — "
    "see the energy block for the measured value"
)


def _checked_energy(measurement: Dict[str, Any]) -> Dict[str, Any]:
    """Прогнать измерение через EnergyResult перед публикацией.

    Схема запрещает записи, у которых статус и числа расходятся, но до этой
    проверки её никто не применял к артефакту профиля — а именно он и несёт
    измеренную энергию. Без неё контракт держался только на честности
    measure_energy, то есть не держался.

    Возвращается исходный словарь: в нём есть диагностика (длительность окна,
    число шагов, энергия простоя), которой в схеме нет и которая нужна тому,
    кто будет разбираться с сомнительным числом.
    """
    # Берутся только реально присутствующие ключи: отсутствующий — это не
    # None, а «поле не заполнялось», и значения по умолчанию у схемы свои.
    EnergyResult.model_validate(
        {key: measurement[key] for key in EnergyResult.model_fields if key in measurement}
    )
    return measurement


def _baseline_activity(*, readout_macs: int, description: str) -> Dict[str, Any]:
    """Блок activity для детерминированного baseline.

    Резервуара здесь нет, поэтому все резервуарные счётчики — честные нули, а
    не «неизвестно»: у модели действительно нет ни рекуррентных весов, ни
    нелинейностей. Разреженность состояния, наоборот, ``None``: рабочее
    состояние baseline — это окно входного ряда, и доля нулей в нём говорит о
    данных, а не о стоимости шага.
    """
    return {
        "operations": {
            "backend": "analytic",
            "reservoir_macs": 0,
            "reservoir_nonlinearities": 0,
            "reservoir_nonzero_recurrent_weights": 0,
            "readout_macs": int(readout_macs),
            "total_macs": int(readout_macs),
            "note": f"analytic MAC estimate ({description}); not an energy measurement",
        },
        "state_sparsity": None,
        "spiking": None,
        "note": _ACTIVITY_NOTE,
    }


def _profile_reservoir_cell(
    spec: ExperimentSpec,
    data: Mapping[str, Any],
    *,
    n_steps: int,
    warmup: int,
    energy_window_s: float,
) -> Dict[str, Any]:
    reservoir_config = {"seed": spec.seed, **spec.reservoir.params}
    reservoir = get_reservoir(spec.reservoir.type, reservoir_config)
    readout, _scaler, X_test_s = _fit_reservoir_readout(data, spec, reservoir)

    if len(X_test_s) == 0:
        raise ProfilingCellError("X_test is empty; cannot build a reservoir step function")

    reservoir.reset_state()
    step_fn = build_reservoir_step_fn(reservoir, readout, X_test_s)
    step_description = (
        "reservoir.step(x_t) -> h; readout.predict(h[None, :]) -> scalar; "
        "inputs cycle over the scaled test split with a wrapping index"
    )
    latency = measure_latency(
        step_fn, warmup=warmup, n_steps=n_steps, keep_raw=True, single_threaded=True
    )

    reservoir.reset_state()
    deployable_step_fn = build_reservoir_compute_step_fn(reservoir, readout, X_test_s)
    deployable_latency = measure_latency(
        deployable_step_fn, warmup=warmup, n_steps=n_steps, keep_raw=True, single_threaded=True
    )

    h_sample = step_fn.last_state["h"]  # type: ignore[attr-defined]
    sizes = _compute_sizes(
        model_obj={"reservoir": reservoir, "readout": readout},
        state_obj=h_sample,
    )

    def build_and_run() -> None:
        fresh_reservoir = get_reservoir(spec.reservoir.type, reservoir_config)
        fresh_readout, _, fresh_X_test_s = _fit_reservoir_readout(data, spec, fresh_reservoir)
        fresh_reservoir.reset_state()
        fresh_step = build_reservoir_step_fn(fresh_reservoir, fresh_readout, fresh_X_test_s)
        for _ in range(n_steps):
            fresh_step()

    memory = _measure_memory(build_and_run)

    # Активность снимается до энергетического окна: спайковые счётчики
    # обнуляются на reset_state(), и после перезапуска резервуара они
    # описывали бы уже другой прогон.
    activity = {
        "operations": count_step_operations(
            reservoir, readout_input_dim=int(np.asarray(h_sample).reshape(-1).size)
        ),
        "state_sparsity": state_sparsity(np.asarray(h_sample)),
        "spiking": spiking_activity(reservoir),
        "note": _ACTIVITY_NOTE,
    }

    reservoir.reset_state()
    energy = _checked_energy(
        measure_energy(
            build_reservoir_compute_step_fn(reservoir, readout, X_test_s),
            p50_ns=deployable_latency.p50_ns,
            min_duration_s=energy_window_s,
        )
    )

    return {
        "protocol": _protocol_block(
            warmup,
            n_steps,
            step_description,
            "reservoir.step(x_t) -> h; coef @ h + intercept (the fitted readout "
            "evaluated as arithmetic, without sklearn's per-sample predict API)",
        ),
        "latency": latency.to_dict(),
        "latency_deployable": deployable_latency.to_dict(),
        "memory": memory,
        "sizes": sizes,
        "activity": activity,
        "energy": energy,
    }


def _profile_ridge_ar_cell(
    spec: ExperimentSpec,
    data: Mapping[str, Any],
    *,
    n_steps: int,
    warmup: int,
    energy_window_s: float,
) -> Dict[str, Any]:
    horizon = spec.protocol.horizon
    n_lags = RIDGE_AR_LAGS

    fit = _fit_ridge_ar(data, spec)
    values = np.asarray(data["y_test"], dtype=float)
    window = horizon + n_lags - 1
    if len(values) <= window:
        raise ProfilingCellError(
            f"y_test too short ({len(values)}) for ridge_ar window={window}"
        )
    tau = window
    step_fn = build_ridge_ar_step_fn(values, tau, horizon, fit.scaler, fit.final_model, n_lags=n_lags)
    step_description = (
        f"assemble {n_lags}-lag AR feature row (most-recent-first) at the "
        "forecast origin from the causal ring buffer, scaler.transform, "
        "model.predict, then push the new observation"
    )
    latency = measure_latency(
        step_fn, warmup=warmup, n_steps=n_steps, keep_raw=True, single_threaded=True
    )

    deployable_step_fn = build_ridge_ar_compute_step_fn(
        values, tau, horizon, fit.scaler, fit.final_model, n_lags=n_lags
    )
    deployable_latency = measure_latency(
        deployable_step_fn, warmup=warmup, n_steps=n_steps, keep_raw=True, single_threaded=True
    )

    state_array = np.array(step_fn.buffer, dtype=np.float64)  # type: ignore[attr-defined]
    sizes = _compute_sizes(
        model_obj={"scaler": fit.scaler, "model": fit.final_model},
        state_obj=state_array,
    )

    def build_and_run() -> None:
        fresh_fit = _fit_ridge_ar(data, spec)
        fresh_step = build_ridge_ar_step_fn(
            values, tau, horizon, fresh_fit.scaler, fresh_fit.final_model, n_lags=n_lags
        )
        for _ in range(n_steps):
            fresh_step()

    memory = _measure_memory(build_and_run)

    # Шаг ridge_ar: масштабирование строки признаков ((row - mean) / scale —
    # одно скалярно-векторное умножение на n_lags) и скалярное произведение с
    # коэффициентами (ещё n_lags).
    activity = _baseline_activity(
        readout_macs=2 * n_lags,
        description=f"{n_lags}-lag feature scaling plus the ridge dot product",
    )
    energy = _checked_energy(
        measure_energy(
            build_ridge_ar_compute_step_fn(
                values, tau, horizon, fit.scaler, fit.final_model, n_lags=n_lags
            ),
            p50_ns=deployable_latency.p50_ns,
            min_duration_s=energy_window_s,
        )
    )

    return {
        "protocol": _protocol_block(
            warmup,
            n_steps,
            step_description,
            f"same {n_lags}-lag buffer, with scaler.transform and model.predict "
            "replaced by the arithmetic they perform ((row - mean) / scale, then "
            "coef @ row + intercept)",
        ),
        "latency": latency.to_dict(),
        "latency_deployable": deployable_latency.to_dict(),
        "memory": memory,
        "sizes": sizes,
        "activity": activity,
        "energy": energy,
    }


def _profile_persistence_family_cell(
    spec: ExperimentSpec,
    data: Mapping[str, Any],
    *,
    n_steps: int,
    warmup: int,
    energy_window_s: float,
) -> Dict[str, Any]:
    model = spec.model_type
    horizon = spec.protocol.horizon
    season = spec.protocol.seasonal_period or SEASONAL_PERIOD

    if model == "persistence":
        lag = horizon
        builder = build_persistence_step_fn
        lag_name = "horizon"
    elif model == "seasonal_persistence":
        lag = season
        builder = build_seasonal_persistence_step_fn
        lag_name = "seasonal_period"
    else:  # pragma: no cover - dispatch guard
        raise ProfilingCellError(f"not a persistence-family model: {model!r}")

    values = np.asarray(data["y_test"], dtype=float)
    if len(values) <= lag:
        raise ProfilingCellError(f"y_test too short ({len(values)}) for {model} lag={lag}")
    tau = lag
    step_fn = builder(values, tau, lag)
    step_description = (
        f"read history[-{lag_name}] (lag={lag}) from the causal ring buffer, "
        "then push the new observation"
    )
    latency = measure_latency(
        step_fn, warmup=warmup, n_steps=n_steps, keep_raw=True, single_threaded=True
    )

    state_array = np.array(step_fn.buffer, dtype=np.float64)  # type: ignore[attr-defined]
    sizes = _compute_sizes(model_obj=state_array, state_obj=state_array)

    def build_and_run() -> None:
        fresh_step = builder(values, tau, lag)
        for _ in range(n_steps):
            fresh_step()

    memory = _measure_memory(build_and_run)

    # Прогноз — это чтение элемента буфера; арифметики в шаге нет вообще.
    activity = _baseline_activity(
        readout_macs=0,
        description="a buffer lookup performs no multiply-accumulate at all",
    )
    energy = _checked_energy(
        measure_energy(
            builder(values, tau, lag),
            p50_ns=latency.p50_ns,
            min_duration_s=energy_window_s,
        )
    )

    return {
        "protocol": _protocol_block(warmup, n_steps, step_description, _SAME_AS_IMPLEMENTED),
        "latency": latency.to_dict(),
        "latency_deployable": latency.to_dict(),
        "memory": memory,
        "sizes": sizes,
        "activity": activity,
        "energy": energy,
    }


def _profile_cell_body(
    family: str,
    model: str,
    spec: ExperimentSpec,
    data: Mapping[str, Any],
    *,
    n_steps: int,
    warmup: int,
    energy_window_s: float = DEFAULT_ENERGY_WINDOW_S,
) -> Dict[str, Any]:
    kwargs = {"n_steps": n_steps, "warmup": warmup, "energy_window_s": energy_window_s}
    if family == "reservoir":
        return _profile_reservoir_cell(spec, data, **kwargs)
    if model == "ridge_ar":
        return _profile_ridge_ar_cell(spec, data, **kwargs)
    if model in ("persistence", "seasonal_persistence"):
        return _profile_persistence_family_cell(spec, data, **kwargs)
    raise ProfilingCellError(f"no profiling strategy for family={family!r} model={model!r}")


# ---------------------------------------------------------------------------
# Path sanitization (DEC-008 / release gate: no absolute paths in artifacts)
# ---------------------------------------------------------------------------


def _sanitize_run_record_path(path: "str | Path") -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _parse_filename(name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    match = _FILENAME_RE.match(name)
    if not match:
        return None, None, None
    return match.group("family"), match.group("model"), int(match.group("horizon"))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _failed_entry(
    family: Optional[str],
    model: Optional[str],
    horizon: Optional[int],
    config_hash: Optional[str],
    run_path: Path,
    exc: BaseException,
) -> Dict[str, Any]:
    return {
        "family": family,
        "model": model,
        "horizon": horizon,
        "config_hash": config_hash,
        "run_record": _sanitize_run_record_path(run_path),
        "status": "FAILED",
        "error": f"{type(exc).__name__}: {exc}",
    }


def _profile_one_cell(
    run_path: Path,
    data: Mapping[str, Any],
    *,
    n_steps: int,
    warmup: int,
    energy_window_s: float = DEFAULT_ENERGY_WINDOW_S,
) -> Dict[str, Any]:
    try:
        record: RunRecord = load_run_record(run_path)
    except Exception as exc:  # noqa: BLE001 — record, never drop
        family, model, horizon = _parse_filename(run_path.name)
        return _failed_entry(family, model, horizon, None, run_path, exc)

    resolved = record.resolved_spec
    family = record.result.model_family or resolved.model_family
    model = resolved.model_type
    horizon = resolved.protocol.horizon
    config_hash = record.result.config_hash

    try:
        expected_name = f"{family}_{model}_h{horizon}.json"
        if run_path.name != expected_name:
            raise ProfilingCellError(
                f"run file name {run_path.name!r} does not match its resolved "
                f"spec (expected {expected_name!r})"
            )

        cell = _profile_cell_body(
            family,
            model,
            resolved,
            data,
            n_steps=n_steps,
            warmup=warmup,
            energy_window_s=energy_window_s,
        )
        cell.update(
            family=family,
            model=model,
            horizon=horizon,
            config_hash=config_hash,
            run_record=_sanitize_run_record_path(run_path),
            status="completed",
        )
        return cell
    except Exception as exc:  # noqa: BLE001 — record, never drop
        return _failed_entry(family, model, horizon, config_hash, run_path, exc)


def _summary_entry(cell: Dict[str, Any]) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "family": cell["family"],
        "model": cell["model"],
        "horizon": cell["horizon"],
        "config_hash": cell["config_hash"],
        "status": cell["status"],
    }
    if cell["status"] == "FAILED":
        base["error"] = cell.get("error")
        return base

    latency = cell["latency"]
    deployable = cell["latency_deployable"]
    memory = cell["memory"]
    sizes = cell["sizes"]
    base.update(
        p50_ns=latency["p50_ns"],
        p95_ns=latency["p95_ns"],
        throughput_samples_per_s=latency["throughput_samples_per_s"],
        deployable_p50_ns=deployable["p50_ns"],
        deployable_p95_ns=deployable["p95_ns"],
        deployable_throughput_samples_per_s=deployable["throughput_samples_per_s"],
        peak_rss_bytes=memory["peak_rss_bytes"],
        peak_rss_delta_bytes=memory["peak_rss_delta_bytes"],
        serialized_model_bytes=sizes["serialized_model_bytes"],
        working_state_bytes=sizes["working_state_bytes"],
    )

    # Ключи выставляются всегда, даже пустые: отсутствие столбца в сводке
    # неотличимо от несобранного профиля, а None читается однозначно.
    energy = cell.get("energy") or {"status": "unavailable"}
    activity = cell.get("activity") or {}
    operations = activity.get("operations") or {}
    sparsity = activity.get("state_sparsity") or {}
    spiking = activity.get("spiking") or {}
    base.update(
        energy_status=energy.get("status"),
        energy_window_target_met=energy.get("window_target_met"),
        net_energy_per_inference_mj=energy.get("net_energy_per_inference_mj"),
        net_samples_per_joule=energy.get("net_samples_per_joule"),
        energy_delay_product_j_s=energy.get("energy_delay_product_j_s"),
        total_macs=operations.get("total_macs"),
        state_near_zero_fraction=sparsity.get("near_zero_fraction"),
        spikes_per_step=spiking.get("spikes_per_step"),
        synaptic_events_per_step=spiking.get("synaptic_events_per_step"),
    )
    return base


def _load_dataset_params(config_path: "str | Path") -> Dict[str, Any]:
    """Extract the dataset-loading parameters from the matrix template YAML,
    matching how ``rc_bench.runners.jmlc_matrix.run_matrix`` loads data."""
    raw = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("profiling config must be a mapping (a single ExperimentSpec template)")
    dataset = raw.get("dataset") or {}
    protocol = raw.get("protocol") or {}
    if "name" not in dataset:
        raise ValueError("profiling config must set dataset.name")
    return {
        "name": dataset["name"],
        "length": dataset.get("length", 2000),
        "train_frac": protocol.get("train_frac", 0.6),
        "val_frac": protocol.get("val_frac", 0.2),
    }


def _hardware_artifact(warmup: int, n_steps: int, energy_window_s: float) -> Dict[str, Any]:
    profile = get_hardware_profile().to_dict()
    profile["measurement_protocol"] = {
        "warmup_steps": warmup,
        "n_steps": n_steps,
        "timer": "perf_counter_ns",
        "single_threaded": True,
        # Окно энергетического измерения на ячейку: столько же длится прогон
        # под нагрузкой и столько же — измерение базовой линии простоя.
        "energy_window_s": energy_window_s,
        "fork_caveat": (
            "Peak RSS is measured in a forked child process, so it inherits "
            "the parent interpreter's resident set; see peak_rss_delta_bytes "
            "in each cell profile for the isolated contribution."
        ),
    }
    # Статус берётся с машины, а не из константы: артефакт должен говорить,
    # что на ней действительно есть, а не что было при написании кода.
    profile["energy"] = energy_backend_status()
    return profile


def run_profiling_pass(
    config_path: "str | Path",
    runs_dir: "str | Path",
    output_dir: "str | Path",
    hardware_output_path: "str | Path",
    *,
    n_steps: int = 1000,
    warmup: int = 100,
    energy_window_s: float = DEFAULT_ENERGY_WINDOW_S,
    cells: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Profile every RunRecord cell under ``runs_dir`` and write the JMLC
    evidence artifacts.

    Writes ``<output_dir>/<family>_<model>_h<horizon>.json`` per cell and
    ``<output_dir>/summary.json``, plus ``hardware_output_path`` (sanitized
    hardware/software profile + measurement protocol + energy-unavailable
    block). Failed cells are recorded (status ``"FAILED"`` + ``error``),
    never dropped, and never crash the pass. Returns the summary list.
    """
    dataset_params = _load_dataset_params(config_path)
    data = get_data_for_experiment(
        dataset_params["name"],
        length=dataset_params["length"],
        train_frac=dataset_params["train_frac"],
        val_frac=dataset_params["val_frac"],
    )

    runs_dir = Path(runs_dir)
    run_paths = sorted(runs_dir.glob("*.json"))
    if cells:
        # Accept either full run-file names or bare stems (e.g. both
        # "reservoir_esn_h1.json" and "reservoir_esn_h1" select the same cell).
        wanted = {c if c.endswith(".json") else f"{c}.json" for c in cells}
        run_paths = [p for p in run_paths if p.name in wanted]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    for run_path in run_paths:
        cell = _profile_one_cell(
            run_path,
            data,
            n_steps=n_steps,
            warmup=warmup,
            energy_window_s=energy_window_s,
        )
        family = cell["family"] or "unknown"
        model = cell["model"] or "unknown"
        horizon = cell["horizon"] if cell["horizon"] is not None else "x"
        cell_path = output_dir / f"{family}_{model}_h{horizon}.json"
        cell_path.write_text(json.dumps(cell, indent=2, ensure_ascii=False))
        summary.append(_summary_entry(cell))

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    hardware_output_path = Path(hardware_output_path)
    hardware_output_path.parent.mkdir(parents=True, exist_ok=True)
    hardware_output_path.write_text(
        json.dumps(
            _hardware_artifact(warmup, n_steps, energy_window_s),
            indent=2,
            ensure_ascii=False,
        )
    )

    return summary
