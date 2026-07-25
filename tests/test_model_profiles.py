"""Tests for rc_bench.profiling.model_profiles (PROF-003).

Design notes:
- Step-function builder tests (items 1-3) use small hand-built numpy arrays
  and stand-in reservoir/readout/model objects — no dataset loading, no
  fitting, fully deterministic and fast.
- The full-pass tests (items 4-5) use the ``narma10`` synthetic generator
  (already wired through ``get_data_for_experiment``) with a tiny length, and
  a tiny ``n_steps``/``warmup``, so the whole file stays well under the ~30s
  budget with no network and no real (UCI) dataset.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from rc_bench.core.baselines import ar_lag_features
from rc_bench.core.reservoirs.logistic_service import LogisticReservoir
from rc_bench.core.schema import (
    BaselineSpec,
    DatasetSpec,
    ExperimentSpec,
    ProtocolSpec,
    ReadoutSpec,
    ReservoirSpec,
    ResultSpec,
)
from rc_bench.profiling.model_profiles import (
    build_persistence_step_fn,
    build_reservoir_compute_step_fn,
    build_reservoir_step_fn,
    build_ridge_ar_compute_step_fn,
    build_ridge_ar_step_fn,
    build_seasonal_persistence_step_fn,
    run_profiling_pass,
)
from rc_bench.readout.ridge import RidgeReadout
from rc_bench.reporting.run_record import RunRecord, save_run_record

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Each of the 4 step-function builders: one finite scalar per call,
#    repeated calls advance state.
# ---------------------------------------------------------------------------


class _SumReadout:
    """Stand-in readout: predict = sum of the state vector. Avoids needing a
    real fit for a pure step-function-builder test."""

    def predict(self, H: np.ndarray) -> np.ndarray:
        return H.sum(axis=1)


def test_reservoir_step_fn_returns_finite_scalar_and_advances_state():
    reservoir = LogisticReservoir({"seed": 1, "units": 6})
    reservoir.reset_state()

    X = np.array([[0.1], [0.4], [0.9], [0.2], [0.7]])
    step_fn = build_reservoir_step_fn(reservoir, _SumReadout(), X)

    outputs = [step_fn() for _ in range(5)]
    assert all(isinstance(o, float) and np.isfinite(o) for o in outputs)
    # Non-constant input drives the logistic map to different states step to
    # step, so consecutive outputs must not all coincide.
    assert len(set(np.round(outputs, 9))) > 1


def test_persistence_step_fn_returns_finite_scalar_and_advances_state():
    values = np.arange(1.0, 21.0)  # 1..20
    step_fn = build_persistence_step_fn(values, tau=10, horizon=3)

    first = step_fn()
    second = step_fn()
    assert np.isfinite(first) and np.isfinite(second)
    # The ring buffer shifted by one real step, so the lookback value changed.
    assert first != second


def test_seasonal_persistence_step_fn_returns_finite_scalar_and_advances_state():
    values = np.sin(np.linspace(0, 8 * np.pi, 40)) + np.arange(40) * 0.01
    step_fn = build_seasonal_persistence_step_fn(values, tau=15, season=5)

    outputs = [step_fn() for _ in range(4)]
    assert all(isinstance(o, float) and np.isfinite(o) for o in outputs)
    assert len(set(np.round(outputs, 9))) > 1


def test_ridge_ar_step_fn_returns_finite_scalar_and_advances_state():
    n_lags = 4
    horizon = 2
    values = np.linspace(1.0, 30.0, 30)

    rng = np.random.default_rng(0)
    X_dummy = rng.normal(size=(20, n_lags))
    y_dummy = rng.normal(size=20)
    scaler = StandardScaler().fit(X_dummy)
    model = Ridge(alpha=1.0).fit(scaler.transform(X_dummy), y_dummy)

    step_fn = build_ridge_ar_step_fn(
        values, tau=10, horizon=horizon, scaler=scaler, model=model, n_lags=n_lags
    )
    outputs = [step_fn() for _ in range(4)]
    assert all(isinstance(o, float) and np.isfinite(o) for o in outputs)
    assert len(set(np.round(outputs, 9))) > 1


# ---------------------------------------------------------------------------
# 2. ridge_ar step features match ar_lag_features for the same origin.
# ---------------------------------------------------------------------------


class _CaptureScaler:
    """Records exactly what row was handed to transform(); returns it as-is."""

    def __init__(self) -> None:
        self.last_row: np.ndarray | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.last_row = np.array(X, copy=True)
        return X


class _ZeroModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


def test_ridge_ar_step_feature_row_matches_ar_lag_features_for_same_origin():
    n_lags = 5
    horizon = 3
    tau = 20
    values = np.linspace(0.0, 50.0, 60)

    scaler = _CaptureScaler()
    step_fn = build_ridge_ar_step_fn(
        values, tau=tau, horizon=horizon, scaler=scaler, model=_ZeroModel(), n_lags=n_lags
    )
    step_fn()

    expected = ar_lag_features(values, np.array([tau]), horizon, n_lags)
    assert scaler.last_row is not None
    np.testing.assert_array_equal(scaler.last_row, expected)


# ---------------------------------------------------------------------------
# 3. Persistence step output equals values[tau - horizon] for a known series.
# ---------------------------------------------------------------------------


def test_persistence_step_output_equals_known_lookback_value():
    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    tau = 5
    horizon = 2
    step_fn = build_persistence_step_fn(values, tau=tau, horizon=horizon)

    output = step_fn()
    assert output == pytest.approx(values[tau - horizon])


def test_seasonal_persistence_step_output_equals_known_lookback_value():
    values = np.array([float(i) for i in range(30)])
    tau = 12
    season = 7
    step_fn = build_seasonal_persistence_step_fn(values, tau=tau, season=season)

    output = step_fn()
    assert output == pytest.approx(values[tau - season])


# ---------------------------------------------------------------------------
# 4-5. Full pass over a fake runs/ directory
# ---------------------------------------------------------------------------


def _dataset_config_yaml(tmp_path: Path, length: int = 240) -> Path:
    config_path = tmp_path / "profiling_smoke.yaml"
    config_path.write_text(
        "dataset:\n"
        f"  name: narma10\n"
        f"  length: {length}\n"
        "protocol:\n"
        "  train_frac: 0.6\n"
        "  val_frac: 0.2\n"
    )
    return config_path


def _baseline_spec(model: str, *, horizon: int, season: int, washout: int, length: int) -> ExperimentSpec:
    return ExperimentSpec(
        dataset=DatasetSpec(name="narma10", length=length),
        baseline=BaselineSpec(type=model),
        protocol=ProtocolSpec(
            washout=washout,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode="fixed_horizon",
            horizon=horizon,
            n_seeds=0,
            use_hpo=False,
            selection_metric="nrmse_std",
            seasonal_period=season,
        ),
        readout=ReadoutSpec(alpha_grid=[0.01, 1.0]),
        seed=None,
    )


def _reservoir_spec(
    model: str, *, horizon: int, season: int, washout: int, length: int, params: dict
) -> ExperimentSpec:
    return ExperimentSpec(
        dataset=DatasetSpec(name="narma10", length=length),
        reservoir=ReservoirSpec(type=model, params=params),
        protocol=ProtocolSpec(
            washout=washout,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode="fixed_horizon",
            horizon=horizon,
            n_seeds=1,
            use_hpo=False,
            selection_metric="nrmse_std",
            seasonal_period=season,
        ),
        readout=ReadoutSpec(alpha_grid=[0.01, 1.0]),
        seed=7,
    )


def _write_cell(runs_dir: Path, family: str, model: str, horizon: int, spec: ExperimentSpec) -> Path:
    result = ResultSpec(
        status="completed",
        config_hash=spec.config_hash(),
        resolved_spec=spec,
        model_family=spec.model_family,
    )
    record = RunRecord.make(spec, result)
    path = runs_dir / f"{family}_{model}_h{horizon}.json"
    save_run_record(record, path)
    return path


_LENGTH = 240
_WASHOUT = 5
_HORIZON = 1
_SEASON = 6


def _two_good_cells(runs_dir: Path) -> None:
    persistence_spec = _baseline_spec(
        "persistence", horizon=_HORIZON, season=_SEASON, washout=_WASHOUT, length=_LENGTH
    )
    reservoir_spec = _reservoir_spec(
        "logistic",
        horizon=_HORIZON,
        season=_SEASON,
        washout=_WASHOUT,
        length=_LENGTH,
        params={"units": 6},
    )
    _write_cell(runs_dir, "baseline", "persistence", _HORIZON, persistence_spec)
    _write_cell(runs_dir, "reservoir", "logistic", _HORIZON, reservoir_spec)


class TestDeployableStepEquivalence:
    """The deployable step must predict what the as-implemented step predicts.

    The second latency measurement exists because sklearn's per-sample API costs
    ~30-40 us of validation overhead that dwarfs the arithmetic, so an edge
    deployment would use the fitted coefficients directly. That is only a
    measurement of the same model if it returns the same numbers — otherwise the
    faster path is simply computing something else.
    """

    def test_reservoir_paths_agree(self):
        rng = np.random.default_rng(7)
        reservoir = LogisticReservoir({"seed": 3, "units": 40})
        X = rng.standard_normal((30, 1))
        H = reservoir.transform(X)
        y = H @ rng.standard_normal(H.shape[1]) + 0.5

        readout = RidgeReadout(0.1)
        readout.fit(H, y)

        reservoir.reset_state()
        framework = build_reservoir_step_fn(reservoir, readout, X)
        framework_values = [framework() for _ in range(12)]

        reservoir.reset_state()
        deployable = build_reservoir_compute_step_fn(reservoir, readout, X)
        deployable_values = [deployable() for _ in range(12)]

        np.testing.assert_allclose(deployable_values, framework_values, rtol=1e-9, atol=1e-12)

    def test_ridge_ar_paths_agree(self):
        rng = np.random.default_rng(11)
        values = np.cumsum(rng.standard_normal(200)) + 50.0
        n_lags = 6
        horizon = 3
        tau = horizon + n_lags - 1

        design = np.stack([values[i : i + n_lags][::-1] for i in range(60)])
        target = values[n_lags : n_lags + 60]
        scaler = StandardScaler().fit(design)
        model = Ridge(alpha=0.5).fit(scaler.transform(design), target)

        framework = build_ridge_ar_step_fn(values, tau, horizon, scaler, model, n_lags=n_lags)
        deployable = build_ridge_ar_compute_step_fn(
            values, tau, horizon, scaler, model, n_lags=n_lags
        )

        framework_values = [framework() for _ in range(15)]
        deployable_values = [deployable() for _ in range(15)]

        np.testing.assert_allclose(deployable_values, framework_values, rtol=1e-9, atol=1e-12)


def test_full_pass_writes_sanitized_per_cell_and_summary_json(tmp_path):
    config_path = _dataset_config_yaml(tmp_path, length=_LENGTH)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    output_dir = tmp_path / "profiles"
    hardware_output = tmp_path / "hardware_profile.json"

    _two_good_cells(runs_dir)

    summary = run_profiling_pass(
        config_path=config_path,
        runs_dir=runs_dir,
        output_dir=output_dir,
        hardware_output_path=hardware_output,
        n_steps=5,
        warmup=2,
    )

    assert len(summary) == 2
    assert all(e["status"] == "completed" for e in summary), summary

    expected_keys = {
        "family", "model", "horizon", "config_hash", "status",
        "p50_ns", "p95_ns", "throughput_samples_per_s",
        "deployable_p50_ns", "deployable_p95_ns",
        "deployable_throughput_samples_per_s",
        "peak_rss_bytes", "peak_rss_delta_bytes",
        "serialized_model_bytes", "working_state_bytes",
    }
    for entry in summary:
        assert expected_keys <= entry.keys()

    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text()) == summary

    for family, model, horizon in [("baseline", "persistence", _HORIZON), ("reservoir", "logistic", _HORIZON)]:
        cell_path = output_dir / f"{family}_{model}_h{horizon}.json"
        assert cell_path.exists()
        text = cell_path.read_text()

        # Sanitization (DEC-008 / release gate): no absolute paths, no
        # hostname key, anywhere in the artifact text.
        assert "/home/" not in text
        assert str(tmp_path) not in text
        assert '"hostname"' not in text

        payload = json.loads(text)
        assert payload["status"] == "completed"
        # tmp_path lives outside the repo, so the sanitized run_record path
        # must fall back to just the basename (never an absolute path).
        assert payload["run_record"] == f"{family}_{model}_h{horizon}.json"
        assert payload["latency"]["raw_ns"]
        assert payload["memory"]["peak_rss_bytes"] > 0
        assert payload["memory"]["peak_rss_delta_bytes"] is not None
        assert payload["sizes"]["working_state_bytes"] > 0
        assert payload["protocol"]["fit_excluded_from_timing"] is True

    hw_text = hardware_output.read_text()
    assert "/home/" not in hw_text
    assert '"hostname"' not in hw_text
    hw_payload = json.loads(hw_text)
    assert hw_payload["energy"] == {
        "status": "unavailable",
        "reason": "no hardware energy counter available on this machine",
    }
    assert "measurement_protocol" in hw_payload
    assert hw_payload["measurement_protocol"]["n_steps"] == 5


def test_cell_that_fails_to_build_is_recorded_failed_and_pass_continues(tmp_path):
    config_path = _dataset_config_yaml(tmp_path, length=_LENGTH)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    output_dir = tmp_path / "profiles"
    hardware_output = tmp_path / "hardware_profile.json"

    good_spec = _baseline_spec(
        "persistence", horizon=_HORIZON, season=_SEASON, washout=_WASHOUT, length=_LENGTH
    )
    # `units` must be int-convertible; LogisticReservoir._build raises inside
    # int("not-a-number") — a genuine "model fails to build" cell.
    broken_spec = _reservoir_spec(
        "logistic",
        horizon=_HORIZON,
        season=_SEASON,
        washout=_WASHOUT,
        length=_LENGTH,
        params={"units": "not-a-number"},
    )
    _write_cell(runs_dir, "baseline", "persistence", _HORIZON, good_spec)
    _write_cell(runs_dir, "reservoir", "logistic", _HORIZON, broken_spec)

    summary = run_profiling_pass(
        config_path=config_path,
        runs_dir=runs_dir,
        output_dir=output_dir,
        hardware_output_path=hardware_output,
        n_steps=5,
        warmup=2,
    )

    assert len(summary) == 2  # both cells recorded, never dropped
    statuses = {(e["family"], e["model"]): e for e in summary}
    assert statuses[("baseline", "persistence")]["status"] == "completed"

    failed = statuses[("reservoir", "logistic")]
    assert failed["status"] == "FAILED"
    assert failed["error"]

    # The failed cell's artifact is still written (never silently dropped).
    failed_path = output_dir / "reservoir_logistic_h1.json"
    assert failed_path.exists()
    failed_payload = json.loads(failed_path.read_text())
    assert failed_payload["status"] == "FAILED"
    assert failed_payload["error"]


def test_mismatched_filename_is_recorded_as_failed(tmp_path):
    """A run file whose name doesn't match the spec it contains must be
    FAILED, not silently profiled under the wrong identity."""
    config_path = _dataset_config_yaml(tmp_path, length=_LENGTH)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    output_dir = tmp_path / "profiles"
    hardware_output = tmp_path / "hardware_profile.json"

    persistence_spec = _baseline_spec(
        "persistence", horizon=_HORIZON, season=_SEASON, washout=_WASHOUT, length=_LENGTH
    )
    # Deliberately wrong horizon in the filename vs. the spec inside.
    _write_cell(runs_dir, "baseline", "persistence", 24, persistence_spec)

    summary = run_profiling_pass(
        config_path=config_path,
        runs_dir=runs_dir,
        output_dir=output_dir,
        hardware_output_path=hardware_output,
        n_steps=5,
        warmup=2,
    )

    assert len(summary) == 1
    assert summary[0]["status"] == "FAILED"
    assert "does not match" in summary[0]["error"]
    # family/model/horizon must come from the resolved spec, not the filename.
    assert summary[0]["family"] == "baseline"
    assert summary[0]["model"] == "persistence"
    assert summary[0]["horizon"] == _HORIZON


def test_cli_process_exits_nonzero_when_a_cell_fails(tmp_path):
    config_path = _dataset_config_yaml(tmp_path, length=_LENGTH)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    output_dir = tmp_path / "profiles"
    hardware_output = tmp_path / "hardware_profile.json"

    good_spec = _baseline_spec(
        "persistence", horizon=_HORIZON, season=_SEASON, washout=_WASHOUT, length=_LENGTH
    )
    broken_spec = _reservoir_spec(
        "logistic",
        horizon=_HORIZON,
        season=_SEASON,
        washout=_WASHOUT,
        length=_LENGTH,
        params={"units": "not-a-number"},
    )
    _write_cell(runs_dir, "baseline", "persistence", _HORIZON, good_spec)
    _write_cell(runs_dir, "reservoir", "logistic", _HORIZON, broken_spec)

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_profiling.py"),
            "--config", str(config_path),
            "--runs", str(runs_dir),
            "--output", str(output_dir),
            "--hardware-output", str(hardware_output),
            "--n-steps", "5",
            "--warmup", "2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_cli_process_exits_zero_when_all_cells_succeed(tmp_path):
    config_path = _dataset_config_yaml(tmp_path, length=_LENGTH)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    output_dir = tmp_path / "profiles"
    hardware_output = tmp_path / "hardware_profile.json"

    _two_good_cells(runs_dir)

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_profiling.py"),
            "--config", str(config_path),
            "--runs", str(runs_dir),
            "--output", str(output_dir),
            "--hardware-output", str(hardware_output),
            "--n-steps", "5",
            "--warmup", "2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
