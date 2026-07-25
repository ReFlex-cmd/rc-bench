"""End-to-end tests for the deterministic baseline runner (BASE-001/002)."""

from __future__ import annotations

import numpy as np
import pytest

from rc_bench.core.schema import (
    BaselineSpec,
    DatasetSpec,
    ExperimentSpec,
    ProtocolSpec,
    ReadoutSpec,
    ReservoirSpec,
)
from rc_bench.protocol.forecasting import aligned_target_positions
from rc_bench.runners.baseline_runner import run_baseline
from rc_bench.runners.experiment_runner import run_experiment

WASHOUT = 24
HORIZON = 1
SEASON = 24
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def _series(rng: np.random.Generator, n: int, offset: float) -> np.ndarray:
    t = np.arange(n)
    return np.sin(2 * np.pi * t / SEASON) + 0.01 * t + offset + 0.05 * rng.standard_normal(n)


def _mask(n: int) -> np.ndarray:
    m = np.ones(n, dtype=np.bool_)
    m[[3, 50, 90]] = False  # imputed hours: one pre-washout, two inside targets
    return m


def _data(seed: int = 0, n: int = 120) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_train, y_val, y_test = (_series(rng, n, off) for off in (0.0, 5.0, 10.0))
    return {
        "X_train": y_train[:, None], "y_train": y_train,
        "X_val": y_val[:, None], "y_val": y_val,
        "X_test": y_test[:, None], "y_test": y_test,
        "target_observed_mask_train": _mask(n),
        "target_observed_mask_val": _mask(n),
        "target_observed_mask_test": _mask(n),
    }


def _baseline_spec(btype: str) -> ExperimentSpec:
    return ExperimentSpec(
        dataset=DatasetSpec(name="uci_household_power", length=12_000),
        baseline=BaselineSpec(type=btype),
        protocol=ProtocolSpec(
            washout=WASHOUT,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode="fixed_horizon",
            horizon=HORIZON,
            n_seeds=0,
            selection_metric="nrmse_std",
            seasonal_period=SEASON,
        ),
        readout=ReadoutSpec(alpha_grid=ALPHA_GRID),
        seed=None,
    )


def _expected_positions(data: dict[str, np.ndarray]) -> np.ndarray:
    return aligned_target_positions(
        data["target_observed_mask_test"], WASHOUT, HORIZON, "fixed_horizon"
    )


def test_persistence_result_shape_and_metadata():
    data = _data()
    result = run_baseline(data, _baseline_spec("persistence"))

    positions = _expected_positions(data)
    np.testing.assert_array_equal(
        result["preds"], data["y_test"][positions - HORIZON]
    )
    np.testing.assert_array_equal(result["y_test"], data["y_test"][positions])

    assert result["model_family"] == "baseline"
    assert result["deterministic"] is True
    assert result["evaluated_seeds"] == []
    assert result["best_alpha"] is None
    assert result["selection"].method == "none"

    evaluation = result["evaluation"]
    assert evaluation.target_start_index == WASHOUT + HORIZON
    assert evaluation.n_test_targets == positions.size
    assert evaluation.seasonal_period == SEASON
    assert evaluation.mase_scale > 0.0
    assert evaluation.n_mase_scale_terms > 0
    assert np.isfinite(result["metrics"].mase)
    assert np.isfinite(result["metrics"].mae_skill)


def test_seasonal_persistence_has_zero_self_skill():
    data = _data()
    result = run_baseline(data, _baseline_spec("seasonal_persistence"))

    positions = _expected_positions(data)
    np.testing.assert_array_equal(
        result["preds"], data["y_test"][positions - SEASON]
    )
    # Seasonal persistence IS the MAE-skill reference, so its skill is exactly 0.
    assert result["metrics"].mae_skill == pytest.approx(0.0)
    assert result["selection"].method == "none"


def test_ridge_ar_selection_metadata_and_determinism():
    data = _data()
    result = run_baseline(data, _baseline_spec("ridge_ar"))

    selection = result["selection"]
    assert selection.method == "fixed_grid"
    assert selection.metric == "nrmse_std"
    assert len(selection.candidates) == len(ALPHA_GRID)
    best = min(selection.candidates, key=lambda c: c.score)
    assert selection.selected_params["alpha"] == best.params["alpha"]
    assert result["best_alpha"] == best.params["alpha"]

    again = run_baseline(data, _baseline_spec("ridge_ar"))
    np.testing.assert_array_equal(result["preds"], again["preds"])


class _StubReservoir:
    """Minimal reservoir returning a 2-column state; used only to drive the
    reservoir runner's target alignment for the DEC-013 equivalence check."""

    DEFAULT_SCALER = "none"

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        return np.column_stack([values[:, 0], values[:, 0] ** 2 + 1.0])

    def sanity_check(self, H: np.ndarray) -> dict[str, bool]:
        return {}


def test_baseline_and_reservoir_share_observed_test_targets():
    data = _data()
    baseline = run_baseline(data, _baseline_spec("persistence"))

    reservoir_spec = ExperimentSpec(
        dataset=DatasetSpec(name="uci_household_power", length=12_000),
        reservoir=ReservoirSpec(type="esn", params={"scaler": "none"}),
        protocol=ProtocolSpec(
            washout=WASHOUT,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode="fixed_horizon",
            horizon=HORIZON,
        ),
        readout=ReadoutSpec(alpha_grid=ALPHA_GRID),
        seed=42,
    )
    reservoir = run_experiment(data, reservoir_spec, _StubReservoir())

    np.testing.assert_array_equal(baseline["y_test"], reservoir["y_test"])


def test_reservoir_reports_jmlc_metrics_on_the_same_seasonal_basis():
    data = _data()
    reservoir_spec = ExperimentSpec(
        dataset=DatasetSpec(name="uci_household_power", length=12_000),
        reservoir=ReservoirSpec(type="esn", params={"scaler": "none"}),
        protocol=ProtocolSpec(
            washout=WASHOUT,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode="fixed_horizon",
            horizon=HORIZON,
            seasonal_period=SEASON,
        ),
        readout=ReadoutSpec(alpha_grid=ALPHA_GRID),
        seed=42,
    )
    metrics = run_experiment(data, reservoir_spec, _StubReservoir())["metrics"]
    assert metrics.mase is not None and np.isfinite(metrics.mase)
    assert metrics.mae_skill is not None and np.isfinite(metrics.mae_skill)

    # The reservoir's MAE-skill reference must be the seasonal-persistence
    # baseline evaluated on the same test target set (DEC-013).
    seasonal = run_baseline(data, _baseline_spec("seasonal_persistence"))
    implied_reference_mae = metrics.mae / (1.0 - metrics.mae_skill)
    np.testing.assert_allclose(
        implied_reference_mae, seasonal["metrics"].mae, rtol=1e-9
    )
