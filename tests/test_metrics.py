"""
Deterministic unit tests for rc_bench.core.metrics.
All expected values are computed analytically from hand-crafted arrays.
"""
import math
import numpy as np
import pytest

from rc_bench.core.metrics import (
    mse,
    rmse,
    mae,
    nrmse_range,
    nrmse_std,
    nrmse_var,
    prediction_horizon,
)

# y_true = [0, 1, 2, 3, 4], y_pred = [0, 1, 2, 3, 5]  — error only at last step
Y_TRUE = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
Y_PRED = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
# MSE = (0+0+0+0+1)/5 = 0.2
# RMSE = sqrt(0.2)
# MAE = 0.2
# range = 4 → nrmse_range = sqrt(0.2)/4
# std = sqrt(2) → nrmse_std = sqrt(0.2)/sqrt(2) = sqrt(0.1)
# var = 2.0 → nrmse_var = 0.2/2.0 = 0.1

_MSE = 0.2
_RMSE = math.sqrt(0.2)
_MAE = 0.2
_NRMSE_RANGE = math.sqrt(0.2) / 4.0
_NRMSE_STD = math.sqrt(0.2) / math.sqrt(2.0)
_NRMSE_VAR = 0.2 / 2.0


class TestMSE:
    def test_exact(self):
        assert math.isclose(mse(Y_TRUE, Y_PRED), _MSE)

    def test_perfect_prediction_is_zero(self):
        assert mse(Y_TRUE, Y_TRUE) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mse(y_true, y_pred) == pytest.approx(1.0)

    def test_symmetric(self):
        assert mse(Y_TRUE, Y_PRED) == mse(Y_PRED, Y_TRUE)

    def test_returns_float(self):
        assert isinstance(mse(np.zeros(5), np.ones(5)), float)


class TestRMSE:
    def test_exact(self):
        assert math.isclose(rmse(Y_TRUE, Y_PRED), _RMSE)

    def test_perfect_prediction_is_zero(self):
        assert rmse(Y_TRUE, Y_TRUE) == 0.0

    def test_equals_sqrt_mse(self):
        assert math.isclose(rmse(Y_TRUE, Y_PRED), math.sqrt(mse(Y_TRUE, Y_PRED)))

    def test_returns_float(self):
        assert isinstance(rmse(np.zeros(5), np.ones(5)), float)


class TestMAE:
    def test_exact(self):
        assert math.isclose(mae(Y_TRUE, Y_PRED), _MAE)

    def test_perfect_prediction_is_zero(self):
        assert mae(Y_TRUE, Y_TRUE) == 0.0

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, -4.0])
        assert mae(y_true, y_pred) == pytest.approx(3.5)

    def test_mae_le_rmse(self):
        assert mae(Y_TRUE, Y_PRED) <= rmse(Y_TRUE, Y_PRED) + 1e-12

    def test_returns_float(self):
        assert isinstance(mae(np.zeros(5), np.ones(5)), float)


class TestNrmseRange:
    def test_exact(self):
        assert math.isclose(nrmse_range(Y_TRUE, Y_PRED), _NRMSE_RANGE)

    def test_perfect_prediction_is_zero(self):
        assert nrmse_range(Y_TRUE, Y_TRUE) == 0.0

    def test_unit_range(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 0.0])
        # rmse = sqrt(0.5), range = 1.0
        assert nrmse_range(y_true, y_pred) == pytest.approx(math.sqrt(0.5))

    def test_constant_signal_returns_nan(self):
        c = np.ones(5)
        assert math.isnan(nrmse_range(c, c + 1.0))

    def test_returns_float(self):
        assert isinstance(nrmse_range(Y_TRUE, Y_PRED), float)


class TestNrmseStd:
    def test_exact(self):
        assert math.isclose(nrmse_std(Y_TRUE, Y_PRED), _NRMSE_STD)

    def test_perfect_prediction_is_zero(self):
        assert nrmse_std(Y_TRUE, Y_TRUE) == 0.0

    def test_constant_signal_returns_nan(self):
        c = np.ones(5)
        assert math.isnan(nrmse_std(c, c + 1.0))

    def test_returns_float(self):
        assert isinstance(nrmse_std(Y_TRUE, Y_PRED), float)


class TestNrmseVar:
    def test_exact(self):
        assert math.isclose(nrmse_var(Y_TRUE, Y_PRED), _NRMSE_VAR)

    def test_perfect_prediction_is_zero(self):
        assert nrmse_var(Y_TRUE, Y_TRUE) == 0.0

    def test_mean_predictor_is_one(self):
        # MSE of mean predictor == var(y_true) → nrmse_var = 1.0
        mean_pred = np.full_like(Y_TRUE, Y_TRUE.mean())
        assert math.isclose(nrmse_var(Y_TRUE, mean_pred), 1.0)

    def test_constant_signal_returns_nan(self):
        c = np.ones(5)
        assert math.isnan(nrmse_var(c, c + 1.0))

    def test_equals_nrmse_std_squared(self):
        # nrmse_var = MSE/var = (RMSE/std)^2 = nrmse_std^2
        assert math.isclose(nrmse_var(Y_TRUE, Y_PRED), nrmse_std(Y_TRUE, Y_PRED) ** 2)

    def test_returns_float(self):
        assert isinstance(nrmse_var(Y_TRUE, Y_PRED), float)


class TestPredictionHorizon:
    def test_perfect_prediction_returns_full_length(self):
        assert prediction_horizon(Y_TRUE, Y_TRUE) == len(Y_TRUE)

    def test_instant_failure(self):
        # std(Y_TRUE) ≈ 1.414; error=10/1.414 >> 0.3 at step 0
        bad_pred = Y_TRUE + 10.0
        assert prediction_horizon(Y_TRUE, bad_pred) == 0

    def test_fails_at_known_step(self):
        # error at step 4 only: |1.0|/sqrt(2) ≈ 0.707 > 0.3 → horizon = 4
        assert prediction_horizon(Y_TRUE, Y_PRED) == 4

    def test_constant_signal_returns_zero(self):
        c = np.ones(5)
        assert prediction_horizon(c, c + 1.0) == 0

    def test_custom_threshold_above_max_error(self):
        # max pointwise error = 0.707; threshold=1.0 → never exceeded → full horizon
        assert prediction_horizon(Y_TRUE, Y_PRED, threshold=1.0) == len(Y_TRUE)

    def test_returns_int(self):
        assert isinstance(prediction_horizon(Y_TRUE, Y_PRED), int)
