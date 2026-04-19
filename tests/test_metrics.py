import numpy as np
import pytest

from rc_bench.core.metrics import mse, mae, nrmse


class TestMSE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # каждое отклонение = 1, квадрат = 1, среднее = 1
        assert mse(y_true, y_pred) == pytest.approx(1.0)

    def test_returns_float(self):
        result = mse(np.zeros(5), np.ones(5))
        assert isinstance(result, float)


class TestMAE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, -4.0])
        # |3| + |-4| = 7, среднее = 3.5
        assert mae(y_true, y_pred) == pytest.approx(3.5)

    def test_returns_float(self):
        result = mae(np.zeros(5), np.ones(5))
        assert isinstance(result, float)


class TestNRMSE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert nrmse(y, y) == 0.0

    def test_constant_signal_returns_nan(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        # max - min = 0 → nan
        assert np.isnan(nrmse(y_true, y_pred))

    def test_known_value(self):
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([0.0, 10.0])
        assert nrmse(y_true, y_pred) == pytest.approx(0.0)

    def test_nrmse_unit_range(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 0.0])
        # rmse = sqrt(0.5), denom = 1.0 → nrmse = sqrt(0.5)
        assert nrmse(y_true, y_pred) == pytest.approx(np.sqrt(0.5))

    def test_returns_float(self):
        result = nrmse(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert isinstance(result, float)
