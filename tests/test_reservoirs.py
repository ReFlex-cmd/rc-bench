import numpy as np
import pytest

from rc_bench.core.data_provider import get_data_for_experiment
from rc_bench.core.reservoirs.esn_service import run_esn_experiment
from rc_bench.core.reservoirs.lsm_service import run_lsm_experiment
from rc_bench.core.reservoirs.fhn_service import run_fhn_experiment
from rc_bench.core.reservoirs.logistic_service import run_logistic_experiment

# Общие данные для всех тестов — маленький размер для скорости
_DATA = get_data_for_experiment("narma10", length=600, seed=42)
_EXPECTED_KEYS = {"metrics", "meta", "preds", "y_test"}
_METRIC_KEYS = {"nrmse", "mae", "mse", "execution_time"}


class TestESN:
    _config = {"n_units": 50, "seed": 42, "washout": 50}

    def test_output_keys(self):
        result = run_esn_experiment(_DATA, self._config)
        assert set(result.keys()) == _EXPECTED_KEYS
        assert set(result["metrics"].keys()) == _METRIC_KEYS

    def test_deterministic(self):
        r1 = run_esn_experiment(_DATA, self._config)
        r2 = run_esn_experiment(_DATA, self._config)
        assert r1["metrics"]["nrmse"] == r2["metrics"]["nrmse"]

    def test_nrmse_is_finite_float(self):
        result = run_esn_experiment(_DATA, self._config)
        val = result["metrics"]["nrmse"]
        assert isinstance(val, float)
        assert np.isfinite(val)


class TestLSM:
    _config = {"units": 50, "seed": 42, "washout": 50, "density": 0.1, "dt": 1.0}

    def test_output_keys(self):
        result = run_lsm_experiment(_DATA, self._config)
        assert set(result.keys()) == _EXPECTED_KEYS
        assert set(result["metrics"].keys()) == _METRIC_KEYS

    def test_deterministic(self):
        r1 = run_lsm_experiment(_DATA, self._config)
        r2 = run_lsm_experiment(_DATA, self._config)
        assert r1["metrics"]["nrmse"] == r2["metrics"]["nrmse"]

    def test_nrmse_is_finite_float(self):
        result = run_lsm_experiment(_DATA, self._config)
        val = result["metrics"]["nrmse"]
        assert isinstance(val, float)
        assert np.isfinite(val)


class TestFHN:
    _config = {"units": 30, "seed": 42, "washout": 50, "density": 0.1, "dt": 0.1, "internal_steps": 1}

    def test_output_keys(self):
        result = run_fhn_experiment(_DATA, self._config)
        assert set(result.keys()) == _EXPECTED_KEYS
        assert set(result["metrics"].keys()) == _METRIC_KEYS

    def test_deterministic(self):
        r1 = run_fhn_experiment(_DATA, self._config)
        r2 = run_fhn_experiment(_DATA, self._config)
        assert r1["metrics"]["nrmse"] == r2["metrics"]["nrmse"]

    def test_nrmse_is_finite_float(self):
        result = run_fhn_experiment(_DATA, self._config)
        val = result["metrics"]["nrmse"]
        assert isinstance(val, float)
        assert np.isfinite(val)


class TestLogistic:
    _config = {"units": 50, "seed": 42, "washout": 50}

    def test_output_keys(self):
        result = run_logistic_experiment(_DATA, self._config)
        assert set(result.keys()) == _EXPECTED_KEYS
        assert set(result["metrics"].keys()) == _METRIC_KEYS

    def test_deterministic(self):
        r1 = run_logistic_experiment(_DATA, self._config)
        r2 = run_logistic_experiment(_DATA, self._config)
        assert r1["metrics"]["nrmse"] == r2["metrics"]["nrmse"]

    def test_nrmse_is_finite_float(self):
        result = run_logistic_experiment(_DATA, self._config)
        val = result["metrics"]["nrmse"]
        assert isinstance(val, float)
        assert np.isfinite(val)
