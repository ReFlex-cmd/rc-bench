"""Pure deterministic baseline predictors (BASE-001 / BASE-002 core).

These operate on a single split's unshifted ``values`` array and the
split-local target positions (indices into ``values``) that the runner has
already aligned to washout + horizon and masked to observed targets. Indexing
uses only causal history within the split; a lag that reaches before index 0
is an explicit protocol error, never a silently wrapped negative index.
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_bench.core.baselines import (
    ar_lag_features,
    persistence_forecast,
    seasonal_persistence_forecast,
    select_and_fit_ridge_ar,
)
from rc_bench.core.metrics import nrmse_std


class TestPersistence:
    def test_predicts_value_at_origin_time(self):
        values = np.arange(10.0)
        preds = persistence_forecast(values, np.array([5, 7, 9]), horizon=1)
        np.testing.assert_array_equal(preds, [4.0, 6.0, 8.0])

    def test_horizon_shifts_the_origin(self):
        values = np.arange(10.0)
        preds = persistence_forecast(values, np.array([5, 7, 9]), horizon=2)
        np.testing.assert_array_equal(preds, [3.0, 5.0, 7.0])

    def test_negative_origin_is_a_protocol_error(self):
        with pytest.raises(ValueError, match="history|before index 0"):
            persistence_forecast(np.arange(3.0), np.array([0]), horizon=1)


class TestSeasonalPersistence:
    def test_predicts_value_one_season_earlier(self):
        values = np.arange(30.0)
        preds = seasonal_persistence_forecast(
            values, np.array([25, 26]), season=24
        )
        np.testing.assert_array_equal(preds, [1.0, 2.0])

    def test_default_season_is_24(self):
        values = np.arange(30.0)
        default = seasonal_persistence_forecast(values, np.array([25]))
        explicit = seasonal_persistence_forecast(values, np.array([25]), season=24)
        np.testing.assert_array_equal(default, explicit)

    def test_lag_before_start_is_a_protocol_error(self):
        with pytest.raises(ValueError, match="history|before index 0"):
            seasonal_persistence_forecast(
                np.arange(10.0), np.array([5]), season=24
            )


class TestArLagFeatures:
    def test_rows_are_most_recent_lags_first(self):
        values = np.arange(10.0)
        feats = ar_lag_features(values, np.array([5]), horizon=1, n_lags=3)
        # origin = 5 - 1 = 4 -> [values[4], values[3], values[2]]
        np.testing.assert_array_equal(feats, [[4.0, 3.0, 2.0]])

    def test_shape_is_positions_by_n_lags(self):
        values = np.arange(50.0)
        feats = ar_lag_features(
            values, np.array([30, 31, 32]), horizon=1, n_lags=24
        )
        assert feats.shape == (3, 24)

    def test_horizon_moves_the_origin_back(self):
        values = np.arange(10.0)
        feats = ar_lag_features(values, np.array([6]), horizon=2, n_lags=2)
        # origin = 6 - 2 = 4 -> [values[4], values[3]]
        np.testing.assert_array_equal(feats, [[4.0, 3.0]])

    def test_lag_reaching_before_start_is_a_protocol_error(self):
        with pytest.raises(ValueError, match="history|before index 0"):
            ar_lag_features(np.arange(10.0), np.array([2]), horizon=1, n_lags=3)


class TestRidgeAR:
    ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]

    @staticmethod
    def _linear_split(rng, n, coefs, intercept=0.5, noise=0.05):
        X = rng.standard_normal((n, coefs.size))
        y = X @ coefs + intercept + noise * rng.standard_normal(n)
        return X, y

    def _dataset(self, seed=0, x_test=None):
        rng = np.random.default_rng(seed)
        coefs = np.array([2.0, -1.0, 0.5])
        X_train, y_train = self._linear_split(rng, 200, coefs)
        X_val, y_val = self._linear_split(rng, 80, coefs)
        if x_test is None:
            X_test, _ = self._linear_split(rng, 40, coefs)
        else:
            X_test = x_test
        return X_train, y_train, X_val, y_val, X_test

    def test_output_shapes(self):
        X_train, y_train, X_val, y_val, X_test = self._dataset()
        fit = select_and_fit_ridge_ar(
            X_train, y_train, X_val, y_val, X_test, self.ALPHA_GRID
        )
        assert fit.test_predictions.shape == (len(X_test),)
        assert fit.val_predictions.shape == (len(X_val),)
        assert len(fit.candidates) == len(self.ALPHA_GRID)

    def test_selected_alpha_minimises_validation_nrmse_std(self):
        X_train, y_train, X_val, y_val, X_test = self._dataset()
        fit = select_and_fit_ridge_ar(
            X_train, y_train, X_val, y_val, X_test, self.ALPHA_GRID
        )
        best = min(fit.candidates, key=lambda c: c[1])
        assert fit.alpha == best[0]
        assert fit.val_nrmse_std == pytest.approx(best[1])
        # val_predictions belong to the selected (train-fit) model
        assert nrmse_std(y_val, fit.val_predictions) == pytest.approx(
            fit.val_nrmse_std
        )

    def test_deterministic(self):
        args = self._dataset()
        first = select_and_fit_ridge_ar(*args, self.ALPHA_GRID)
        second = select_and_fit_ridge_ar(*args, self.ALPHA_GRID)
        np.testing.assert_array_equal(first.test_predictions, second.test_predictions)
        assert first.alpha == second.alpha

    def test_scaler_and_selection_are_independent_of_test_extremes(self):
        base = self._dataset()
        extreme_test = np.full_like(base[4], 1_000_000.0)
        with_extreme = select_and_fit_ridge_ar(
            base[0], base[1], base[2], base[3], extreme_test, self.ALPHA_GRID
        )
        normal = select_and_fit_ridge_ar(*base, self.ALPHA_GRID)
        # train-only scaler + val selection must not see the test extremes
        assert with_extreme.alpha == normal.alpha
        np.testing.assert_array_equal(
            with_extreme.val_predictions, normal.val_predictions
        )

    def test_empty_alpha_grid_is_a_protocol_error(self):
        X_train, y_train, X_val, y_val, X_test = self._dataset()
        with pytest.raises(ValueError, match="alpha"):
            select_and_fit_ridge_ar(X_train, y_train, X_val, y_val, X_test, [])
