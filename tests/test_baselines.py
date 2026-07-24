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
)


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
