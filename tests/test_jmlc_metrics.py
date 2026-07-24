"""Edge-case tests for the JMLC quality metrics (MET-001).

Per DEC-013 these are the seasonal-naive MASE scale (train-only, lag 24),
MASE itself, and the MAE skill relative to seasonal persistence. An empty
target set or a zero denominator is an explicit protocol error, never a
published NaN/Infinity.
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_bench.core.metrics import (
    mae_skill,
    mase,
    seasonal_naive_mae_scale,
)


class TestSeasonalNaiveMaeScale:
    def test_regular_series_scale_and_term_count(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        scale, n_terms = seasonal_naive_mae_scale(y, season=2)
        # |3-1|,|4-2|,|5-3|,|6-4| = 2 each
        assert scale == pytest.approx(2.0)
        assert n_terms == 4

    def test_mask_excludes_pairs_touching_imputed_hours(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        observed = np.array([True, True, False, True, True, True])
        # valid t: t=3 (|4-2|), t=5 (|6-4|); t=2 and t=4 touch imputed index 2
        scale, n_terms = seasonal_naive_mae_scale(
            y, season=2, observed_mask=observed
        )
        assert scale == pytest.approx(2.0)
        assert n_terms == 2

    def test_default_season_is_24(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(200)
        scale_default, n_default = seasonal_naive_mae_scale(y)
        scale_24, n_24 = seasonal_naive_mae_scale(y, season=24)
        assert scale_default == scale_24
        assert n_default == n_24 == 200 - 24

    def test_season_below_one_is_a_protocol_error(self):
        with pytest.raises(ValueError, match="season"):
            seasonal_naive_mae_scale(np.arange(10.0), season=0)

    def test_series_shorter_than_season_has_no_pairs(self):
        with pytest.raises(ValueError, match="no seasonal pairs|insufficient"):
            seasonal_naive_mae_scale(np.array([1.0, 2.0]), season=2)

    def test_mask_leaving_no_valid_pairs_is_a_protocol_error(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        observed = np.array([False, False, True, True])
        with pytest.raises(ValueError, match="no seasonal pairs|insufficient"):
            seasonal_naive_mae_scale(y, season=2, observed_mask=observed)

    def test_constant_series_zero_scale_is_a_protocol_error(self):
        with pytest.raises(ValueError, match="zero"):
            seasonal_naive_mae_scale(np.full(10, 5.0), season=1)


class TestMase:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mase(y, y, seasonal_scale=2.0) == pytest.approx(0.0)

    def test_scaled_absolute_error(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        # MAE = 1.0, scale = 2.0 -> 0.5
        assert mase(y_true, y_pred, seasonal_scale=2.0) == pytest.approx(0.5)

    def test_zero_scale_is_a_protocol_error(self):
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="scale"):
            mase(y, y, seasonal_scale=0.0)

    def test_empty_targets_is_a_protocol_error(self):
        with pytest.raises(ValueError, match="empty"):
            mase(np.array([]), np.array([]), seasonal_scale=1.0)


class TestMaeSkill:
    def test_model_better_than_reference_is_positive(self):
        assert mae_skill(mae_model=0.5, mae_reference=1.0) == pytest.approx(0.5)

    def test_equal_to_reference_is_zero(self):
        assert mae_skill(mae_model=1.0, mae_reference=1.0) == pytest.approx(0.0)

    def test_worse_than_reference_is_negative(self):
        assert mae_skill(mae_model=2.0, mae_reference=1.0) == pytest.approx(-1.0)

    def test_zero_reference_is_a_protocol_error(self):
        with pytest.raises(ValueError, match="reference"):
            mae_skill(mae_model=0.5, mae_reference=0.0)
