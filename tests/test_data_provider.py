import numpy as np
import pytest

from rc_bench.core.data_provider import generate_narma10, get_data_for_experiment


class TestGenerateNarma10:
    def test_output_shapes(self):
        T = 500
        X, y = generate_narma10(T)
        assert X.shape == (T, 1)
        assert y.shape == (T,)

    def test_deterministic_with_seed(self):
        X1, y1 = generate_narma10(300, seed=123)
        X2, y2 = generate_narma10(300, seed=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, y1 = generate_narma10(300, seed=1)
        X2, y2 = generate_narma10(300, seed=2)
        assert not np.array_equal(X1, X2)

    def test_input_range(self):
        X, _ = generate_narma10(1000)
        assert np.all(X >= 0.0)
        assert np.all(X <= 0.5)


class TestGetDataForExperiment:
    def test_correct_split_sizes(self):
        length = 1000
        data = get_data_for_experiment("narma10", length=length, train_frac=0.6, val_frac=0.2, seed=42)
        assert data["X_train"].shape[0] == 600
        assert data["X_val"].shape[0] == 200
        assert data["X_test"].shape[0] == 200

    def test_all_keys_present(self):
        data = get_data_for_experiment("narma10", length=500)
        expected_keys = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
        assert set(data.keys()) == expected_keys

    def test_y_shapes_1d(self):
        data = get_data_for_experiment("narma10", length=500)
        assert data["y_train"].ndim == 1
        assert data["y_val"].ndim == 1
        assert data["y_test"].ndim == 1

    def test_X_shapes_2d(self):
        data = get_data_for_experiment("narma10", length=500)
        assert data["X_train"].ndim == 2
        assert data["X_train"].shape[1] == 1

    def test_zscore_scaling_applied(self):
        data = get_data_for_experiment("narma10", length=1000, scaler_name="zscore")
        # После z-score нормализации train должен иметь mean ≈ 0, std ≈ 1
        assert abs(data["X_train"].mean()) < 0.1
        assert abs(data["X_train"].std() - 1.0) < 0.1

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_data_for_experiment("nonexistent_dataset")

    def test_deterministic(self):
        d1 = get_data_for_experiment("narma10", length=500, seed=77)
        d2 = get_data_for_experiment("narma10", length=500, seed=77)
        np.testing.assert_array_equal(d1["X_train"], d2["X_train"])
        np.testing.assert_array_equal(d1["y_test"], d2["y_test"])
