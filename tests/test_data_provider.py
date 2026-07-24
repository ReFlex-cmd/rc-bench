from pathlib import Path

import numpy as np
import pytest

import rc_bench.core.data_provider as data_provider_module
from rc_bench.core.data_provider import (
    DATASET_CATALOG,
    JMLC_RAW_DATA_ENV,
    UCI_HOUSEHOLD_POWER_DATASET,
    generate_lorenz63,
    generate_mackey_glass,
    generate_narma10,
    generate_narma30,
    get_data_for_experiment,
)
from rc_bench.data.jmlc import HourlyPowerSeries


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _check_generator(gen, T, seed=42):
    X, y = gen(T, seed=seed)
    assert X.shape == (T, 1), f"X.shape {X.shape} != ({T}, 1)"
    assert y.shape == (T,),   f"y.shape {y.shape} != ({T},)"
    assert np.all(np.isfinite(X)), "X contains non-finite values"
    assert np.all(np.isfinite(y)), "y contains non-finite values"
    return X, y


def _uci_hourly_series() -> HourlyPowerSeries:
    length = 12_000
    start = np.datetime64("2006-12-16T17", "h")
    timestamps = np.arange(
        start,
        start + np.timedelta64(length, "h"),
        dtype="datetime64[h]",
    )
    values = np.arange(length, dtype=np.float64)
    observed = np.ones(length, dtype=np.bool_)
    for index in (5, 7_205, 9_605):
        observed[index] = False
        values[index] = values[index - 1]
    return HourlyPowerSeries(
        timestamps=timestamps,
        values=values,
        observed_mask=observed,
        imputed_mask=~observed,
        valid_minute_counts=np.where(observed, 60, 0),
    )


# ---------------------------------------------------------------------------
# TestGenerateNarma10
# ---------------------------------------------------------------------------

class TestGenerateNarma10:
    def test_output_shapes(self):
        _check_generator(generate_narma10, T=500)

    def test_deterministic_with_seed(self):
        X1, y1 = generate_narma10(300, seed=123)
        X2, y2 = generate_narma10(300, seed=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_narma10(300, seed=1)
        X2, _ = generate_narma10(300, seed=2)
        assert not np.array_equal(X1, X2)

    def test_input_range(self):
        X, _ = generate_narma10(1000)
        assert np.all(X >= 0.0)
        assert np.all(X <= 0.5)

    def test_output_non_negative(self):
        _, y = generate_narma10(2000)
        assert np.all(y >= 0.0)


# ---------------------------------------------------------------------------
# TestGenerateNarma30
# ---------------------------------------------------------------------------

class TestGenerateNarma30:
    def test_output_shapes(self):
        _check_generator(generate_narma30, T=500)

    def test_no_divergence(self):
        _, y = generate_narma30(3000, seed=42)
        assert np.all(np.isfinite(y)), "NARMA30 diverged"
        assert y.max() < 10.0

    def test_deterministic_with_seed(self):
        X1, y1 = generate_narma30(300, seed=7)
        X2, y2 = generate_narma30(300, seed=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_narma30(300, seed=1)
        X2, _ = generate_narma30(300, seed=2)
        assert not np.array_equal(X1, X2)

    def test_input_range(self):
        X, _ = generate_narma30(1000)
        assert np.all(X >= 0.0)
        assert np.all(X <= 0.5)

    def test_output_non_negative(self):
        _, y = generate_narma30(3000)
        assert np.all(y >= 0.0)

    def test_first_30_outputs_are_zero(self):
        _, y = generate_narma30(500)
        np.testing.assert_array_equal(y[:30], 0.0)

    def test_longer_memory_than_narma10(self):
        _, y10 = generate_narma10(500, seed=42)
        _, y30 = generate_narma30(500, seed=42)
        assert np.all(y10[:10] == 0.0)
        assert np.all(y30[:30] == 0.0)
        assert y10[10] > 0.0
        assert y30[30] > 0.0


# ---------------------------------------------------------------------------
# TestGenerateMackeyGlass
# ---------------------------------------------------------------------------

class TestGenerateMackeyGlass:
    def test_output_shapes(self):
        _check_generator(generate_mackey_glass, T=500)

    def test_deterministic_with_seed(self):
        X1, y1 = generate_mackey_glass(300, seed=5)
        X2, y2 = generate_mackey_glass(300, seed=5)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_mackey_glass(300, seed=1)
        X2, _ = generate_mackey_glass(300, seed=2)
        assert not np.array_equal(X1, X2)

    def test_positive_output(self):
        X, y = generate_mackey_glass(5000)
        assert X.min() > 0.0, "Mackey-Glass solution must be positive"
        assert X.max() < 2.5

    def test_one_step_ahead_shift(self):
        X, y = generate_mackey_glass(200)
        np.testing.assert_array_equal(y[:-1], X[1:, 0])

    def test_bounded_trajectory(self):
        _, y = generate_mackey_glass(5000)
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# TestGenerateLorenz63
# ---------------------------------------------------------------------------

class TestGenerateLorenz63:
    def test_output_shapes(self):
        _check_generator(generate_lorenz63, T=500)

    def test_deterministic_with_seed(self):
        X1, y1 = generate_lorenz63(300, seed=3)
        X2, y2 = generate_lorenz63(300, seed=3)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_lorenz63(300, seed=0)
        X2, _ = generate_lorenz63(300, seed=1)
        assert not np.array_equal(X1, X2)

    def test_chaotic_attractor_range(self):
        X, _ = generate_lorenz63(5000)
        assert X.min() > -25.0
        assert X.max() < 25.0
        assert np.all(np.isfinite(X))

    def test_one_step_ahead_shift(self):
        X, y = generate_lorenz63(200)
        np.testing.assert_array_equal(y[:-1], X[1:, 0])

    def test_bounded_trajectory(self):
        _, y = generate_lorenz63(5000)
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# TestGetDataForExperiment
# ---------------------------------------------------------------------------

class TestGetDataForExperiment:
    def test_correct_split_sizes(self):
        length = 1000
        data = get_data_for_experiment("narma10", length=length, train_frac=0.6, val_frac=0.2, seed=42)
        assert data["X_train"].shape[0] == 600
        assert data["X_val"].shape[0] == 200
        assert data["X_test"].shape[0] == 200

    def test_all_keys_present(self):
        data = get_data_for_experiment("narma10", length=500)
        assert set(data.keys()) == {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}

    def test_y_shapes_1d(self):
        data = get_data_for_experiment("narma10", length=500)
        assert data["y_train"].ndim == 1
        assert data["y_val"].ndim == 1
        assert data["y_test"].ndim == 1

    def test_X_shapes_2d(self):
        data = get_data_for_experiment("narma10", length=500)
        assert data["X_train"].ndim == 2
        assert data["X_train"].shape[1] == 1

    def test_data_returned_unscaled(self):
        """data_provider returns raw splits; scaling is the runner's job (audit/03 §3.6.4)."""
        data = get_data_for_experiment("narma10", length=1000)
        # NARMA-10 input ∼ U(0, 0.5): mean ≈ 0.25, std ≈ 0.144 — NOT z-scored.
        assert 0.15 < data["X_train"].mean() < 0.35
        assert data["X_train"].std() > 0.05  # not flattened to ~1.0 either

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_data_for_experiment("nonexistent_dataset")

    def test_deterministic(self):
        d1 = get_data_for_experiment("narma10", length=500, seed=77)
        d2 = get_data_for_experiment("narma10", length=500, seed=77)
        np.testing.assert_array_equal(d1["X_train"], d2["X_train"])
        np.testing.assert_array_equal(d1["y_test"],  d2["y_test"])

    @pytest.mark.parametrize("name", ["narma10", "narma30", "mackey_glass", "lorenz63"])
    def test_all_datasets_return_finite_data(self, name):
        data = get_data_for_experiment(name, length=200, seed=42)
        for key, arr in data.items():
            assert np.all(np.isfinite(arr)), f"{name}/{key} has non-finite values"

    @pytest.mark.parametrize("name", ["narma10", "narma30", "mackey_glass", "lorenz63"])
    def test_all_datasets_have_correct_input_dim(self, name):
        data = get_data_for_experiment(name, length=200)
        assert data["X_train"].shape[1] == DATASET_CATALOG[name]["input_dim"]


class TestUCIHouseholdPowerProvider:
    def test_returns_unshifted_splits_masks_and_timestamps(
        self,
        monkeypatch,
        tmp_path,
    ):
        raw_path = tmp_path / "household_power_consumption.txt"
        captured = {}

        def fake_loader(path):
            captured["path"] = Path(path)
            return _uci_hourly_series()

        monkeypatch.setenv(JMLC_RAW_DATA_ENV, str(raw_path))
        monkeypatch.setattr(
            data_provider_module,
            "load_uci_household_power_series",
            fake_loader,
        )

        data = get_data_for_experiment(
            UCI_HOUSEHOLD_POWER_DATASET,
            length=12_000,
            train_frac=0.6,
            val_frac=0.2,
            seed=999,
        )

        assert captured["path"] == raw_path
        assert set(data) == {
            "X_train",
            "y_train",
            "X_val",
            "y_val",
            "X_test",
            "y_test",
            "target_observed_mask_train",
            "target_observed_mask_val",
            "target_observed_mask_test",
            "timestamps_train",
            "timestamps_val",
            "timestamps_test",
        }
        assert data["X_train"].shape == (7_200, 1)
        assert data["X_val"].shape == (2_400, 1)
        assert data["X_test"].shape == (2_400, 1)
        np.testing.assert_array_equal(data["X_train"][:, 0], data["y_train"])
        np.testing.assert_array_equal(data["X_val"][:, 0], data["y_val"])
        np.testing.assert_array_equal(data["X_test"][:, 0], data["y_test"])
        assert not data["target_observed_mask_train"][5]
        assert not data["target_observed_mask_val"][5]
        assert not data["target_observed_mask_test"][5]
        assert data["timestamps_train"][-1] < data["timestamps_val"][0]
        assert data["timestamps_val"][-1] < data["timestamps_test"][0]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"length": 11_999},
            {"train_frac": 0.5},
            {"val_frac": 0.1},
        ],
    )
    def test_rejects_noncanonical_length_or_split(self, monkeypatch, overrides):
        monkeypatch.setattr(
            data_provider_module,
            "load_uci_household_power_series",
            lambda path: pytest.fail(f"loader must not run for {path}"),
        )
        kwargs = {
            "length": 12_000,
            "train_frac": 0.6,
            "val_frac": 0.2,
            **overrides,
        }

        with pytest.raises(ValueError, match="requires length=12000"):
            get_data_for_experiment(UCI_HOUSEHOLD_POWER_DATASET, **kwargs)


# ---------------------------------------------------------------------------
# TestDatasetCatalog
# ---------------------------------------------------------------------------

class TestDatasetCatalog:
    def test_all_datasets_present(self):
        assert set(DATASET_CATALOG) == {
            "narma10",
            "narma30",
            "mackey_glass",
            "lorenz63",
            UCI_HOUSEHOLD_POWER_DATASET,
        }

    def test_required_keys_in_each_entry(self):
        required = {"description", "default_length", "input_dim", "output_dim"}
        for name, info in DATASET_CATALOG.items():
            assert required <= set(info), f"{name} missing keys"

    def test_input_output_dims_are_one(self):
        for name, info in DATASET_CATALOG.items():
            assert info["input_dim"] == 1
            assert info["output_dim"] == 1
