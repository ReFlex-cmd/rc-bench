from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import rc_bench.runners.experiment_runner as runner_module
from rc_bench.core.schema import (
    DatasetSpec,
    ExperimentSpec,
    ProtocolSpec,
    ReadoutSpec,
    ReservoirSpec,
)
from rc_bench.runners.experiment_runner import _scale_inputs, run_experiment


WASHOUT = 2


class RecordingReservoir:
    DEFAULT_SCALER = "none"

    def __init__(self) -> None:
        self.seen_inputs: list[np.ndarray] = []

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        self.seen_inputs.append(values.copy())
        return np.column_stack([values[:, 0], values[:, 0] ** 2 + 1.0])

    def sanity_check(self, H: np.ndarray) -> dict[str, bool]:
        return {}


def _spec(mode: str, *, horizon: int = 1) -> ExperimentSpec:
    return ExperimentSpec(
        dataset=DatasetSpec(name="uci_household_power", length=12_000),
        reservoir=ReservoirSpec(type="esn", params={"scaler": "none"}),
        protocol=ProtocolSpec(
            washout=WASHOUT,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode=mode,
            horizon=horizon,
        ),
        readout=ReadoutSpec(alpha_grid=[0.1, 1.0]),
        seed=42,
    )


def _masked_data() -> dict[str, np.ndarray]:
    return {
        "X_train": np.arange(8, dtype=np.float64).reshape(-1, 1),
        "y_train": np.arange(100, 108, dtype=np.float64),
        "X_val": np.arange(10, 17, dtype=np.float64).reshape(-1, 1),
        "y_val": np.arange(200, 207, dtype=np.float64),
        "X_test": np.arange(20, 28, dtype=np.float64).reshape(-1, 1),
        "y_test": np.arange(300, 308, dtype=np.float64),
        "target_observed_mask_train": np.array(
            [True, False, True, False, True, False, True, True]
        ),
        "target_observed_mask_val": np.array(
            [False, True, True, False, False, True, True]
        ),
        "target_observed_mask_test": np.array(
            [True, False, False, True, False, False, True, True]
        ),
    }


def _install_readout_spies(monkeypatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"predict_lengths": []}

    def fake_select_alpha(H_train, y_train, H_val, y_val, alphas):
        calls["select"] = (
            np.asarray(H_train).copy(),
            np.asarray(y_train).copy(),
            np.asarray(H_val).copy(),
            np.asarray(y_val).copy(),
        )
        calls["alphas"] = list(alphas)
        return {"alpha": 0.1, "val_nrmse": 0.25}

    class SpyReadout:
        def __init__(self, alpha: float) -> None:
            calls["readout_alpha"] = alpha

        def fit(self, H: np.ndarray, y: np.ndarray) -> None:
            calls["fit"] = (np.asarray(H).copy(), np.asarray(y).copy())

        def predict(self, H: np.ndarray) -> np.ndarray:
            states = np.asarray(H)
            calls["predict_lengths"].append(len(states))
            return states[:, 0]

    def fake_closed_loop_predict(readout, reservoir, X_seed, n_steps):
        calls["closed_loop_n_steps"] = n_steps
        calls["closed_loop_seed"] = np.asarray(X_seed).copy()
        return np.arange(n_steps, dtype=np.float64) + 0.5

    monkeypatch.setattr(runner_module, "select_alpha", fake_select_alpha)
    monkeypatch.setattr(runner_module, "RidgeReadout", SpyReadout)
    monkeypatch.setattr(
        runner_module,
        "closed_loop_predict",
        fake_closed_loop_predict,
    )
    return calls


@pytest.mark.parametrize(
    ("mode", "horizon", "target_offset"),
    [
        ("one_step", 1, WASHOUT),
        ("fixed_horizon", 2, WASHOUT + 2),
        ("closed_loop", 1, WASHOUT),
    ],
)
def test_runner_masks_only_aligned_targets(
    monkeypatch,
    mode: str,
    horizon: int,
    target_offset: int,
) -> None:
    data = _masked_data()
    calls = _install_readout_spies(monkeypatch)
    reservoir = RecordingReservoir()

    result = run_experiment(data, _spec(mode, horizon=horizon), reservoir)

    train_mask = data["target_observed_mask_train"][target_offset:]
    val_mask = data["target_observed_mask_val"][target_offset:]
    test_mask = data["target_observed_mask_test"][target_offset:]
    expected_train_y = data["y_train"][target_offset:][train_mask]
    expected_val_y = data["y_val"][target_offset:][val_mask]
    expected_test_y = data["y_test"][target_offset:][test_mask]

    _, selected_train_y, _, selected_val_y = calls["select"]
    np.testing.assert_array_equal(selected_train_y, expected_train_y)
    np.testing.assert_array_equal(selected_val_y, expected_val_y)
    np.testing.assert_array_equal(
        calls["fit"][1],
        np.concatenate([expected_train_y, expected_val_y]),
    )
    np.testing.assert_array_equal(result["y_test"], expected_test_y)
    assert len(result["preds"]) == len(expected_test_y)

    # Masking targets must not remove causally imputed inputs before state evolution.
    assert len(reservoir.seen_inputs[0]) == len(data["X_train"])
    assert len(reservoir.seen_inputs[1]) == len(data["X_val"])
    if mode == "closed_loop":
        assert len(reservoir.seen_inputs) == 2
        assert calls["closed_loop_n_steps"] == len(data["y_test"]) - WASHOUT
    else:
        assert len(reservoir.seen_inputs[2]) == len(data["X_test"])
        assert calls["predict_lengths"][-1] == len(data["y_test"]) - target_offset


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_runner_rejects_zero_observed_aligned_targets(
    monkeypatch,
    split: str,
) -> None:
    data = _masked_data()
    data[f"target_observed_mask_{split}"][:] = False
    _install_readout_spies(monkeypatch)

    with pytest.raises(
        ValueError,
        match=rf"{split}.*zero observed targets after washout/forecast alignment",
    ):
        run_experiment(data, _spec("one_step"), RecordingReservoir())


@pytest.mark.parametrize(
    ("bad_mask", "message"),
    [
        (np.ones(8, dtype=np.int64), "boolean dtype"),
        (np.ones(7, dtype=np.bool_), "shape"),
        (None, "boolean dtype"),
    ],
)
def test_runner_validates_target_masks(
    bad_mask: Any,
    message: str,
) -> None:
    data = _masked_data()
    data["target_observed_mask_train"] = bad_mask

    with pytest.raises(ValueError, match=message):
        run_experiment(data, _spec("one_step"), RecordingReservoir())


@pytest.mark.parametrize("missing_split", ["train", "val", "test"])
def test_runner_rejects_partial_target_mask_group(
    missing_split: str,
) -> None:
    data = _masked_data()
    del data[f"target_observed_mask_{missing_split}"]

    with pytest.raises(
        ValueError,
        match="target observed masks must provide train, val, and test together",
    ):
        run_experiment(data, _spec("one_step"), RecordingReservoir())


def test_zscore_scaler_fit_is_independent_of_val_and_test_extremes() -> None:
    X_train = np.array([[0.0], [2.0]])
    first = _scale_inputs(
        X_train,
        np.array([[1_000.0]]),
        np.array([[-1_000.0]]),
        "zscore",
    )
    second = _scale_inputs(
        X_train,
        np.array([[1_000_000.0]]),
        np.array([[-1_000_000.0]]),
        "zscore",
    )

    np.testing.assert_allclose(first[0], [[-1.0], [1.0]])
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1][0, 0] == pytest.approx(999.0)
    assert first[2][0, 0] == pytest.approx(-1_001.0)
