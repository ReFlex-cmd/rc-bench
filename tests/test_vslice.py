"""VSLICE-001: end-to-end horizon-1 persistence + ESN through run_pipeline.

Proves the vertical slice data dict -> run_pipeline -> ResultSpec works for
both a deterministic baseline and a reservoir model, and that both are scored
on the same observed test target set (checked via the saved prediction
artifacts).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rc_bench.core.schema import (
    BaselineSpec,
    DatasetSpec,
    ExperimentSpec,
    ProtocolSpec,
    ReadoutSpec,
    ReservoirSpec,
)
from rc_bench.runners.pipeline import run_pipeline

WASHOUT = 24
HORIZON = 1
SEASON = 24
ALPHA_GRID = [0.01, 0.1, 1.0]


def _data(seed: int = 0, n: int = 120) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def series(offset: float) -> np.ndarray:
        t = np.arange(n)
        return np.sin(2 * np.pi * t / SEASON) + 0.01 * t + offset + 0.05 * rng.standard_normal(n)

    y_train, y_val, y_test = (series(off) for off in (0.0, 5.0, 10.0))
    mask = np.ones(n, dtype=np.bool_)
    mask[[3, 50, 90]] = False
    return {
        "X_train": y_train[:, None], "y_train": y_train,
        "X_val": y_val[:, None], "y_val": y_val,
        "X_test": y_test[:, None], "y_test": y_test,
        "target_observed_mask_train": mask.copy(),
        "target_observed_mask_val": mask.copy(),
        "target_observed_mask_test": mask.copy(),
    }


def _protocol() -> ProtocolSpec:
    return ProtocolSpec(
        washout=WASHOUT,
        train_frac=0.6,
        val_frac=0.2,
        forecasting_mode="fixed_horizon",
        horizon=HORIZON,
    )


def _persistence_spec() -> ExperimentSpec:
    protocol = _protocol().model_copy(
        update={"n_seeds": 0, "selection_metric": "nrmse_std", "seasonal_period": SEASON}
    )
    return ExperimentSpec(
        dataset=DatasetSpec(name="uci_household_power", length=12_000),
        baseline=BaselineSpec(type="persistence"),
        protocol=protocol,
        readout=ReadoutSpec(alpha_grid=ALPHA_GRID),
        seed=None,
    )


def _esn_spec(n_seeds: int = 1, *, seasonal: bool = False) -> ExperimentSpec:
    update: dict[str, object] = {"n_seeds": n_seeds}
    if seasonal:
        update.update(selection_metric="nrmse_std", seasonal_period=SEASON)
    protocol = _protocol().model_copy(update=update)
    return ExperimentSpec(
        dataset=DatasetSpec(name="uci_household_power", length=12_000),
        reservoir=ReservoirSpec(type="esn", params={"n_units": 30, "scaler": "none"}),
        protocol=protocol,
        readout=ReadoutSpec(alpha_grid=ALPHA_GRID),
        seed=42,
    )


def test_vslice_persistence_and_esn_share_target_set(tmp_path: Path) -> None:
    data = _data()

    persistence = run_pipeline(
        data, _persistence_spec(), artifact_dir=tmp_path, save_predictions=True
    )
    assert persistence.status == "completed"
    assert persistence.model_family == "baseline"
    assert persistence.deterministic is True
    assert persistence.evaluated_seeds == []
    assert persistence.selection.method == "none"
    assert persistence.metrics is not None
    assert np.isfinite(persistence.metrics.mase)
    assert np.isfinite(persistence.metrics.mae_skill)
    assert persistence.evaluation.target_start_index == WASHOUT + HORIZON

    esn = run_pipeline(
        data, _esn_spec(), artifact_dir=tmp_path, save_predictions=True
    )
    assert esn.status == "completed"
    assert esn.model_family == "reservoir"
    assert esn.deterministic is False
    assert esn.evaluated_seeds == [42]
    assert esn.selection.method == "none"
    assert esn.metrics is not None
    assert np.isfinite(esn.metrics.nrmse_std)

    persistence_npz = np.load(persistence.artifact_paths["predictions"])
    esn_npz = np.load(esn.artifact_paths["predictions"])
    np.testing.assert_array_equal(persistence_npz["y_test"], esn_npz["y_test"])


def test_both_families_record_the_same_evaluation_context() -> None:
    """Evidence-level proof that the fair table compares like with like.

    A reader of the bundle must be able to check the shared target set and the
    shared MASE scale from the RunRecords alone, without reloading predictions,
    so both families have to record the EvaluationContext — not just baselines.
    """
    data = _data()

    persistence = run_pipeline(data, _persistence_spec())
    esn = run_pipeline(data, _esn_spec(seasonal=True))

    assert esn.evaluation is not None
    assert esn.evaluation == persistence.evaluation
    assert esn.evaluation.n_test_targets == persistence.evaluation.n_test_targets
    assert esn.evaluation.mase_scale == persistence.evaluation.mase_scale


def test_multi_seed_reservoir_records_the_evaluation_context() -> None:
    """The fair matrix runs reservoirs with 5 seeds — that path needs it too."""
    result = run_pipeline(_data(), _esn_spec(n_seeds=2, seasonal=True))

    assert result.multi_seed_result is not None
    assert result.evaluation is not None
    assert result.evaluation.target_start_index == WASHOUT + HORIZON
    assert result.evaluation.seasonal_period == SEASON
