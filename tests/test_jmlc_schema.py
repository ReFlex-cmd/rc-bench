from __future__ import annotations

import pytest
from pydantic import ValidationError

from rc_bench.core.schema import (
    BaselineSpec,
    DatasetSpec,
    EvaluationContext,
    ExperimentSpec,
    MetricsResult,
    MetricsSummary,
    ProtocolSpec,
    ReadoutSpec,
    ReservoirSpec,
    ResultSpec,
    SelectionCandidate,
    SelectionResult,
)


def _legacy_reservoir_spec() -> ExperimentSpec:
    return ExperimentSpec(
        dataset=DatasetSpec(name="narma10", length=300),
        reservoir=ReservoirSpec(type="esn", params={"units": 20}),
        protocol=ProtocolSpec(
            washout=20,
            train_frac=0.6,
            val_frac=0.2,
            n_seeds=1,
        ),
        readout=ReadoutSpec(alpha_grid=[0.01, 0.1, 1.0]),
        seed=42,
    )


def _metrics(**updates: float | None) -> MetricsResult:
    values = {
        "rmse": 0.4,
        "nrmse_range": 0.2,
        "nrmse_std": 0.5,
        "nrmse_var": 0.25,
        "mae": 0.3,
        "mse": 0.16,
        "prediction_horizon": 3,
        "val_nrmse_range": 0.21,
        "train_time": 1.0,
        "inference_latency": 0.01,
        "peak_memory": 1024,
        **updates,
    }
    return MetricsResult(**values)


def test_legacy_reservoir_hash_is_unchanged_by_additive_schema_fields() -> None:
    spec = _legacy_reservoir_spec()

    assert spec.model_family == "reservoir"
    assert spec.model_type == "esn"
    assert spec.config_hash() == "610160f0e11c6eb0"


@pytest.mark.parametrize(
    "baseline_type",
    ["persistence", "seasonal_persistence", "ridge_ar"],
)
def test_baseline_spec_is_explicit_deterministic_and_seedless(
    baseline_type: str,
) -> None:
    spec = ExperimentSpec(
        dataset=DatasetSpec(
            name="uci_household_power",
            length=12_000,
            raw_sha256="a" * 64,
        ),
        baseline=BaselineSpec(type=baseline_type, params={}),
        protocol=ProtocolSpec(
            washout=200,
            train_frac=0.6,
            val_frac=0.2,
            forecasting_mode="fixed_horizon",
            horizon=1,
            n_seeds=0,
            selection_metric="nrmse_std",
            seasonal_period=24,
        ),
        seed=None,
    )

    assert spec.model_family == "baseline"
    assert spec.model_type == baseline_type
    assert spec.reservoir is None
    assert spec.seed is None


def test_experiment_requires_exactly_one_model_family() -> None:
    common = {
        "dataset": DatasetSpec(name="narma10"),
        "protocol": ProtocolSpec(n_seeds=1),
    }

    with pytest.raises(ValidationError, match="exactly one"):
        ExperimentSpec(**common)
    with pytest.raises(ValidationError, match="exactly one"):
        ExperimentSpec(
            **common,
            reservoir=ReservoirSpec(type="esn"),
            baseline=BaselineSpec(type="persistence"),
        )


@pytest.mark.parametrize(
    ("seed", "n_seeds", "use_hpo", "message"),
    [
        (42, 0, False, "seed=None"),
        (None, 1, False, "n_seeds=0"),
        (None, 0, True, "cannot use Optuna HPO"),
    ],
)
def test_baseline_rejects_fake_seed_multiseed_or_hpo(
    seed: int | None,
    n_seeds: int,
    use_hpo: bool,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ExperimentSpec(
            dataset=DatasetSpec(name="uci_household_power", length=12_000),
            baseline=BaselineSpec(type="persistence"),
            protocol=ProtocolSpec(n_seeds=n_seeds, use_hpo=use_hpo),
            seed=seed,
        )


def test_reservoir_still_requires_seed_and_at_least_one_evaluation() -> None:
    with pytest.raises(ValidationError, match="reservoir experiments require a seed"):
        ExperimentSpec(
            dataset=DatasetSpec(name="narma10"),
            reservoir=ReservoirSpec(type="esn"),
            protocol=ProtocolSpec(n_seeds=1),
            seed=None,
        )
    with pytest.raises(ValidationError, match="n_seeds >= 1"):
        ExperimentSpec(
            dataset=DatasetSpec(name="narma10"),
            reservoir=ReservoirSpec(type="esn"),
            protocol=ProtocolSpec(n_seeds=0),
            seed=42,
        )


def test_optional_jmlc_metrics_aggregate_all_numeric_or_all_none() -> None:
    legacy = [_metrics(), _metrics()]
    legacy_summary = MetricsSummary.from_metrics_list(legacy, lambda xs: sum(xs) / len(xs))
    assert legacy_summary.mase is None
    assert legacy_summary.mae_skill is None
    assert legacy_summary.val_nrmse_std is None

    jmlc = [
        _metrics(mase=0.8, mae_skill=0.2, val_nrmse_std=0.4),
        _metrics(mase=1.0, mae_skill=0.0, val_nrmse_std=0.6),
    ]
    jmlc_summary = MetricsSummary.from_metrics_list(jmlc, lambda xs: sum(xs) / len(xs))
    assert jmlc_summary.mase == pytest.approx(0.9)
    assert jmlc_summary.mae_skill == pytest.approx(0.1)
    assert jmlc_summary.val_nrmse_std == pytest.approx(0.5)

    with pytest.raises(ValueError, match="mixed missing and numeric values"):
        MetricsSummary.from_metrics_list(
            [legacy[0], jmlc[0]],
            lambda xs: sum(xs) / len(xs),
        )


def test_result_traceability_metadata_roundtrips() -> None:
    selection = SelectionResult(
        method="fixed_grid",
        metric="nrmse_std",
        candidates=[
            SelectionCandidate(params={"alpha": 0.01}, score=0.4),
            SelectionCandidate(params={"alpha": 0.1}, score=0.3),
        ],
        selected_params={"alpha": 0.1},
    )
    evaluation = EvaluationContext(
        target_start_index=201,
        n_test_targets=2_199,
        seasonal_period=24,
        mase_scale=0.25,
        seasonal_test_mae=0.3,
        n_mase_scale_terms=7_111,
    )
    result = ResultSpec(
        status="completed",
        config_hash="abc",
        model_family="baseline",
        deterministic=True,
        evaluated_seeds=[],
        selection=selection,
        evaluation=evaluation,
        metrics=_metrics(mase=1.2, mae_skill=0.0, val_nrmse_std=0.3),
    )

    loaded = ResultSpec.model_validate_json(result.model_dump_json())
    assert loaded == result
