from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rc_bench.core.schema import (
    BaselineSpec,
    DatasetSpec,
    EnergyResult,
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


PUBLISHED_ESN_H1_RUN = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "jmlc_2026"
    / "fair"
    / "runs"
    / "reservoir_esn_h1.json"
)


def test_legacy_reservoir_hash_is_unchanged_by_additive_schema_fields() -> None:
    spec = _legacy_reservoir_spec()

    assert spec.model_family == "reservoir"
    assert spec.model_type == "esn"
    assert spec.config_hash() == "610160f0e11c6eb0"


def test_default_mode_does_not_change_published_config_hashes() -> None:
    """Значение по умолчанию выбрасывается из payload, как уже сделано для
    selection_metric и seasonal_period.

    Иначе добавление поля переименует каждую опубликованную ячейку
    evidence-бандла: `config_hash` связывает результат, профиль и агрегат, и
    его смена рвёт traceability, ничего не меняя по существу. Хеш берётся из
    самой опубликованной записи, а не из строкового литерала, — так тест
    ломается, если разъедется любая из двух сторон.
    """
    published = json.loads(PUBLISHED_ESN_H1_RUN.read_text(encoding="utf-8"))
    spec = ExperimentSpec.model_validate(published["spec"])

    assert spec.protocol.mode == "fair"
    assert spec.config_hash() == published["result"]["frozen_config_hash"]


def test_best_effort_mode_changes_the_config_hash() -> None:
    """Режимы обязаны различаться по хешу: иначе fair- и best-effort-ячейка с
    совпадающими прочими параметрами склеились бы в одну строку агрегата."""
    published = json.loads(PUBLISHED_ESN_H1_RUN.read_text(encoding="utf-8"))
    fair = ExperimentSpec.model_validate(published["spec"])
    best_effort = ExperimentSpec.model_validate(
        {
            **published["spec"],
            "protocol": {**published["spec"]["protocol"], "mode": "best_effort"},
        }
    )

    assert best_effort.protocol.mode == "best_effort"
    assert fair.config_hash() != best_effort.config_hash()


class TestEnergyResult:
    def test_a_measured_backend_is_accepted(self) -> None:
        measured = EnergyResult(
            status="measured",
            backend="intel_rapl",
            window_target_met=True,
            net_energy_per_inference_mj=0.0123,
            net_samples_per_joule=81_300.0,
            energy_delay_product_j_s=2.7e-10,
            domains=["package-0"],
        )

        assert measured.status == "measured"
        assert measured.domains == ["package-0"]

    def test_a_truncated_window_is_recorded_not_hidden(self) -> None:
        """Окно, упёршееся в потолок шагов, — всё ещё измерение, но читатель
        обязан узнать об этом из самой записи."""
        truncated = EnergyResult(
            status="measured",
            backend="intel_rapl",
            window_target_met=False,
            net_energy_per_inference_mj=0.0123,
            net_samples_per_joule=81_300.0,
            energy_delay_product_j_s=2.7e-10,
        )

        assert truncated.window_target_met is False

    def test_unavailable_stays_the_default(self) -> None:
        """Прежние записи читаются без миграции, а машина без счётчика
        по умолчанию не притворяется измеренной."""
        default = EnergyResult()

        assert default.status == "unavailable"
        assert default.backend is None
        assert default.net_energy_per_inference_mj is None

    @pytest.mark.parametrize(
        "omitted",
        [
            "backend",
            "window_target_met",
            "net_energy_per_inference_mj",
            "net_samples_per_joule",
            "energy_delay_product_j_s",
        ],
    )
    def test_measured_without_its_numbers_is_refused(self, omitted: str) -> None:
        """`measured` без числа — худший вид записи: она утверждает, что
        энергия померена, но не говорит сколько."""
        payload = {
            "status": "measured",
            "backend": "intel_rapl",
            "window_target_met": True,
            "net_energy_per_inference_mj": 0.0123,
            "net_samples_per_joule": 81_300.0,
            "energy_delay_product_j_s": 2.7e-10,
        }
        payload.pop(omitted)

        with pytest.raises(ValidationError, match=omitted):
            EnergyResult(**payload)

    def test_unavailable_with_numbers_is_refused(self) -> None:
        """Обратная подмена: числа есть, а статус говорит, что мерить было
        нечем. Одно из двух утверждений ложно."""
        with pytest.raises(ValidationError, match="unavailable"):
            EnergyResult(status="unavailable", net_energy_per_inference_mj=0.0123)


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
