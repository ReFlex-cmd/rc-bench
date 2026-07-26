import hashlib
import json
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetSpec(BaseModel):
    name: str
    length: int = 2000
    seed: int = 42
    raw_sha256: Optional[str] = None

    @field_validator("raw_sha256")
    @classmethod
    def _validate_raw_sha256(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized
        ):
            raise ValueError("raw_sha256 must be a 64-character hexadecimal digest")
        return normalized


class ReservoirSpec(BaseModel):
    type: Literal["esn", "lsm", "fhn", "logistic", "leaky_esn", "deep_esn", "qrc"]
    params: Dict[str, Any] = Field(default_factory=dict)


class BaselineSpec(BaseModel):
    type: Literal["persistence", "seasonal_persistence", "ridge_ar"]
    params: Dict[str, Any] = Field(default_factory=dict)


class ProtocolSpec(BaseModel):
    washout: int = 200
    train_frac: float = 0.6
    val_frac: float = 0.2
    forecasting_mode: Literal["one_step", "fixed_horizon", "closed_loop"] = "one_step"
    horizon: int = 1      # steps ahead; only used when forecasting_mode == "fixed_horizon"
    # Stage 8: HPO
    use_hpo: bool = False
    # Per ТЗ §2: minimum 50, target 100 trials per (model, task) cell.
    hpo_budget: int = 100
    # Stage 8: multi-seed. Per ТЗ §3: target 10, minimum 5.
    n_seeds: int = 10
    # JMLC uses the headline metric for selection; the legacy default remains
    # range-normalized NRMSE so existing synthetic specs keep their semantics.
    selection_metric: Literal["nrmse_range", "nrmse_std"] = "nrmse_range"
    # Enables train-only MASE and test seasonal-skill accounting when present.
    seasonal_period: Optional[int] = None
    # Режим сравнения. ``fair`` — единый вычислительный бюджет и одинаковые
    # правила отбора для всех reservoir-моделей (DEC-004); ``best_effort`` —
    # индивидуальная настройка каждой архитектуры. Поле живёт в протоколе, а
    # не в шаблоне матрицы, чтобы попадать во frozen и resolved spec каждой
    # ячейки: иначе по самой записи нельзя было бы сказать, в каких условиях
    # получено число. Результаты двух режимов никогда не сводятся в одну
    # таблицу.
    mode: Literal["fair", "best_effort"] = "fair"


class ReadoutSpec(BaseModel):
    type: str = "ridge"
    alpha_grid: List[float] = Field(
        default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0]
    )


class ExperimentSpec(BaseModel):
    dataset: DatasetSpec
    reservoir: Optional[ReservoirSpec] = None
    baseline: Optional[BaselineSpec] = None
    protocol: ProtocolSpec = Field(default_factory=ProtocolSpec)
    readout: ReadoutSpec = Field(default_factory=ReadoutSpec)
    seed: Optional[int] = 42

    @model_validator(mode="after")
    def _validate_model_family(self) -> "ExperimentSpec":
        if (self.reservoir is None) == (self.baseline is None):
            raise ValueError(
                "experiment must define exactly one of reservoir or baseline"
            )

        if self.baseline is not None:
            if self.seed is not None:
                raise ValueError("baseline experiments require seed=None")
            if self.protocol.n_seeds != 0:
                raise ValueError("baseline experiments require protocol.n_seeds=0")
            if self.protocol.use_hpo:
                raise ValueError("baseline experiments cannot use Optuna HPO")
        else:
            if self.seed is None:
                raise ValueError("reservoir experiments require a seed")
            if self.protocol.n_seeds < 1:
                raise ValueError(
                    "reservoir experiments require protocol.n_seeds >= 1"
                )
        return self

    @property
    def model_family(self) -> Literal["reservoir", "baseline"]:
        return "baseline" if self.baseline is not None else "reservoir"

    @property
    def model_type(self) -> str:
        if self.baseline is not None:
            return self.baseline.type
        assert self.reservoir is not None
        return self.reservoir.type

    def config_hash(self) -> str:
        payload = self.model_dump()
        if payload.get("reservoir") is None:
            payload.pop("reservoir", None)
        if payload.get("baseline") is None:
            payload.pop("baseline", None)

        dataset = payload["dataset"]
        if dataset.get("raw_sha256") is None:
            dataset.pop("raw_sha256", None)

        protocol = payload["protocol"]
        if protocol.get("selection_metric") == "nrmse_range":
            protocol.pop("selection_metric", None)
        if protocol.get("seasonal_period") is None:
            protocol.pop("seasonal_period", None)
        # Значение по умолчанию выбрасывается, иначе добавление поля
        # переименовало бы каждую уже опубликованную ячейку бандла, ничего в
        # ней не изменив по существу.
        if protocol.get("mode") == "fair":
            protocol.pop("mode", None)

        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class MetricsResult(BaseModel):
    # Accuracy
    rmse: float
    nrmse_range: float
    nrmse_std: float
    nrmse_var: float
    mae: float
    mse: float
    prediction_horizon: int
    # Validation
    val_nrmse_range: float
    val_nrmse_std: Optional[float] = None
    # JMLC baseline-relative accuracy. Optional for legacy/synthetic records.
    mase: Optional[float] = None
    mae_skill: Optional[float] = None
    # Cost
    train_time: float
    inference_latency: float
    peak_memory: int  # bytes


class MetricsSummary(BaseModel):
    """Floating-point aggregate (mean or std) over multiple MetricsResult instances.

    All fields are float so that mean/std of int MetricsResult fields
    (prediction_horizon, peak_memory) can be represented without loss.
    """
    rmse: float
    nrmse_range: float
    nrmse_std: float
    nrmse_var: float
    mae: float
    mse: float
    prediction_horizon: float
    val_nrmse_range: float
    val_nrmse_std: Optional[float] = None
    mase: Optional[float] = None
    mae_skill: Optional[float] = None
    train_time: float
    inference_latency: float
    peak_memory: float

    @classmethod
    def from_metrics_list(
        cls,
        metrics: List[MetricsResult],
        agg: Callable[[List[float]], float],
    ) -> "MetricsSummary":
        fields = cls.model_fields.keys()
        aggregated: Dict[str, Optional[float]] = {}
        for field in fields:
            values = [getattr(metric, field) for metric in metrics]
            if all(value is None for value in values):
                aggregated[field] = None
                continue
            if any(value is None for value in values):
                raise ValueError(
                    f"{field} has mixed missing and numeric values"
                )
            aggregated[field] = float(
                agg([float(value) for value in values if value is not None])
            )
        return cls(**aggregated)


class MultiSeedResult(BaseModel):
    n_seeds: int
    seeds: List[int]
    metrics_per_seed: List[MetricsResult]
    mean: MetricsSummary
    std: MetricsSummary


#: Поля, которые есть ровно тогда, когда энергия действительно измерена.
MEASURED_ENERGY_FIELDS = (
    "backend",
    "window_target_met",
    "net_energy_per_inference_mj",
    "net_samples_per_joule",
    "energy_delay_product_j_s",
)


class EnergyResult(BaseModel):
    """Энергия вывода: измерена аппаратным счётчиком или явно недоступна.

    Прокси-метрики активности (MAC, разреженность состояния, спайки) сюда не
    попадают ни при каких условиях — они живут в блоке ``activity`` профиля.
    Перевод прокси в джоули требует модели энергии железа, которой у нас нет,
    а число в поле «энергия» читается как измерение (DEC-007).

    ``unavailable`` остаётся значением по умолчанию, поэтому записи, сделанные
    до появления RAPL-бэкенда, читаются без миграции.
    """

    status: Literal["measured", "unavailable"] = "unavailable"
    reason: Optional[str] = "No supported hardware energy counter available"
    backend: Optional[Literal["intel_rapl"]] = None
    domains: List[str] = Field(default_factory=list)
    #: Успело ли окно измерения набрать заданную длительность. ``False``
    #: означает, что измерение упёрлось в потолок шагов раньше, и его точность
    #: ограничена периодом обновления счётчика; без этого поля усечённое окно
    #: было бы неотличимо от полноценного.
    window_target_met: Optional[bool] = None
    net_energy_per_inference_mj: Optional[float] = None
    net_samples_per_joule: Optional[float] = None
    energy_delay_product_j_s: Optional[float] = None

    @model_validator(mode="after")
    def _status_and_numbers_must_agree(self) -> "EnergyResult":
        """Статус и числа — два утверждения об одном факте; расходиться им
        нельзя. ``measured`` без чисел не говорит, сколько намерено, а числа
        при ``unavailable`` появились неизвестно откуда."""
        if self.status == "measured":
            missing = [
                name for name in MEASURED_ENERGY_FIELDS if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"status='measured' requires {', '.join(missing)}")
            return self

        # Обратное направление: любое поле, существующее только при измерении,
        # при 'unavailable' появилось неизвестно откуда. domains проверяется
        # тоже — список доменов счётчика взять неоткуда, если счётчика не было.
        present = [
            name for name in MEASURED_ENERGY_FIELDS if getattr(self, name) is not None
        ]
        if self.domains:
            present.append("domains")
        if present:
            raise ValueError(
                f"status='unavailable' cannot carry {', '.join(present)}"
            )
        return self


class SelectionCandidate(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)
    score: float


class SelectionResult(BaseModel):
    method: Literal["none", "fixed_grid", "optuna"]
    metric: Optional[Literal["nrmse_range", "nrmse_std"]] = None
    candidates: List[SelectionCandidate] = Field(default_factory=list)
    selected_params: Dict[str, Any] = Field(default_factory=dict)


class EvaluationContext(BaseModel):
    target_start_index: Optional[int] = None
    n_test_targets: Optional[int] = None
    seasonal_period: Optional[int] = None
    mase_scale: Optional[float] = None
    seasonal_test_mae: Optional[float] = None
    n_mase_scale_terms: Optional[int] = None


class ResultSpec(BaseModel):
    status: Literal["completed", "failed"]
    # ``config_hash`` always identifies the actually evaluated resolved spec.
    config_hash: str
    # Kept alongside the resolved hash so HPO runs remain traceable to the
    # immutable user input. Optional defaults preserve legacy/failed records.
    frozen_config_hash: Optional[str] = None
    resolved_spec: Optional[ExperimentSpec] = None
    metrics: Optional[MetricsResult] = None
    multi_seed_result: Optional[MultiSeedResult] = None
    hpo_best_params: Optional[Dict[str, Any]] = None
    # Best-so-far val_nrmse_range after each completed trial; used for the
    # hpo_convergence_<model>_<task>.png plots required by ТЗ §5.2.
    hpo_convergence: Optional[List[float]] = None
    # HPO accounting (n_completed/n_pruned/n_failed/best_trial_number) — see audit/03 §3.7.3.
    hpo_diagnostics: Optional[Dict[str, int]] = None
    # Explicit execution identity for JMLC evidence. Optional defaults preserve
    # compatibility with pre-JMLC ResultSpec JSON.
    model_family: Optional[Literal["reservoir", "baseline"]] = None
    deterministic: Optional[bool] = None
    evaluated_seeds: Optional[List[int]] = None
    selection: Optional[SelectionResult] = None
    evaluation: Optional[EvaluationContext] = None
    energy: EnergyResult = Field(default_factory=EnergyResult)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
