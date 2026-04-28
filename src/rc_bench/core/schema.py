import hashlib
import json
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field


class DatasetSpec(BaseModel):
    name: str
    length: int = 2000
    seed: int = 42


class ReservoirSpec(BaseModel):
    type: Literal["esn", "lsm", "fhn", "logistic", "leaky_esn", "deep_esn", "qrc"]
    params: Dict[str, Any] = Field(default_factory=dict)


class ProtocolSpec(BaseModel):
    washout: int = 200
    train_frac: float = 0.6
    val_frac: float = 0.2
    forecasting_mode: Literal["one_step", "fixed_horizon", "closed_loop"] = "one_step"
    horizon: int = 1      # steps ahead; only used when forecasting_mode == "fixed_horizon"
    # Stage 8: HPO
    use_hpo: bool = False
    hpo_budget: int = 20  # number of Optuna trials
    # Stage 8: multi-seed (set n_seeds > 1 to enable; 5 recommended)
    n_seeds: int = 1


class ReadoutSpec(BaseModel):
    type: str = "ridge"
    alpha_grid: List[float] = Field(
        default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0]
    )


class ExperimentSpec(BaseModel):
    dataset: DatasetSpec
    reservoir: ReservoirSpec
    protocol: ProtocolSpec = Field(default_factory=ProtocolSpec)
    readout: ReadoutSpec = Field(default_factory=ReadoutSpec)
    seed: int = 42

    def config_hash(self) -> str:
        raw = json.dumps(self.model_dump(), sort_keys=True, default=str)
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
        return cls(**{
            f: float(agg([getattr(m, f) for m in metrics]))
            for f in fields
        })


class MultiSeedResult(BaseModel):
    n_seeds: int
    seeds: List[int]
    metrics_per_seed: List[MetricsResult]
    mean: MetricsSummary
    std: MetricsSummary


class ResultSpec(BaseModel):
    status: Literal["completed", "failed"]
    config_hash: str
    metrics: Optional[MetricsResult] = None
    multi_seed_result: Optional[MultiSeedResult] = None
    hpo_best_params: Optional[Dict[str, Any]] = None
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
