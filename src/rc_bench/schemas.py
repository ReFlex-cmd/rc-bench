import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from rc_bench.models import ExperimentStatus
from rc_bench.core.schema import (
    ExperimentSpec,
    DatasetSpec,
    ReservoirSpec,
    ProtocolSpec,
    ReadoutSpec,
    MetricsResult,
    ResultSpec,
)

# Re-export core contracts so API layer has a single import point
__all__ = [
    "ExperimentSpec",
    "DatasetSpec",
    "ReservoirSpec",
    "ProtocolSpec",
    "ReadoutSpec",
    "MetricsResult",
    "ResultSpec",
    "ExperimentCreate",
    "ResultRead",
    "ExperimentRead",
    "UserCreate",
    "UserRead",
    "Token",
    "TokenData",
]


# --------------------------------------------------------
# Experiment input — ExperimentSpec IS the create payload
# --------------------------------------------------------
class ExperimentCreate(ExperimentSpec):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "dataset": {"name": "narma10", "length": 2000, "seed": 42},
                    "reservoir": {
                        "type": "esn",
                        "params": {
                            "n_units": 500,
                            "spectral_radius": 0.9,
                            "input_scaling": 0.1,
                        },
                    },
                    "protocol": {"washout": 200},
                    "readout": {"alpha_grid": [0.001, 0.01, 0.1, 1.0, 10.0]},
                    "seed": 42,
                }
            ]
        }
    )


# --------------------------------------------------------
# Result read — maps from JSONB result_data
# --------------------------------------------------------
class ResultRead(BaseModel):
    id: int
    result_data: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)

    @property
    def metrics(self) -> Optional[MetricsResult]:
        m = self.result_data.get("metrics")
        return MetricsResult(**m) if m else None

    @property
    def status(self) -> str:
        return self.result_data.get("status", "unknown")


# --------------------------------------------------------
# Experiment read
# --------------------------------------------------------
class ExperimentRead(BaseModel):
    id: int
    created_at: datetime.datetime
    status: ExperimentStatus
    reservoir_type: str
    dataset_name: str
    config: Dict[str, Any]
    results: List[ResultRead] = []

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------
# Auth
# --------------------------------------------------------
class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class UserRead(UserBase):
    id: int
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None
