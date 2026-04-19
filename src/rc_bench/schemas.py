import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ConfigDict, Field
from rc_bench.models import ExperimentStatus

# --------------------------------------------------------
# Базовый класс (общие поля)
# --------------------------------------------------------
class ExperimentBase(BaseModel):
    reservoir_type: str = Field(..., description="Тип резервуара (ESN, LSM, etc.)")
    dataset_name: str = Field(..., description="Название датасета (NARMA10, Lorenz)")
    # Dict[str, Any] позволяет принимать любой JSON объект
    config: Dict[str, Any] = Field(..., description="Параметры резервуара")

# --------------------------------------------------------
# Схема для СОЗДАНИЯ (то, что шлет юзер)
# --------------------------------------------------------
class ExperimentCreate(ExperimentBase):
    """
    Схема для создания эксперимента.
    Содержит пример (example), который отобразится в Swagger UI.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "reservoir_type": "esn",
                    "dataset_name": "narma10",
                    "config": {
                        "n_units": 500,
                        "spectral_radius": 0.9,
                        "input_scaling": 0.1,
                        "seed": 42,
                        "length": 5000
                    }
                }
            ]
        }
    )
    
# --------------------------------------------------------
# Схема для РЕЗУЛЬТАТА (НОВАЯ)
# --------------------------------------------------------
class ResultRead(BaseModel):
    id: int
    nrmse: Optional[float]
    mse: Optional[float]
    mae: Optional[float]
    val_nrmse: Optional[float]
    execution_time: Optional[float]
    meta_data: dict

    model_config = ConfigDict(from_attributes=True)

# --------------------------------------------------------
# Схема для ЧТЕНИЯ Эксперимента (ОБНОВЛЕННАЯ)
# --------------------------------------------------------
class ExperimentRead(ExperimentBase):
    id: int
    created_at: datetime.datetime
    status: ExperimentStatus
    
    # Добавляем список результатов. 
    # По умолчанию пустой список, если результатов нет.
    results: List[ResultRead] = [] 

    model_config = ConfigDict(from_attributes=True)

# --------------------------------------------------------
# Схемы для ПОЛЬЗОВАТЕЛЕЙ (Auth)
# --------------------------------------------------------
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str 

class UserRead(UserBase):
    id: int
    is_active: bool
    
    model_config = ConfigDict(from_attributes=True)

# --------------------------------------------------------
# Схемы для ТОКЕНА (JWT)
# --------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
