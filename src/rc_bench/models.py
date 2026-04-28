import enum
import datetime
from typing import List
from sqlalchemy import String, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean 

# Импортируем наш базовый класс Base, который мы создали в database.py
# Это связывает модели с движком БД.
from rc_bench.database import Base

# -------------------------------------------------------------------------
# Enums (Перечисления)
# Мы используем Enum, чтобы жестко ограничить возможные статусы.
# Это защищает от опечаток (например, "runing" вместо "RUNNING").
# В БД это будет сохранено как строка.
# -------------------------------------------------------------------------
class ExperimentStatus(str, enum.Enum):
    CREATED = "CREATED"     # Эксперимент создан, но еще не запущен
    QUEUED = "QUEUED"       # Отправлен в очередь (Celery)
    RUNNING = "RUNNING"     # Сейчас считается
    COMPLETED = "COMPLETED" # Успешно завершен
    FAILED = "FAILED"       # Упал с ошибкой

# -------------------------------------------------------------------------
# Модель Пользователя
# -------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Связь с экспериментами: User -> Many Experiments
    experiments: Mapped[List["Experiment"]] = relationship(
        "Experiment", back_populates="owner", cascade="all, delete-orphan"
    )

# -------------------------------------------------------------------------
# Обновляем модель Experiment (добавляем владельца)
# -------------------------------------------------------------------------
class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    reservoir_type: Mapped[str] = mapped_column(String(50), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(50), nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[ExperimentStatus] = mapped_column(
        String(20), default=ExperimentStatus.CREATED
    )

    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    
    owner: Mapped["User"] = relationship("User", back_populates="experiments")

    results: Mapped[List["Result"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    
# -------------------------------------------------------------------------
# Модель Результата
# -------------------------------------------------------------------------
class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))

    # Typed result: serialized ResultSpec (status, metrics, artifact_paths, error)
    result_data: Mapped[dict] = mapped_column(JSONB, nullable=False)

    experiment: Mapped["Experiment"] = relationship(back_populates="results")
