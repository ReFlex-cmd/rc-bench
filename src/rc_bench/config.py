from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic сам найдет эти переменные в .env
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    REDIS_URL: str = "redis://localhost:6379/0"

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Directory where the Celery worker saves prediction artifacts (.npz).
    # Override via ARTIFACT_DIR env var or in .env.
    ARTIFACT_DIR: Path = Field(default=Path("./outputs/artifacts"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def DATABASE_URL(self) -> str:
        # Собираем строку подключения из переменных Postgres
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.POSTGRES_DB}"
        
    @property
    def DATABASE_URL_SYNC(self) -> str:
        # Для Celery и синхронных скриптов
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.POSTGRES_DB}"

settings = Settings()