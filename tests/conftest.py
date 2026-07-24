"""Shared pytest configuration.

Application settings are deliberately strict in production. Tests receive
non-secret local defaults so unit collection does not depend on a developer's
shell, ``.env`` file, PostgreSQL, or Redis. Explicitly exported values still
win, which lets the integration suite target provisioned test services.
"""

import os
from typing import AsyncGenerator

import pytest


_TEST_ENV_DEFAULTS = {
    "POSTGRES_USER": "rc_test",
    "POSTGRES_PASSWORD": "rc_test",
    "POSTGRES_DB": "rc_bench_test",
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "REDIS_URL": "redis://localhost:6379/15",
    "SECRET_KEY": "pytest-only-secret-key",
}

for _setting_name, _default_value in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_setting_name, _default_value)

# БД-зависимости импортируются лениво — unit-тесты (metrics, data_provider,
# reservoirs) не требуют PostgreSQL и не должны падать при его отсутствии.
# TODO: для запуска test_flow.py нужны: PostgreSQL, asyncpg, psycopg2-binary,
#       httpx, pytest-asyncio и все зависимости rc_bench (см. pyproject.toml).


@pytest.fixture(scope="function", autouse=False)
async def _setup_db():
    """Создаём все таблицы перед тестами и дропаем после."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from rc_bench.config import settings
    from rc_bench.database import Base

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture()
async def db_session(_setup_db) -> AsyncGenerator:
    """Каждый тест получает сессию внутри транзакции, которая откатывается после теста."""
    from sqlalchemy.ext.asyncio import AsyncSession

    engine = _setup_db
    async with engine.connect() as conn:
        txn = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        yield session
        await session.close()
        await txn.rollback()


@pytest.fixture()
async def client(db_session) -> AsyncGenerator:
    """HTTP-клиент с переопределённой зависимостью get_db — использует тестовую сессию."""
    from httpx import AsyncClient, ASGITransport
    from rc_bench.database import get_db
    from rc_bench.main import app

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
