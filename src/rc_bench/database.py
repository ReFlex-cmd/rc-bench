from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
# Вот этой строки не хватало:
from sqlalchemy import create_engine 

from rc_bench.config import settings

# -----------------------------------------------------------------------------
# 1. Асинхронное подключение (для FastAPI)
# -----------------------------------------------------------------------------
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True 
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# -----------------------------------------------------------------------------
# 2. Синхронное подключение (для Celery и скриптов)
# -----------------------------------------------------------------------------
sync_engine = create_engine(
    settings.DATABASE_URL_SYNC,
    echo=False # Для воркера обычно отключаем лишний шум SQL
)

# Фабрика синхронных сессий
sync_session_factory = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=sync_engine
)

# -----------------------------------------------------------------------------
# 3. Базовый класс моделей
# -----------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass

# -----------------------------------------------------------------------------
# 4. Утилиты (Dependencies)
# -----------------------------------------------------------------------------

# Асинхронная зависимость для FastAPI
async def get_db():
    async with async_session_factory() as session:
        yield session

# Синхронный контекстный менеджер для Celery
def get_sync_db_session():
    return sync_session_factory()
