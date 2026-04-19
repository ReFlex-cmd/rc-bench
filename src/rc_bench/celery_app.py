from celery import Celery
from rc_bench.config import settings

# Инициализация Celery
# "rc_bench" — имя нашего приложения
# broker — куда слать задачи (Redis)
# backend — куда сохранять технические результаты задач (тоже Redis)
celery_app = Celery(
    "rc_bench",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Настройки конфигурации
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Автоматический поиск задач (tasks) в нашем пакете
# Мы создадим файл tasks.py позже, но укажем его сейчас
celery_app.autodiscover_tasks(["rc_bench"])
