#!/bin/bash
# entrypoint.sh

# Если любая команда упадет — скрипт остановится (Best Practice)
set -e

# Накатываем миграции
echo "🚀 Running database migrations..."
alembic upgrade head

# Запускаем команду, которую передал Docker (uvicorn или celery)
echo "🔥 Starting application..."
exec "$@"
