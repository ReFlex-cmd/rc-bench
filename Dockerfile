# ---------------------------------------------------------------------
# STAGE 1: Builder
# ---------------------------------------------------------------------
FROM python:3.12-slim as builder

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y gcc

RUN pip install "poetry>=2.3.0"

ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

COPY pyproject.toml poetry.lock README.md alembic.ini ./
COPY src ./src
COPY orchestrator ./orchestrator
COPY migrations ./migrations

RUN poetry install --only main

# ---------------------------------------------------------------------
# STAGE 2: Runtime
# ---------------------------------------------------------------------
FROM python:3.12-slim

# Создаем пользователя (безопасность)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Копируем библиотеки из builder-а
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем код
COPY --chown=user:user . .

# Копируем скрипт и даем права на выполнение
COPY --chown=user:user entrypoint.sh .
RUN chmod +x entrypoint.sh

# Устанавливаем точку входа
ENTRYPOINT ["./entrypoint.sh"]

EXPOSE 8000

CMD ["bash"]
