# Architecture Map — Ground-Truth Audit

_Дата: 2026-06-03. Проверено по исходникам, не по документации._

---

## 1. Компоненты и файлы

| Компонент | Файл(ы) | Статус |
|---|---|---|
| **FastAPI app** | `src/rc_bench/main.py` | ✅ Полностью подключён |
| **Celery app** | `src/rc_bench/celery_app.py` | ✅ Полностью подключён |
| **Celery task** | `src/rc_bench/tasks.py` | ✅ Одна задача `run_experiment_task` |
| **PostgreSQL models** | `src/rc_bench/models.py` | ✅ 3 таблицы: `users`, `experiments`, `results` |
| **DB сессии** | `src/rc_bench/database.py` | ✅ async (FastAPI) + sync (Celery) |
| **Config / Settings** | `src/rc_bench/config.py` | ✅ pydantic-settings, читает `.env` |
| **Schemas (API)** | `src/rc_bench/schemas.py` | ✅ Pydantic-модели для API layer |
| **Core schemas** | `src/rc_bench/core/schema.py` | ✅ ExperimentSpec, MetricsResult, ResultSpec, … |
| **Security (JWT)** | `src/rc_bench/core/security.py` | ✅ bcrypt + python-jose |
| **Data provider** | `src/rc_bench/core/data_provider.py` | ✅ 4 датасета генерируются на лету |
| **Reservoir registry** | `src/rc_bench/core/reservoirs/registry.py` | ✅ 7 типов |
| **Base reservoir** | `src/rc_bench/core/reservoirs/base.py` | ✅ ABC + step-by-step API |
| **ESN** | `…/reservoirs/esn_service.py` | ✅ |
| **LeakyESN** | `…/reservoirs/leaky_esn_service.py` | ✅ |
| **DeepESN** | `…/reservoirs/deep_esn_service.py` | ✅ |
| **LSM** | `…/reservoirs/lsm_service.py` | ✅ |
| **FHN** | `…/reservoirs/fhn_service.py` | ✅ |
| **Logistic** | `…/reservoirs/logistic_service.py` | ✅ |
| **QRC** | `…/reservoirs/qrc_service.py` | ✅ |
| **Readout (Ridge)** | `src/rc_bench/readout/ridge.py` | ✅ alpha выбирается по val |
| **Experiment runner** | `src/rc_bench/runners/experiment_runner.py` | ✅ полный train/val/test цикл |
| **Multi-seed runner** | `src/rc_bench/runners/multi_seed.py` | ✅ n_seeds варьирует только seed резервуара |
| **Pipeline** | `src/rc_bench/runners/pipeline.py` | ✅ точка входа: HPO → multi-seed/single |
| **HPO tuner** | `src/rc_bench/hpo/tuner.py` | ✅ Optuna TPE + MedianPruner |
| **HPO search spaces** | `src/rc_bench/hpo/search_spaces.py` | ✅ |
| **Forecasting protocol** | `src/rc_bench/protocol/forecasting.py` | ✅ one_step / fixed_horizon / closed_loop |
| **State collector** | `src/rc_bench/protocol/state_collector.py` | ✅ washout trimming |
| **Metrics** | `src/rc_bench/core/metrics.py` | ✅ rmse, nrmse_*, mae, mse, ph |
| **CLI** | `src/rc_bench/cli/app.py` | ✅ Typer; вызывает pipeline **напрямую** (без API/Celery) |
| **Orchestrator CLI** | `orchestrator/cli.py` | ✅ batch-запуск через CLI |
| **Orchestrator data** | `orchestrator/prepare_data.py` | ✅ сохраняет .npy в `data/prepared/` |
| **Scripts** | `scripts/full_grid.py`, `smoke_grid.py`, `make_plots.py`, `rerun_lorenz.py` | ✅ ad-hoc батчи |
| **Reporting** | `src/rc_bench/reporting/` | ✅ отчёты + plots |
| **Nginx** | `nginx/nginx.conf` | ✅ reverse proxy → api:8000, только HTTP |
| **Docker** | `Dockerfile`, `docker-compose.yml`, `entrypoint.sh` | ✅ 5 сервисов |
| **Alembic migrations** | `migrations/versions/` | ✅ 4 версии |
| **Ansible** | `ansible/playbook.yml`, `inventory.ini` | ⚠️ Файлы есть, содержимое не проверялось |

---

## 2. Реальные связи между компонентами

```
[Клиент / Browser]
        │  HTTP :80
        ▼
[Nginx]  nginx/nginx.conf
        │  proxy_pass http://api:8000
        ▼
[FastAPI]  src/rc_bench/main.py
  ├── POST /experiments/
  │     ├── asyncpg → PostgreSQL (запись Experiment)
  │     └── run_experiment_task.delay(id) → Redis (брокер)
  ├── GET  /experiments/{id}
  │     └── asyncpg → PostgreSQL (чтение Experiment + Result)
  ├── GET  /experiments/{id}/plot
  │     ├── asyncpg → PostgreSQL (чтение artifact_paths из result_data)
  │     └── filesystem → outputs/artifacts/{id}/*.npz
  ├── POST /token  → asyncpg → PostgreSQL
  └── POST /register → asyncpg → PostgreSQL

[Redis]  broker + backend
        │  AMQP-like queue
        ▼
[Celery Worker]  src/rc_bench/tasks.py :: run_experiment_task
  ├── psycopg2 → PostgreSQL (sync: чтение Experiment, запись Result)
  ├── run_pipeline()
  │     ├── (опционально) HPO: Optuna in-process
  │     ├── (опционально) multi_seed runner
  │     └── reservoir.transform() → RidgeReadout → MetricsResult
  └── filesystem → outputs/artifacts/{exp_id}/*.npz

[CLI / rcbench run]  src/rc_bench/cli/app.py
  └── run_pipeline() напрямую (БЕЗ API, БЕЗ Celery)
        └── filesystem → outputs/ (локально)

[Orchestrator / scripts]  orchestrator/cli.py, scripts/*.py
  └── run_pipeline() напрямую (БЕЗ API, БЕЗ Celery)
```

---

## 3. PostgreSQL — таблицы и миграции

| Таблица | Ключевые колонки | Заметки |
|---|---|---|
| `users` | id, email, hashed_password, is_active | миграция `a253bc84cc78` |
| `experiments` | id, reservoir_type, dataset_name, config (JSONB), status, owner_id | owner_id NOT NULL с `c7f3a1d9e042` |
| `results` | id, experiment_id, result_data (JSONB) | полная замена fixed-колонок на JSONB в `d4e8f1a2b3c5` |

`result_data` хранит сериализованный `ResultSpec` целиком:
`{status, config_hash, metrics, multi_seed_result, hpo_best_params, hpo_convergence, hpo_diagnostics, artifact_paths, error}`.

Цепочка миграций: `4b02e7a79d21` → `a253bc84cc78` → `c7f3a1d9e042` → `d4e8f1a2b3c5`.

---

## 4. Redis — реальное использование

- **Брокер**: очередь задач Celery (`REDIS_URL` → `broker=`).
- **Backend**: хранение технических статусов задач Celery (`backend=`).
- **Прямого хранения данных нет**: кеш, сессии, pub/sub — не используются.

---

## 5. Docker Compose — 5 сервисов

| Сервис | Image / Build | Команда | Порты (внешние) |
|---|---|---|---|
| `db` | postgres:15-alpine | — | 5432 |
| `redis` | redis:7-alpine | — | 6379 |
| `api` | build: . | `uvicorn rc_bench.main:app --host 0.0.0.0 --port 8000` | — (закомментирован) |
| `worker` | build: . | `celery -A rc_bench.celery_app worker` | — |
| `nginx` | nginx:alpine | — | **80** |

`entrypoint.sh` запускает `alembic upgrade head` перед стартом сервиса.
`./outputs` монтируется как volume в `api` и `worker` для передачи `.npz`-артефактов.

---

## 6. Что отсутствует / частично

| Пункт | Состояние |
|---|---|
| **HTTPS / TLS в Nginx** | Отсутствует: только HTTP :80 |
| **Flower (Celery UI)** | Отсутствует в docker-compose |
| **Celery Beat (планировщик)** | Не настроен |
| **Rate limiting** | Нет ни в Nginx, ни в FastAPI |
| **Streaming / WebSocket** для статуса задачи | Нет: только polling GET /experiments/{id} |
| **Frontend / UI** | Только REST API + CLI |
| **`data/prepared/`** | Есть только `narma10/*.npy` (5 файлов); runtime их **не использует** — данные генерируются заново через `data_provider.py` |
| **Ansible** | Файлы есть (`playbook.yml`, `inventory.ini`), содержимое не аудировалось |
| **graphify-out/** | Кеш внешнего инструмента визуализации, не часть приложения |

---

## 7. Два независимых пути запуска

Это архитектурный факт, важный для главы 4:

| Путь | Точка входа | БД | Redis/Celery | Артефакты |
|---|---|---|---|---|
| **API** | `POST /experiments/` → Celery | PostgreSQL (async+sync) | ✅ | `outputs/artifacts/{id}/` |
| **CLI / скрипты** | `rcbench run` / `scripts/*.py` | ❌ не используется | ❌ | `outputs/` локально |

Pipeline (`run_pipeline`) один и тот же в обоих случаях.
