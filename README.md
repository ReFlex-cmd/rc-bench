# Сравнение возможностей резервуарных вычислений (RC-Bench)

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Poetry](https://img.shields.io/badge/dependency-poetry-purple)
![License](https://img.shields.io/badge/license-MIT-green)

Репозиторий содержит программный комплекс **rc-bench** и материалы выпускной квалификационной работы (ВКР). 

Проект посвящен систематическому сравнению различных классов резервуарных нейронных сетей (Reservoir Computing, RC) на едином наборе задач и метрик.

## 📌 О проекте

Резервуарные вычисления — это подход к обучению рекуррентных сетей, где обучается только выходной слой. В данной работе исследуется эффективность различных типов резервуаров для задач прогнозирования временных рядов и моделирования хаотических систем.

**Основные цели:**
1. Разработка инструментария (`rc-bench`) для проведения экспериментов.
2. Сравнительный анализ классических (ESN), спайковых (LSM) и хаотических подходов.
3. Формирование рекомендаций по выбору архитектуры.

## 🧠 Реализованные модели

В директории `src/rc_bench/core/reservoirs/` реализованы следующие типы резервуаров:

*   **ESN (Echo State Networks):** Классический подход на rate-based нейронах.
*   **LSM (Liquid State Machines):** Спайковые нейронные сети (Spiking Neural Networks).
*   **Discrete Chaos:** Резервуары на основе логистического отображения (Logistic Map).
*   **Continuous Chaos:** Резервуары на основе модели ФитцХью–Нагумо (FitzHugh–Nagumo).

## 🚀 Установка и настройка

Проект использует **Poetry** для управления зависимостями.

### Требования
* **Python 3.12+**
* **Poetry** (инструкция по установке: [python-poetry.org](https://python-poetry.org/docs/))

### Инструкция

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/ReFlex-cmd/rc-bench.git
   cd rc-bench
   ```

2. **Установите зависимости:**
   ```bash
   poetry install
   ```

3. **Подготовьте переменные окружения для API/Celery:**
   ```bash
   cp .env.example .env
   ```

4. **Команды запуска выполняйте через `poetry run`:**
   ```bash
   poetry run rcbench --help
   ```

## 🛠 Использование

В проекте реализован CLI-инструмент `rcbench` (на базе `Typer`) для управления этапами работы.

### 1. Подготовка данных
Внешние файлы скачивать не нужно. Проект использует генераторы синтетических временных рядов. 
Для генерации датасета (по умолчанию NARMA10) выполните:

```bash
poetry run rcbench prepare --task narma10 --length 20000
```
*Данные сохранятся в папку `data/prepared/narma10/` (массивы `.npy` и спецификация `spec.json`).*

### 2. Запуск API и воркера
Локально сервисную часть удобнее поднимать через Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

После запуска будут доступны:
- API через Nginx на `http://localhost/`
- Swagger UI на `http://localhost/docs`

Если нужен запуск без Docker, используйте два отдельных процесса:

```bash
poetry run uvicorn rc_bench.main:app --reload
poetry run celery -A rc_bench.celery_app worker --loglevel=info
```

### 3. Работа через API

**Регистрация и получение токена:**
```bash
# Регистрация
curl -X POST http://localhost/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secret"}'

# Получение JWT-токена
curl -X POST http://localhost/token \
  -d "username=user@example.com&password=secret"
```

**Запуск эксперимента:**
```bash
curl -X POST http://localhost/experiments/ \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "reservoir_type": "esn",
    "dataset_name": "narma10",
    "config": {"units": 500, "spectral_radius": 0.9}
  }'
```

Поддерживаемые типы резервуаров: `esn`, `lsm`, `fhn`, `logistic`.

**Получение результатов и сравнение:**
```bash
# Статус и результаты эксперимента
curl http://localhost/experiments/1

# Сравнительная таблица метрик нескольких экспериментов
curl "http://localhost/experiments/compare?ids=1,2,3" \
  -H "Authorization: Bearer <TOKEN>"

# То же в формате CSV
curl "http://localhost/experiments/compare?ids=1,2,3&format=csv" \
  -H "Authorization: Bearer <TOKEN>"

# График предсказаний (возвращает PNG)
curl http://localhost/experiments/1/plot \
  -H "Authorization: Bearer <TOKEN>" --output plot.png
```

## 📂 Структура проекта

* **`src/rc_bench/`** — Основной пакет (FastAPI + Celery):
    * `main.py` — FastAPI-приложение с эндпоинтами.
    * `tasks.py` — Celery-воркер для асинхронного запуска экспериментов.
    * `models.py` — SQLAlchemy-модели (User, Experiment, Result).
    * `schemas.py` — Pydantic-схемы запросов/ответов.
    * `core/reservoirs/` — Сервисы резервуаров:
        * `esn_service.py` — Echo State Networks.
        * `lsm_service.py` — Liquid State Machines.
        * `fhn_service.py` — FitzHugh–Nagumo (Continuous Chaos).
        * `logistic_service.py` — Logistic Map (Discrete Chaos).
    * `core/metrics.py` — Расчёт NRMSE, MSE, MAE.
    * `core/data_provider.py` — Загрузка подготовленных данных.
* **`orchestrator/`** — CLI-инструменты:
    * `cli.py` — Точка входа CLI `rcbench`.
    * `prepare_data.py` — Генератор временных рядов (NARMA10).
* **`data/`** — Сгенерированные датасеты (создаются автоматически).
* **`outputs/runs/`** — Файлы предсказаний экспериментов (preds.npy, y_test.npy).

## 📊 Результаты и метрики

Результаты экспериментов хранятся в PostgreSQL (таблица `results`) и включают:
- **NRMSE**, **MSE**, **MAE** — числовые метрики качества.
- **val_nrmse** — метрика на валидационной выборке.
- **execution_time** — время выполнения (секунды).
- **meta_data** — JSON с путями к файлам предсказаний и дополнительной информацией.

Файлы предсказаний (`preds.npy`, `y_test.npy`) сохраняются в `outputs/runs/<experiment_id>/`.

Для визуализации и сравнения используйте API:
- `GET /experiments/{id}/plot` — график «предсказание vs реальность» (PNG).
- `GET /experiments/compare?ids=1,2,3` — сравнительная таблица метрик (JSON/CSV).

## 📚 Источники

Основные работы, на которые опирается проект:
1. **Jaeger H.** (2001). The “echo state” approach to analysing and training recurrent neural networks.
2. **Maass W. et al.** (2002). Real-time computing without stable states (LSM).
3. **Kleyko et al.** (2025). Principled neuromorphic reservoir computing.

---
*Автор: Комаров Дмитрий*
