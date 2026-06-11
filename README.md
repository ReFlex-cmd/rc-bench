# rc-bench

> Воспроизводимый бенчмарк-сьют для сравнения типов резервуарных вычислений (Reservoir Computing).

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-289%20passed-brightgreen.svg)](#)

## Описание

**rc-bench** — открытая Python-библиотека для честного сравнительного бенчмаркинга разных
типов резервуаров на стандартных задачах прогнозирования временных рядов (NARMA-10/30,
Mackey-Glass, Lorenz-63). В ней реализовано семь моделей резервуаров — от классических
эхо-сетей (ESN, Leaky ESN, Deep ESN) до биофизических (LSM, FitzHugh–Nagumo), хаотических
(Logistic) и квантово-вдохновлённых (QRC) — под единым унифицированным протоколом обучения.
Все модели прогоняются через общий пайплайн с автоматическим подбором гиперпараметров (Optuna
TPE), линейным Ridge-readout и многосидовой статистикой (mean ± std), что делает сравнение
методологически корректным и полностью воспроизводимым. Каждый запуск сохраняет версии
библиотек, git-хеш и аппаратный профиль для повторяемости результатов.

## Реализованные модели

| Model | Type | Key parameters | Reference |
|---|---|---|---|
| **ESN** | Эхо-сеть (tanh, sparse W_rec) | `spectral_radius`, `rc_connectivity`, `input_scaling`, `units=300` | Jaeger 2001 |
| **Leaky ESN** | Эхо-сеть с утечкой | `leak_rate`, `spectral_radius`, `density`, `units=300` | Jaeger et al. 2007; Lukoševičius 2012 |
| **Deep ESN** | Иерархия ESN-слоёв | `n_layers`, per-layer `sr`/`leak_rate`, `units=100`/слой | Gallicchio et al. 2017 |
| **LSM** | Liquid State Machine (LIF-спайки) | `tau_mem`, `tau_syn`, `density`, `units=400` | Maass et al. 2002 |
| **FHN** | Сеть нейронов FitzHugh–Nagumo | `coupling_strength`, `density`, `units=200` | FitzHugh 1961; Nagumo et al. 1962 |
| **Logistic** | Связанные логистические отображения | `r_min`/`r_max`, `coupling`, `units=500` | Coupled logistic map |
| **QRC*** | Quantum Reservoir (mean-field Ising) | `n_qubits=50`, `depth`, `coupling` | Fujii & Nakajima 2017 |

\* QRC — упрощённая mean-field Ising-симуляция (классический tanh-update, не unitary);
результаты **нельзя** интерпретировать как quantum advantage.

## Quick start

### Установка

С помощью Poetry (рекомендуется):

```bash
git clone https://github.com/ReFlex-cmd/rc-bench.git
cd rc-bench
poetry install
```

Или через pip (editable-режим):

```bash
pip install -e .
```

### Запуск через CLI

После установки доступна команда `rcbench`:

```bash
# Список доступных датасетов
rcbench list-datasets

# Список зарегистрированных типов резервуаров
rcbench list-reservoirs
```

Минимальный эксперимент описывается JSON/YAML-спецификацией. Создайте `spec.yaml`:

```yaml
dataset:
  name: narma10        # narma10 | narma30 | mackey_glass | lorenz63
  length: 2000
  seed: 42
reservoir:
  type: esn            # esn | leaky_esn | deep_esn | lsm | fhn | logistic | qrc
  params:
    spectral_radius: 0.9
    rc_connectivity: 0.1
protocol:
  washout: 200
  forecasting_mode: one_step
  use_hpo: false
  n_seeds: 1
seed: 42
```

Проверьте и запустите:

```bash
# Валидация спецификации
rcbench validate-spec spec.yaml

# Запуск эксперимента с сохранением результата
rcbench run spec.yaml --output run.json

# Сводная таблица по нескольким run-record файлам
rcbench aggregate reports/runs --sort-by nrmse_range --top 10

# Markdown/CSV-отчёт + график сравнения
rcbench report reports/runs --metric nrmse_range
```

## Docker-деплой

Репозиторий включает полноценный сервис (FastAPI + Celery + PostgreSQL + Redis + Nginx) для
постановки экспериментов в очередь. Поднять стек целиком:

```bash
# 1. Создайте .env с переменными окружения (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB и т.д.)
cp .env.example .env   # отредактируйте под себя

# 2. Соберите образ и запустите все сервисы
docker compose up -d --build

# 3. Проверьте статус
docker compose ps
```

После старта API доступен через Nginx на `http://localhost:80`. Сервисы:

| Сервис | Образ / сборка | Назначение |
|---|---|---|
| `nginx` | `nginx:alpine` | Reverse-proxy, порт **80** |
| `api` | сборка из `Dockerfile` | FastAPI (`uvicorn rc_bench.main:app`) |
| `worker` | сборка из `Dockerfile` | Celery-воркер для фоновых прогонов |
| `db` | `postgres:15-alpine` | Хранилище экспериментов, порт 5432 |
| `redis` | `redis:7-alpine` | Брокер задач Celery, порт 6379 |

Остановить и удалить контейнеры:

```bash
docker compose down          # с сохранением данных
docker compose down -v       # вместе с volume postgres_data
```

## Структура проекта

```
src/rc_bench/
├── cli/
│   └── app.py                 # Typer CLI: list-*, validate-spec, run, aggregate, report
├── core/
│   ├── data_provider.py       # Каталог датасетов (NARMA, Mackey-Glass, Lorenz-63)
│   ├── metrics.py             # NRMSE, RMSE, prediction horizon, latency, memory
│   ├── schema.py              # Pydantic-схемы ExperimentSpec / ResultSpec
│   ├── security.py
│   └── reservoirs/            # Реализации резервуаров
│       ├── base.py            #   Абстрактный BaseReservoir
│       ├── registry.py        #   REGISTRY: type → класс
│       ├── esn_service.py     #   ESN (reservoirpy)
│       ├── leaky_esn_service.py
│       ├── deep_esn_service.py
│       ├── lsm_service.py     #   Liquid State Machine (LIF)
│       ├── fhn_service.py     #   FitzHugh–Nagumo
│       ├── logistic_service.py
│       └── qrc_service.py     #   Quantum Reservoir (mean-field Ising)
├── hpo/
│   ├── search_spaces.py       # Пространства поиска гиперпараметров
│   └── tuner.py               # Optuna TPE + feasibility-gate
├── protocol/
│   ├── forecasting.py         # one_step / fixed_horizon / closed_loop
│   ├── splitter.py            # train/val/test, washout
│   └── state_collector.py     # Сбор состояний резервуара
├── readout/
│   └── ridge.py               # Линейный Ridge-readout
├── runners/
│   ├── pipeline.py            # Полный пайплайн эксперимента
│   ├── experiment_runner.py
│   └── multi_seed.py          # Многосидовая статистика
├── reporting/
│   ├── run_record.py          # RunRecord: spec + metrics + метаданные
│   ├── report.py              # Генерация Markdown/CSV
│   └── plots.py               # Графики сравнения / сходимости HPO
├── main.py                    # FastAPI-приложение
├── celery_app.py              # Конфигурация Celery
├── tasks.py                   # Celery-задачи
├── database.py / models.py / schemas.py / config.py
```

## Результаты бенчмарков

Унифицированный протокол: HPO budget **100 trials** (P0/P1) или 50 (P2/P3); **10 seeds**
(P0/P1) или 5 (P2/P3); washout=200; train/val/test = 60/20/20; Ridge-readout с тюнингом α
внутри HPO; sample std (`ddof=1`) по сидам.

**NRMSE_range на тесте (mean ± std)** — чем меньше, тем лучше; **жирным** отмечен лучший
результат в столбце:

| Модель | NARMA-10 | NARMA-30 | Mackey-Glass | Lorenz-63 (closed-loop) |
|---|---|---|---|---|
| **ESN** | **0.0481 ± 0.0060** | **0.0703 ± 0.0060** | 0.0002 ± 0.0000 | 1.4735 ± 0.9346 |
| **Leaky ESN** | 0.0687 ± 0.0324 | 0.1848 ± 0.1912 | **0.0001 ± 0.0000** | **0.3027 ± 0.0107** |
| **Deep ESN** | 0.0656 ± 0.0133 | 0.0853 ± 0.0068 | **0.0001 ± 0.0000** | **0.3038 ± 0.0167** |
| **LSM** | 0.1352 ± 0.0065 | — | 0.0358 ± 0.0021 | — |
| **FHN** | 0.1667 ± 0.0152 | — | 0.2178 ± 0.0020 | — |
| **Logistic** | 0.1975 ± 0.0044 | — | 0.0707 ± 0.0017 | — |
| **QRC*** | 0.1064 ± 0.0046 | — | — | — |

Полная эмпирическая сводка с иерархией моделей, сравнением с литературой и известными
ограничениями — в [reports/empirical_summary.md](reports/empirical_summary.md).

## Лицензия

Проект распространяется под лицензией [MIT](LICENSE).
