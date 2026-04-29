# RC-Bench — Benchmark Suite for Reservoir Computing

![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-287%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

Воспроизводимый benchmark-suite для систематического сравнения архитектур **reservoir computing** на единых датасетах, метриках и протоколе оценки. Реализован как ВКР.

---

## Что умеет

- **7 архитектур резервуаров** — от классического ESN до квантового QRC
- **4 датасета** — NARMA10/30, Mackey-Glass, Lorenz-63
- **3 режима прогнозирования** — one-step, fixed-horizon, closed-loop
- **HPO (Optuna)** — автоматический подбор гиперпараметров
- **Multi-seed** — оценка mean ± std по N запускам
- **Reporting** — `report.md`, `report.csv`, bar-charts, `RunRecord` с reproducibility-метаданными
- **REST API** — FastAPI + Celery + PostgreSQL + Redis (Docker Compose)

---

## Архитектуры резервуаров

| Тип | Класс | Описание |
|-----|-------|----------|
| `esn` | ESNReservoir | Echo State Network, rate-based нейроны |
| `leaky_esn` | LeakyESNReservoir | ESN с leak-rate: `x = (1-α)x + α·tanh(...)` |
| `deep_esn` | DeepESNReservoir | Стек ESN-слоев, выход конкатенируется |
| `lsm` | LSMReservoir | Liquid State Machine, LIF спайковые нейроны |
| `fhn` | FHNReservoir | FitzHugh-Nagumo (RK4, непрерывный хаос) |
| `logistic` | LogisticReservoir | Логистическое отображение (дискретный хаос) |
| `qrc` | QRCReservoir | Квантовый RC — Ising mean-field + virtual nodes |

Все архитектуры реализуют единый интерфейс `BaseReservoir`: `transform()`, `step()`, `warmup()`, `reset_state()`, `sanity_check()`.

---

## Датасеты

| Имя | Описание | Длина по умолчанию |
|-----|----------|--------------------|
| `narma10` | NARMA порядка 10 | 2 000 |
| `narma30` | NARMA порядка 30 (длинная память) | 3 000 |
| `mackey_glass` | DDE Mackey-Glass (τ=17, хаотический режим) | 5 000 |
| `lorenz63` | Аттрактор Лоренца, x-компонента (RK4, dt=0.01) | 5 000 |

---

## Установка

```bash
git clone https://github.com/ReFlex-cmd/rc-bench.git
cd rc-bench
poetry install
```

**Для REST API и воркера** нужен `.env` (скопируйте из примера):

```bash
cp .env.example .env   # заполните POSTGRES_*, SECRET_KEY
docker compose up --build
```

---

## CLI — быстрый старт

Основной инструмент — `rcbench` (Typer CLI).

### Просмотр доступных ресурсов

```bash
rcbench list-datasets
rcbench list-reservoirs
```

### Создание ExperimentSpec

Эксперимент описывается YAML- или JSON-файлом:

```yaml
# spec.yaml
dataset:
  name: narma10
  length: 2000
  seed: 42

reservoir:
  type: esn
  params:
    units: 500
    spectral_radius: 0.9
    density: 0.1

protocol:
  washout: 200
  train_frac: 0.6
  val_frac: 0.2
  forecasting_mode: one_step   # one_step | fixed_horizon | closed_loop
  use_hpo: false
  hpo_budget: 20
  n_seeds: 1

readout:
  alpha_grid: [0.001, 0.01, 0.1, 1.0, 10.0]

seed: 42
```

### Валидация

```bash
rcbench validate-spec spec.yaml
# => ✓ Valid  config_hash=a3f1c7e2b9d40851
```

### Запуск эксперимента

```bash
# Простой запуск
rcbench run spec.yaml

# Сохранить run record + prediction artifacts
rcbench run spec.yaml --output results/run1.json --artifacts results/artifacts/

# Переопределить seed
rcbench run spec.yaml --seed 7
```

Каждый run record (`.json`) содержит полный `ExperimentSpec`, `ResultSpec` и reproducibility-метаданные: timestamp, git hash, hostname, python version.

### Multi-seed и HPO

```yaml
protocol:
  use_hpo: true
  hpo_budget: 30     # число trials Optuna
  n_seeds: 5         # mean ± std по 5 seed'ам
```

```bash
rcbench run spec_hpo.yaml --output results/hpo_run.json
```

### Сравнение нескольких запусков

```bash
# Таблица в терминале
rcbench aggregate results/ --sort-by nrmse_range --top 5

# Полный отчёт: report.md + report.csv + comparison_nrmse_range.png
rcbench report results/ --output-dir report/ --metric nrmse_range
```

---

## Метрики

Все метрики вычисляются в `rc_bench/core/metrics.py` и возвращаются в `MetricsResult`.

| Метрика | Описание |
|---------|----------|
| `nrmse_range` | RMSE / (max - min) — основная метрика точности |
| `nrmse_std` | RMSE / std(y_true) |
| `nrmse_var` | MSE / var(y_true) — 1.0 у mean-predictor |
| `rmse` | Root Mean Squared Error |
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `prediction_horizon` | Шагов до превышения порога 0.3·std |
| `val_nrmse_range` | `nrmse_range` на валидационной выборке |
| `train_time` | Время обучения, сек |
| `inference_latency` | Время инференса, сек |
| `peak_memory` | Пиковое потребление памяти, байт |

---

## Протокол оценки

Единый протокол гарантирует честное сравнение:

1. **Split** — contiguous train/val/test (без перемешивания)
2. **Washout** — первые N шагов состояния резервуара отбрасываются
3. **Readout** — Ridge Regression; alpha выбирается по `val_nrmse_range` на grid
4. **Fit** — финальная модель дообучается на train + val
5. **Evaluate** — метрики считаются только на test

```
[-------- train (60%) --------][-- val (20%) --][-- test (20%) --]
[  washout  ][  H_train used  ]
```

---

## Структура проекта

```
src/rc_bench/
├── cli/app.py                  # Typer CLI (list-datasets, run, report, aggregate, ...)
├── core/
│   ├── schema.py               # Pydantic-контракты (ExperimentSpec, MetricsResult, ...)
│   ├── metrics.py              # Все метрики качества
│   ├── data_provider.py        # Генераторы датасетов
│   └── reservoirs/
│       ├── base.py             # ABC BaseReservoir
│       ├── registry.py         # REGISTRY + get_reservoir()
│       ├── esn_service.py
│       ├── leaky_esn_service.py
│       ├── deep_esn_service.py
│       ├── lsm_service.py
│       ├── fhn_service.py
│       ├── logistic_service.py
│       └── qrc_service.py
├── protocol/
│   ├── splitter.py             # Train/val/test split
│   ├── state_collector.py      # Сбор состояний резервуара
│   └── forecasting.py          # One-step, fixed-horizon, closed-loop
├── readout/ridge.py            # RidgeReadout + select_alpha
├── hpo/
│   ├── tuner.py                # run_hpo() — Optuna TPE + MedianPruner
│   └── search_spaces.py        # Поисковые пространства для всех 7 архитектур
├── runners/
│   ├── experiment_runner.py    # Единый runner: reservoir → states → readout → metrics
│   ├── multi_seed.py           # run_multi_seed() → MultiSeedResult
│   └── pipeline.py             # run_pipeline(): HPO → multi-seed → ResultSpec
├── reporting/
│   ├── run_record.py           # RunRecord с reproducibility-метаданными
│   ├── report.py               # generate_report() → report.md + report.csv
│   └── plots.py                # plot_metric_bar(), plot_predictions()
├── main.py                     # FastAPI приложение
├── tasks.py                    # Celery worker
├── models.py                   # SQLAlchemy (User, Experiment, Result)
├── schemas.py                  # API-схемы
└── config.py                   # Settings (pydantic-settings)

tests/
├── test_metrics.py             # 36 детерминированных тестов метрик
├── test_data_provider.py       # 43 теста генераторов + golden tests
├── test_golden.py              # 8 bit-exact тестов (seed=42)
├── test_reservoirs.py          # 42 теста всех 7 архитектур
├── test_protocol.py            # 43 теста protocol layer
├── test_hpo.py                 # 35 тестов HPO + multi-seed
├── test_cli.py                 # 21 smoke-тест CLI
├── test_reporting.py           # 34 теста reporting + RunRecord
└── test_worker.py              # 20 тестов worker + artifact format
```

---

## Тесты

```bash
poetry run pytest tests/ --ignore=tests/test_flow.py -q
# => 287 passed
```

`test_flow.py` — интеграционный тест (требует PostgreSQL + Redis), запускается только в Docker.

---

## REST API

Поднять через Docker:

```bash
docker compose up --build
# API: http://localhost/
# Swagger: http://localhost/docs
```

**Регистрация и токен:**

```bash
curl -X POST http://localhost/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secret"}'

curl -X POST http://localhost/token \
  -d "username=user@example.com&password=secret"
# => {"access_token": "...", "token_type": "bearer"}
```

**Запуск эксперимента (воркер выполняет асинхронно):**

```bash
curl -X POST http://localhost/experiments/ \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": {"name": "narma10", "length": 2000},
    "reservoir": {"type": "esn", "params": {"units": 500}},
    "protocol": {"washout": 200},
    "seed": 42
  }'
```

**Результаты:**

```bash
# Статус + метрики
curl http://localhost/experiments/1

# Сравнение нескольких экспериментов (JSON или CSV)
curl "http://localhost/experiments/compare?ids=1,2,3&format=csv" \
  -H "Authorization: Bearer <TOKEN>"

# График предсказание vs реальность (PNG)
curl http://localhost/experiments/1/plot \
  -H "Authorization: Bearer <TOKEN>" --output plot.png
```

Статусы эксперимента: `QUEUED → RUNNING → COMPLETED / FAILED`.

---

## Воспроизводимость

Каждый `rcbench run --output run.json` сохраняет `RunRecord`:

```json
{
  "spec": { ... },
  "result": { "config_hash": "a3f1c7e2b9d40851", "metrics": { ... } },
  "timestamp": "2026-04-29T10:00:00+00:00",
  "git_hash": "d217b26",
  "hostname": "myhost",
  "python_version": "3.12.13",
  "rc_bench_version": "0.1.0"
}
```

`config_hash` — SHA-256 от полного `ExperimentSpec`. Один и тот же хэш гарантирует идентичные условия эксперимента.

---

## Источники

1. Jaeger H. (2001). The "echo state" approach to analysing and training recurrent neural networks. *GMD Report 148*.
2. Maass W., Natschläger T., Markram H. (2002). Real-time computing without stable states. *Neural Computation*.
3. Tanaka G. et al. (2019). Recent advances in physical reservoir computing. *Neural Networks*.
4. Kleyko D. et al. (2025). Principled neuromorphic reservoir computing. *arXiv*.

---

*Автор: Комаров Дмитрий*
