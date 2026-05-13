# RC-Bench — Benchmark Suite for Reservoir Computing

![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-289%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

Воспроизводимый benchmark-suite для систематического сравнения архитектур **reservoir computing** на единых датасетах, метриках и протоколе оценки. Реализован как ВКР.

Методологическая основа — шесть критериев Wringe, Trefzer & Stepney (2024, arXiv:2405.06561): равный HPO-бюджет, единый Ridge-readout, multi-seed контроль (≥5), единые разбиения, единые метрики с явной нормировкой, полное логирование (включая версии библиотек, hardware profile, HPO convergence).

---

## Что умеет

- **7 архитектур резервуаров** — от классического ESN до квантово-вдохновлённого QRC
- **4 датасета** — NARMA-10/30, Mackey-Glass, Lorenz-63
- **3 режима прогнозирования** — one-step, fixed-horizon, closed-loop (autonomous)
- **HPO (Optuna TPE + MedianPruner + feasibility-gate)** — автоматический подбор гиперпараметров с защитой от вырожденных конфигураций
- **Multi-seed** — оценка mean ± std (sample, ddof=1) по N запускам
- **Per-layer HPO для Deep ESN** — независимые `sr_l, leak_rate_l` на слой (Gallicchio & Micheli 2017)
- **RunRecord с полной reproducibility** — git hash, lib_versions, hardware_profile, HPO convergence trajectory, diagnostics
- **REST API** — FastAPI + Celery + PostgreSQL + Redis (Docker Compose)

---

## Архитектуры резервуаров

| Тип | Класс | Описание |
|-----|-------|----------|
| `esn` | `ESNReservoir` | Классический ESN (Jaeger-style): `x = tanh(W·x + W_in·u)`, плотный W_in, без leak |
| `leaky_esn` | `LeakyESNReservoir` | ESN с leak-rate: `x = (1-α)x + α·tanh(W·x + W_in·u)` |
| `deep_esn` | `DeepESNReservoir` | Стек ESN-слоёв; **per-layer** `sr_l, leak_rate_l` тюнятся независимо |
| `lsm` | `LSMReservoir` | Liquid State Machine, LIF + рефрактерный период, синаптический след как readout-feature |
| `fhn` | `FHNReservoir` | FitzHugh-Nagumo (RK4, dt=0.01 фиксирован, параметризация через ε) |
| `logistic` | `LogisticReservoir` | Логистические узлы с диверсифицированными `r_i ∈ [3.7, 3.99]`, аддитивный input |
| `qrc` | `QRCReservoir` | **Mean-field Ising surrogate** + virtual nodes (не unitary; см. `audit/notes_qrc.md`) |

Все архитектуры реализуют единый интерфейс `BaseReservoir`: `transform()`, `step()`, `warmup()`, `reset_state()`, `sanity_check()`.

---

## Датасеты

| Имя | Описание | Длина по умолчанию |
|-----|----------|--------------------|
| `narma10` | NARMA порядка 10 (canonical Atiya & Parlos 2000) | 2 000 |
| `narma30` | NARMA-30 со стабилизирующим β/3 — **отклонение от canonical**, см. docstring | 3 000 |
| `mackey_glass` | DDE Mackey-Glass (τ=17, хаотический режим) | 5 000 |
| `lorenz63` | Аттрактор Лоренца, x-компонента (RK4, dt=0.01, subsample=10) | 5 000 |

Для `lorenz63` рекомендуется `forecasting_mode: closed_loop` — autonomous prediction, основной интерес для главы 4 ВКР (см. Pathak et al. 2018, PRL).

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
    n_units: 500
    spectral_radius: 0.9
    rc_connectivity: 0.1

protocol:
  washout: 200
  train_frac: 0.6
  val_frac: 0.2
  forecasting_mode: one_step   # one_step | fixed_horizon | closed_loop
  use_hpo: true
  hpo_budget: 100              # default = 100 (per ТЗ §2)
  n_seeds: 10                  # default = 10 (per ТЗ §3)

readout:
  alpha_grid: [1.0]            # overridden when use_hpo=true (HPO tunes alpha)

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

Каждый run record (`.json`) содержит полный `ExperimentSpec`, `ResultSpec` и reproducibility-метаданные (см. §**Воспроизводимость**).

### Сравнение нескольких запусков

```bash
# Таблица в терминале
rcbench aggregate results/ --sort-by nrmse_range --top 5

# Полный отчёт: report.md + report.csv + comparison_nrmse_range.png
rcbench report results/ --output-dir report/ --metric nrmse_range
```

---

## Эмпирическая батарея (этап 4 ВКР)

В `scripts/` лежат три готовых пайплайна:

| Скрипт | Что делает | Время |
|---|---|---|
| `scripts/smoke_grid.py` | Smoke-тест: 5 trials × 2 seeds × 19 ячеек | ~1 минута |
| `scripts/full_grid.py` | Полный прогон: 100 trials × 10 seeds (P0/P1), 50 × 5 (P2/P3) | ~18 минут |
| `scripts/make_plots.py` | Все графики из `reports/runs/*.json` | ~10 секунд |

Приоритетная таблица ячеек (модель × задача) — см. `scripts/full_grid.py:CELLS`. После прогона:

```bash
python scripts/full_grid.py     # → reports/full_results.json + reports/runs/*.json
python scripts/make_plots.py    # → reports/plots/*.png (43 графика)
```

Сводка результатов — в `reports/empirical_summary.md`. Главные числа из последнего прогона (NRMSE_range, mean ± std, n_seeds=10):

| Модель | NARMA-10 | NARMA-30 | Mackey-Glass | Lorenz-63 (closed-loop, λ_max·t) |
|---|---|---|---|---|
| ESN | **0.048 ± 0.006** | **0.070 ± 0.006** | 0.0002 ± 0.0000 | 0.25 ± 0.04 |
| Leaky ESN | 0.069 ± 0.032 | 0.185 ± 0.191 | **0.0001 ± 0.0000** | **0.26 ± 0.03** |
| Deep ESN | 0.066 ± 0.013 | 0.085 ± 0.007 | **0.0001 ± 0.0000** | **0.27 ± 0.00** |
| LSM | 0.135 ± 0.007 | — | 0.036 ± 0.002 | — |
| FHN | 0.167 ± 0.015 | — | 0.218 ± 0.002 | — |
| Logistic | 0.198 ± 0.004 | — | 0.071 ± 0.002 | — |
| QRC* | 0.106 ± 0.005 | — | — | — |

\* Mean-field Ising-симуляция, не настоящая unitary эволюция; см. `audit/notes_qrc.md`.

---

## Метрики

Все метрики вычисляются в `rc_bench/core/metrics.py` и возвращаются в `MetricsResult`.

| Метрика | Описание |
|---------|----------|
| `nrmse_range` | RMSE / (max - min) — основная метрика точности и HPO-objective |
| `nrmse_std` | RMSE / std(y_true) |
| `nrmse_var` | MSE / var(y_true) — 1.0 у mean-predictor |
| `rmse`, `mae`, `mse` | Стандартные метрики ошибки |
| `prediction_horizon` | Шагов до превышения порога 0.3·std (для closed-loop / Lorenz) |
| `val_nrmse_range` | `nrmse_range` на валидационной выборке (метрика HPO) |
| `train_time`, `inference_latency`, `peak_memory` | Стоимость |

Multi-seed возвращает `MultiSeedResult` с `mean`, `std` (sample, ddof=1) и `metrics_per_seed` для построения boxplot.

---

## Протокол оценки

Единый протокол гарантирует честное сравнение:

1. **Split** — contiguous train/val/test 60/20/20 (без перемешивания)
2. **Washout** — первые N=200 шагов состояния резервуара отбрасываются на каждом сегменте
3. **Scaling** — z-score через `StandardScaler` для моделей с `DEFAULT_SCALER='zscore'` (LSM, FHN, Logistic). ESN-семейство получает raw input.
4. **HPO** — Optuna TPE на val_nrmse_range, MedianPruner + feasibility-gate (skip, если `H.std() < 1e-6`)
5. **Readout** — Ridge Regression, α тюнится внутри HPO в `[1e-6, 1e+2]`
6. **Final fit** — train + val
7. **Multi-seed** — N запусков с разными seed на лучших гиперпараметрах, mean ± std (ddof=1) на test

```
[-------- train (60%) --------][-- val (20%) --][-- test (20%) --]
[ washout ][ H_train used   ][ wash ][ H_val ][ wash ][ H_test ]
```

Mode multi-seed: HPO один раз → лучшие параметры → 10 seeds (variability **гиперпараметров** между seeds не оценивается; см. `audit/03_protocol.md` §3.3).

---

## Структура проекта

```
src/rc_bench/
├── cli/app.py                  # Typer CLI (list-datasets, run, report, aggregate, ...)
├── core/
│   ├── schema.py               # Pydantic-контракты (ExperimentSpec, MetricsResult, ResultSpec)
│   ├── metrics.py              # Все метрики качества
│   ├── data_provider.py        # Генераторы датасетов (raw, без scaling)
│   └── reservoirs/             # 7 реализаций + base/registry
├── protocol/
│   ├── splitter.py
│   ├── state_collector.py
│   └── forecasting.py          # one_step, fixed_horizon, closed_loop (off-by-one fix 2026-05-13)
├── readout/ridge.py            # RidgeReadout + select_alpha
├── hpo/
│   ├── tuner.py                # run_hpo() → HPOResult (best_params, score, convergence, diagnostics)
│   └── search_spaces.py        # 7 пространств + per-layer Deep ESN suggest
├── runners/
│   ├── experiment_runner.py    # reservoir → states → readout → metrics
│   ├── multi_seed.py           # run_multi_seed() (ddof=1)
│   └── pipeline.py             # run_pipeline(): HPO → multi-seed → ResultSpec
├── reporting/
│   ├── run_record.py           # RunRecord + lib_versions + hardware_profile
│   ├── report.py               # generate_report() → report.md + report.csv
│   └── plots.py                # plot_metric_bar(), plot_predictions()
├── main.py / tasks.py / models.py / schemas.py / config.py    # FastAPI + Celery + SQLAlchemy

scripts/
├── smoke_grid.py               # Stage 4.1: smoke (5 trials × 2 seeds × 19 cells)
├── full_grid.py                # Stage 4.2: full grid (100 × 10 для P0/P1, 50 × 5 для P2/P3)
├── rerun_lorenz.py             # Re-run только Lorenz cells (например, после fix forecasting)
└── make_plots.py               # 43 графика: NRMSE bars, HPO convergence, seed boxplots, Lorenz horizon

audit/                          # Этапы аудита 2026-05-13 (этапы 1-3 ТЗ)
├── 01_implementations.md       # Аудит 7 моделей (5H + 12M + 8L)
├── 02_hpo_spaces.md            # Аудит HPO-пространств (6H + 19M)
├── 03_protocol.md              # Аудит протокола (3H + 8M, 6 критериев Wringe)
├── 04_out_of_scope.md          # Замеченное вне scope
├── 05_open_questions.md        # Q1-Q4 с решениями пользователя
└── notes_qrc.md                # Зафиксированное ограничение QRC (mean-field, не unitary)

reports/                        # Этапы 4-5 ТЗ
├── empirical_summary.md        # Главная сводка для главы 4
├── project_state.md            # Состояние проекта, TODO для ВКР
├── section_4_3_skeleton.md     # Конспект раздела 4.3 «Реализованные модели»
├── full_results.json           # Агрегатная таблица всех ячеек
├── smoke_results.json
├── runs/<model>__<task>.json   # 19 RunRecord с полной метадатой
└── plots/                      # 43 графика для ВКР

tests/                          # 289 passed (без test_flow — нужна Postgres)
├── test_metrics.py             # 36 тестов метрик
├── test_data_provider.py       # 43 теста генераторов
├── test_golden.py              # 8 bit-exact тестов (seed=42)
├── test_reservoirs.py          # 42 теста всех 7 архитектур
├── test_protocol.py            # 43 теста protocol layer
├── test_hpo.py                 # 35 тестов HPO + multi-seed (включая convergence/diagnostics)
├── test_cli.py                 # 21 smoke-тест CLI
├── test_reporting.py           # 34 теста reporting + RunRecord
└── test_worker.py              # 20 тестов worker + artifact format
```

---

## Тесты

```bash
poetry run pytest tests/ --ignore=tests/test_flow.py -q
# => 289 passed
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
    "reservoir": {"type": "esn", "params": {"n_units": 500}},
    "protocol": {"washout": 200, "use_hpo": true, "hpo_budget": 100, "n_seeds": 10},
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

Каждый `rcbench run --output run.json` сохраняет полный `RunRecord`:

```json
{
  "spec": { ... },
  "result": {
    "config_hash": "a3f1c7e2b9d40851",
    "metrics": null,
    "multi_seed_result": { "mean": {...}, "std": {...}, "metrics_per_seed": [...] },
    "hpo_best_params": { "reservoir_params": {...}, "readout_alpha": 0.16 },
    "hpo_convergence": [0.45, 0.32, 0.18, 0.12, ...],
    "hpo_diagnostics": { "n_completed": 56, "n_pruned": 44, "best_trial_number": 38 }
  },
  "timestamp": "2026-05-13T12:34:56+00:00",
  "git_hash": "a1b2c3d",
  "hostname": "myhost",
  "python_version": "3.12.13",
  "rc_bench_version": "0.1.0",
  "lib_versions": { "numpy": "2.3.5", "scipy": "...", "scikit-learn": "1.7.2",
                    "optuna": "4.8.0", "reservoirpy": "0.4.1", ... },
  "hardware_profile": { "cpu_model": "...", "cpu_count": 16, "ram_total_gb": 32.0,
                        "platform": "Linux-...", "machine": "x86_64" }
}
```

`config_hash` — SHA-256 от полного `ExperimentSpec`. Один и тот же хэш гарантирует идентичные условия эксперимента. Все библиотечные версии и hardware-метаданные собираются автоматически на каждом прогоне.

---

## Известные ограничения (для главы 4 ВКР)

Подробно — в `reports/empirical_summary.md` §4 и `reports/project_state.md` §3.

1. **Lorenz-63 closed-loop horizon ≈ 0.26 λ_max·t** vs Pathak 2018 ~8 λ_max·t. Гэп объясним отсутствием curriculum learning / noise injection / multi-output (x, y, z) Lorenz / меньшим N. Это direction for future work, не дефект текущей системы.
2. **NARMA-30 со стабилизирующим β/3** — сознательное отклонение от Atiya & Parlos. Документировано в docstring `generate_narma30()`.
3. **QRC = mean-field Ising surrogate**, не unitary эволюция. В тексте ВКР называть «quantum-inspired reservoir».
4. **Multi-seed mode 1** (HPO один раз → 10 seeds на лучших). Variability гиперпараметров не оценивается.
5. **Размерности резервуара не унифицированы** (LSM=400, FHN=200 vs ESN/Leaky=300, Logistic=500). Причина — стоимость Python-цикла; зафиксировано как осознанный выбор.

---

## Источники

1. Jaeger H. (2001). The "echo state" approach to analysing and training recurrent neural networks. *GMD Report 148*.
2. Maass W., Natschläger T., Markram H. (2002). Real-time computing without stable states. *Neural Computation* 14(11).
3. FitzHugh R. (1961). Impulses and physiological states in theoretical models of nerve membrane. *Biophys J* 1(6).
4. Gallicchio C., Micheli A. (2017). Deep reservoir computing: A critical experimental analysis. *Neurocomputing* 268.
5. Fujii K., Nakajima K. (2017). Harnessing disordered-ensemble quantum dynamics for machine learning. *Phys. Rev. Applied* 8.
6. Pathak J. et al. (2018). Model-free prediction of large spatiotemporally chaotic systems from data. *PRL* 120.
7. Wringe C., Trefzer M., Stepney S. (2024). Reservoir computing benchmarks: a tutorial review and critique. *arXiv:2405.06561* — методологическая основа протокола.
8. Tanaka G. et al. (2019). Recent advances in physical reservoir computing. *Neural Networks* 115.

---

*Автор: Комаров Дмитрий*
