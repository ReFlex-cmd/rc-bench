# RC-Bench: Полный план рефакторинга

**Why:** Превратить rc-bench из набора отдельных сервисов в единый воспроизводимый benchmark-suite для reservoir computing с честным сравнением архитектур.

**How to apply:** Этапы выполняются последовательно в рамках приоритетных блоков P0 → P1 → P2. Каждый этап — логически завершённая PR-пачка.

---

## Текущее состояние (snapshot 2026-04-08)

### Что уже работает
- Микросервисная архитектура: FastAPI + Celery + PostgreSQL + Redis + Nginx (docker-compose)
- 4 reservoir-сервиса: ESN (reservoirpy), LSM (LIF numpy), FHN (Euler), Logistic Map
- Worker маршрутизирует все 4 типа через if/elif в tasks.py
- NARMA10 генератор в core/data_provider.py
- 3 метрики: mse, mae, nrmse (range-нормировка)
- 34 unit-теста (metrics: 13, data_provider: 9, reservoirs: 12)
- CLI: только команда `prepare narma10`
- Auth (JWT), CRUD experiments, compare endpoint, plot endpoint

### Ключевые проблемы
1. **Массивное дублирование**: select_alpha скопирована 4 раза, scale_inputs — 3 раза, washout — 4 раза, readout Ridge — 4 раза (~60% кода каждого reservoir service — одинаковая обвязка)
2. **Нет абстракции reservoir**: каждый — монолитная run_X_experiment(), нельзя переиспользовать reservoir отдельно от readout
3. **Нет pydantic-контрактов для core**: config — бестиповой Dict[str, Any], результат — ad hoc dict
4. **Только NARMA10**: data_provider.py → ValueError для любого другого dataset
5. **Нет protocol layer**: washout/split/forecasting mode зашиты ad hoc
6. **NRMSE без явного имени**: только range-нормировка, нет вариантов
7. **FHN на Euler**: нестабильная интеграция, нужен RK4
8. **LSM Python for-loop**: медленный поэлементный цикл
9. **CLI не умеет запускать эксперименты**: только prepare
10. **Нет multi-seed, нет HPO budget control, нет reproducibility manifest**
11. **БД схема зашита под фиксированные метрики**: колонки nrmse/mse/mae/val_nrmse

---

## Блок P0 — Фундамент

### Этап 1. Свести архитектуру к одному ядру — ЧАСТИЧНО ВЫПОЛНЕН

Статус: orchestrator/prepare_data.py уже использует rc_bench.core.data_provider. Worker поддерживает 4 типа. Но CLI = только `prepare`, нет runner.

**TODO:**
- [ ] Создать `rc_bench/runners/experiment_runner.py` — единый runner, вызываемый из CLI и Celery
- [ ] Создать `rc_bench/cli.py` (полноценный CLI): prepare, run, list-datasets, list-reservoirs, validate-spec, aggregate
- [ ] Мигрировать содержимое `orchestrator/` в rc_bench, deprecate orchestrator/
- [ ] Обновить `pyproject.toml` [project.scripts]: rcbench → rc_bench.cli:app
- [ ] Smoke test: `pytest -m smoke` — install + run ExperimentSpec (NARMA10+ESN) → valid ResultSpec JSON

### Этап 2. Формализовать контракты — НЕ ВЫПОЛНЕН

Текущее: schemas.py = только API-уровень. Core оперирует Dict[str, Any].

**TODO:**
- [ ] Создать `rc_bench/schema.py` с pydantic-моделями:
  - DatasetSpec(name, params)
  - ReservoirSpec(type, params) — с валидацией по type
  - ProtocolSpec(split, washout, forecasting_mode, multi_seed, hpo_budget)
  - ReadoutSpec(type, params)
  - ExperimentSpec = DatasetSpec + ReservoirSpec + ProtocolSpec + ReadoutSpec + seed_list
  - MetricsResult — именованные метрики
  - RunResult — per-seed (metrics, timings, memory)
  - ResultSpec — status, config_hash, per_seed_results, aggregated, timings, manifest, artifact_paths
- [ ] API schemas.py наследует от core schema.py
- [ ] Убрать фиксированные колонки метрик из models.py → JSONB ResultSpec

### Этап 3. Унифицировать datasets — НЕ ВЫПОЛНЕН (кроме NARMA10)

**TODO:**
- [ ] Создать `rc_bench/datasets/`:
  - `narma.py` — NARMA10 + NARMA30
  - `mackey_glass.py` — Mackey-Glass (DDE, tau=17, beta=0.2, gamma=0.1, n=10)
  - `lorenz.py` — Lorenz-63 (RK4, sigma=10, rho=28, beta=8/3)
  - `registry.py` — фабрика get_dataset(name, **params)
- [ ] Canonical parameters, split convention, washout convention для каждого
- [ ] Golden tests: фиксированный seed → первые K значений → assert exact match
- [ ] Stub-каркас: Kuramoto-Sivashinsky, spiking benchmark, real multivariate

### Этап 4. Унифицировать protocol layer — НЕ ВЫПОЛНЕН

**TODO:**
- [ ] Создать `rc_bench/protocol/`:
  - `splitter.py` — train/val/test contiguous split с washout exclusion
  - `forecasting.py` — 3 режима: one-step teacher forcing, fixed-horizon multi-step, closed-loop rollout
  - `state_collector.py` — state collection → readout fit → metric evaluation windows
- [ ] Protocol conformance tests: synthetic task где teacher forcing != closed-loop

### Этап 5. Унифицировать метрики — ЧАСТИЧНО ВЫПОЛНЕН

Есть: mse, mae, nrmse (range). 13 тестов.

**TODO:**
- [ ] Переименовать nrmse → nrmse_range
- [ ] Добавить: rmse, nrmse_std, nrmse_var
- [ ] Добавить: prediction_horizon (valid prediction time), trajectory_error
- [ ] Добавить cost metrics: train_time, inference_latency, throughput, peak_memory, energy_proxy
- [ ] Deterministic тесты на hand-crafted массивах для каждой метрики

### Этап 6. Единый reservoir интерфейс — НЕ ВЫПОЛНЕН

Текущее: 4 монолитных run_X_experiment() с ~60% дублированного кода.

**TODO:**
- [ ] ABC/Protocol `BaseReservoir`: reset_state(), transform(X) → H, fit_transform(X)
- [ ] Реализовать: ESNReservoir, LSMReservoir, LogisticReservoir, FHNReservoir
- [ ] Единый `rc_bench/readout/ridge.py` с select_alpha (вместо 4 копий)
- [ ] Единый `runners/experiment_runner.py` — reservoir + readout + protocol + metrics
- [ ] Убрать run_X_experiment() → заменить на runner.run(spec)
- [ ] Sanity checks: ESN (bounded states, spectral radius), LSM (spiking occurs), Logistic (chaotic regime), FHN (bounded oscillation)
- [ ] FHN: Euler → RK4
- [ ] LSM: рассмотреть numba.njit для lif_lsm_run

### Этап 8. Единый readout и HPO — НЕ ВЫПОЛНЕН

**TODO:**
- [ ] `rc_bench/readout/ridge.py` — единый readout модуль
- [ ] `rc_bench/readout/classifier.py` — stub для classification
- [ ] Запрет нелинейных readout в core benchmark
- [ ] Единый HPO budget: одинаковое число trials, split, val selection для всех архитектур
- [ ] Multi-seed: минимум 5 seeds, mean +/- std, опционально CI/bootstrap

---

## Блок P1 — Новый функционал

### Этап 7. Новые архитектуры — НЕ ВЫПОЛНЕН

**TODO:**
- [ ] **Leaky-ESN**: explicit leak parameter, HPO grid по leak rate, сравнение с vanilla ESN
- [ ] **DeepESN**: stacked reservoirs, concat-to-readout / last-layer-to-readout, per-layer leak/SR
- [ ] **QRC stub**: minimal simulated baseline, одна reference formulation, experimental/ модуль

### Этап 9. Reporting и reproducibility — НЕ ВЫПОЛНЕН

**TODO:**
- [ ] Каждый запуск сохраняет: full spec, resolved hyperparams, seed list, package versions, platform info, artifact manifest
- [ ] Агрегированные отчёты: per-seed table, aggregated table, best config, quality-vs-cost
- [ ] Export: JSON + CSV + markdown report
- [ ] Reproducibility command: rerun from saved manifest

### Этап 10. Обновить CLI и worker — ЧАСТИЧНО ВЫПОЛНЕН

**TODO:**
- [ ] CLI commands: list-datasets, list-reservoirs, validate-spec, run, run-multi-seed, aggregate
- [ ] Worker: принимает ExperimentSpec, маршрутизация через registry (не if/elif)
- [ ] Async lifecycle: pending → running → completed → failed → artifact paths

---

## Блок P2 — Polish

### Этап 11. Документация — ЧАСТИЧНО ВЫПОЛНЕН

**TODO:**
- [ ] Переписать README: architecture overview, все reservoir types, все datasets
- [ ] docs/benchmark_protocol.md — формальное описание протокола
- [ ] docs/validation.md — какие тесты что проверяют
- [ ] Примеры ExperimentSpec, reproducibility guide

### Этап 12. Acceptance suite — НЕ ВЫПОЛНЕН

**Критерии прохождения:**
- [ ] Один ExperimentSpec → CLI и worker без изменений логики
- [ ] Все 6 архитектур (ESN, Leaky-ESN, DeepESN, LSM, Logistic, FHN) через единый runner
- [ ] NARMA10 + NARMA30 + Mackey-Glass + Lorenz-63 реализованы и покрыты тестами
- [ ] Метрики из одного модуля с явными именами
- [ ] Multi-seed aggregation работает
- [ ] E2E benchmark report: NARMA10 + MG/Lorenz x ESN vs Leaky-ESN vs LSM vs Logistic vs FHN
- [ ] README соответствует реальности

---

## Рекомендуемый порядок реализации

1. Этап 6 (reservoir interface + deduplicate) — максимальный ROI
2. Этап 2 (pydantic contracts)
3. Этап 5 (metrics rename + extend)
4. Этап 1 remainder (CLI + runner + kill orchestrator)
5. Этап 4 (protocol layer)
6. Этап 8 (readout + multi-seed + HPO)
7. Этап 3 (NARMA30 + MG + Lorenz)
8. Этап 7 (Leaky-ESN + DeepESN)
9. Этап 9 (reporting)
10. Этап 10 (CLI/worker update)
11. Этап 11 (docs)
12. Этап 12 (acceptance)
