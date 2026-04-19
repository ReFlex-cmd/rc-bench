# Graph Report - .  (2026-04-17)

## Corpus Check
- Corpus is ~19,255 words - fits in a single context window. You may not need a graph.

## Summary
- 282 nodes · 415 edges · 26 communities detected
- Extraction: 71% EXTRACTED · 29% INFERRED · 0% AMBIGUOUS · INFERRED: 121 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Refactoring Plan & Architecture|Refactoring Plan & Architecture]]
- [[_COMMUNITY_Datasets & Protocol Design|Datasets & Protocol Design]]
- [[_COMMUNITY_ESN Service Implementation|ESN Service Implementation]]
- [[_COMMUNITY_Test Infrastructure|Test Infrastructure]]
- [[_COMMUNITY_CLI & Data Pipeline|CLI & Data Pipeline]]
- [[_COMMUNITY_FHN Reservoir (ODE-based)|FHN Reservoir (ODE-based)]]
- [[_COMMUNITY_API Models & Auth Schemas|API Models & Auth Schemas]]
- [[_COMMUNITY_FastAPI Endpoints|FastAPI Endpoints]]
- [[_COMMUNITY_LSM Reservoir|LSM Reservoir]]
- [[_COMMUNITY_Logistic Map Reservoir|Logistic Map Reservoir]]
- [[_COMMUNITY_NARMA10 Benchmark Results|NARMA10 Benchmark Results]]
- [[_COMMUNITY_ESN Prediction Visualization|ESN Prediction Visualization]]
- [[_COMMUNITY_App Configuration|App Configuration]]
- [[_COMMUNITY_DB Migration Initial Schema|DB Migration: Initial Schema]]
- [[_COMMUNITY_DB Migration User Model|DB Migration: User Model]]
- [[_COMMUNITY_DB Migration Metrics & Auth|DB Migration: Metrics & Auth]]
- [[_COMMUNITY_Integration Flow Test|Integration Flow Test]]
- [[_COMMUNITY_Metrics Documentation|Metrics Documentation]]
- [[_COMMUNITY_Orchestrator Package|Orchestrator Package]]
- [[_COMMUNITY_App Package Init|App Package Init]]
- [[_COMMUNITY_Celery Worker|Celery Worker]]
- [[_COMMUNITY_Core Package Init|Core Package Init]]
- [[_COMMUNITY_Reservoirs Package Init|Reservoirs Package Init]]
- [[_COMMUNITY_Orchestrator CLI Docs|Orchestrator CLI Docs]]
- [[_COMMUNITY_Data Preparation Docs|Data Preparation Docs]]
- [[_COMMUNITY_Refactoring Stage 11|Refactoring Stage 11]]

## God Nodes (most connected - your core abstractions)
1. `rc-bench project` - 20 edges
2. `nrmse()` - 16 edges
3. `ExperimentStatus` - 13 edges
4. `run_lsm_experiment()` - 13 edges
5. `run_fhn_experiment()` - 13 edges
6. `run_logistic_experiment()` - 13 edges
7. `Base` - 12 edges
8. `get_data_for_experiment()` - 11 edges
9. `run_esn_experiment()` - 11 edges
10. `Module: rc_bench/reservoirs` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Создаём все таблицы перед тестами и дропаем после.` --uses--> `Base`  [INFERRED]
  tests/conftest.py → src/rc_bench/database.py
- `Каждый тест получает сессию внутри транзакции, которая откатывается после теста.` --uses--> `Base`  [INFERRED]
  tests/conftest.py → src/rc_bench/database.py
- `HTTP-клиент с переопределённой зависимостью get_db — использует тестовую сессию.` --uses--> `Base`  [INFERRED]
  tests/conftest.py → src/rc_bench/database.py
- `# TODO: для запуска test_flow.py нужны: PostgreSQL, asyncpg, psycopg2-binary,` --uses--> `Base`  [INFERRED]
  tests/conftest.py → src/rc_bench/database.py
- `Typer CLI (rcbench)` --will_use--> `Target: unified ExperimentSpec`  [INFERRED]
  README.md → plan.md

## Communities

### Community 0 - "Refactoring Plan & Architecture"
Cohesion: 0.07
Nodes (40): Dataset: NARMA30, Module: rc_bench/reservoirs, New Architecture: DeepESN, New Architecture: Leaky-ESN, New Architecture: QRC (simulated stub), Priority P1: New functionality (NARMA30, MG, Lorenz-63, Leaky-ESN, DeepESN, cost metrics, reporting), Priority P2: Polish (KS, event benchmark, real multivariate, QRC stub, energy profiling), Refactoring Goal: unified reproducible benchmark-suite for RC (+32 more)

### Community 1 - "Datasets & Protocol Design"
Cohesion: 0.06
Nodes (40): Closed-Loop Rollout Protocol Mode, Dataset: Lorenz-63, Dataset: Mackey-Glass, Module: rc_bench/datasets, Module: rc_bench/metrics, Module: rc_bench/protocol, Module: rc_bench/readout, Module: rc_bench/reporting (+32 more)

### Community 2 - "ESN Service Implementation"
Cohesion: 0.11
Nodes (14): build_reservoir(), Создает резервуар, используя параметры из конфига., Подбор alpha для Ridge (как в твоем скрипте)., Основная точка входа.      Принимает данные и конфиг.     Возвращает словарь с м, run_esn_experiment(), select_alpha(), mae(), mse() (+6 more)

### Community 3 - "Test Infrastructure"
Cohesion: 0.11
Nodes (22): Base, client(), db_session(), Создаём все таблицы перед тестами и дропаем после., Каждый тест получает сессию внутри транзакции, которая откатывается после теста., HTTP-клиент с переопределённой зависимостью get_db — использует тестовую сессию., # TODO: для запуска test_flow.py нужны: PostgreSQL, asyncpg, psycopg2-binary,, _setup_db() (+14 more)

### Community 4 - "CLI & Data Pipeline"
Cohesion: 0.12
Nodes (10): prepare(), Подготовить данные для выбранной задачи (сейчас: NARMA10)., generate_narma10(), get_data_for_experiment(), Фабричный метод: получает имя датасета и возвращает готовые для обучения массивы, Генерация временного ряда NARMA10 (чистая математика)., prepare_narma10(), Генерация и сохранение данных для задачи NARMA10 в общем формате:       X_train (+2 more)

### Community 5 - "FHN Reservoir (ODE-based)"
Cohesion: 0.2
Nodes (10): fhn_reservoir_run(), fhn_step(), init_fhn_reservoir(), Точка входа для FHN-эксперимента., Один шаг интегрирования ФитцХью–Нагумо (Эйлер)., Запуск FHN-резервуара. Выход: H [T, units]., run_fhn_experiment(), scale_inputs() (+2 more)

### Community 6 - "API Models & Auth Schemas"
Cohesion: 0.33
Nodes (13): BaseModel, get_current_user(), ExperimentStatus, ExperimentBase, ExperimentCreate, ExperimentRead, Схема для создания эксперимента.     Содержит пример (example), который отобрази, ResultRead (+5 more)

### Community 7 - "FastAPI Endpoints"
Cohesion: 0.18
Nodes (7): login_for_access_token(), register_user(), create_access_token(), get_password_hash(), Проверяет, совпадает ли введенный пароль с хешем в БД., Генерирует хеш пароля., verify_password()

### Community 8 - "LSM Reservoir"
Cohesion: 0.26
Nodes (8): init_lsm_reservoir(), lif_lsm_run(), Точка входа для LSM-эксперимента., Запуск LIF-резервуара в стиле LSM., run_lsm_experiment(), scale_inputs(), select_alpha(), TestLSM

### Community 9 - "Logistic Map Reservoir"
Cohesion: 0.26
Nodes (8): init_logistic_params(), logistic_reservoir_run(), Дискретный хаотический резервуар на логистическом отображении., Точка входа для Logistic-эксперимента., run_logistic_experiment(), scale_inputs(), select_alpha(), TestLogistic

### Community 10 - "NARMA10 Benchmark Results"
Cohesion: 0.33
Nodes (12): Test NRMSE (Normalized Root Mean Square Error), NARMA10: test NRMSE by reservoir (bar chart), NARMA10 task (10th-order nonlinear autoregressive moving average), ESN NRMSE ~0.069 (lowest / best), FHN NRMSE ~0.164, Logistic NRMSE ~0.210 (highest / worst), LSM NRMSE ~0.171, RC-Bench reservoir computing benchmark suite (+4 more)

### Community 11 - "ESN Prediction Visualization"
Cohesion: 0.39
Nodes (9): Echo State Network (ESN), ESN on NARMA10: true vs predicted (first 500 steps), NARMA10 Task (Nonlinear Autoregressive Moving Average order 10), Output value range approximately [0.15, 0.75], Predicted output signal (ESN prediction, orange dashed line), Experiment run with random seed=42, Evaluation time horizon: 500 time steps, Good overall tracking quality with occasional peak underestimation (+1 more)

### Community 12 - "App Configuration"
Cohesion: 0.4
Nodes (2): BaseSettings, Settings

### Community 13 - "DB Migration: Initial Schema"
Cohesion: 0.5
Nodes (1): Initial tables  Revision ID: 4b02e7a79d21 Revises:  Create Date: 2026-01-19 05:4

### Community 14 - "DB Migration: User Model"
Cohesion: 0.5
Nodes (1): Add User model  Revision ID: a253bc84cc78 Revises: 4b02e7a79d21 Create Date: 202

### Community 15 - "DB Migration: Metrics & Auth"
Cohesion: 0.5
Nodes (1): Add mse, val_nrmse to results; make owner_id NOT NULL  Revision ID: c7f3a1d9e042

### Community 16 - "Integration Flow Test"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Metrics Documentation"
Cohesion: 1.0
Nodes (2): core/metrics.py, Metrics (NRMSE, MSE, MAE)

### Community 18 - "Orchestrator Package"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "App Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 20 - "Celery Worker"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Core Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Reservoirs Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Orchestrator CLI Docs"
Cohesion: 1.0
Nodes (1): orchestrator/cli.py

### Community 24 - "Data Preparation Docs"
Cohesion: 1.0
Nodes (1): orchestrator/prepare_data.py

### Community 25 - "Refactoring Stage 11"
Cohesion: 1.0
Nodes (1): Stage 11: documentation (README rewrite, docs/benchmark_protocol.md, docs/validation.md)

## Knowledge Gaps
- **72 isolated node(s):** `Initial tables  Revision ID: 4b02e7a79d21 Revises:  Create Date: 2026-01-19 05:4`, `Add User model  Revision ID: a253bc84cc78 Revises: 4b02e7a79d21 Create Date: 202`, `Add mse, val_nrmse to results; make owner_id NOT NULL  Revision ID: c7f3a1d9e042`, `Подготовить данные для выбранной задачи (сейчас: NARMA10).`, `Генерация и сохранение данных для задачи NARMA10 в общем формате:       X_train` (+67 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Integration Flow Test`** (2 nodes): `test_full_flow()`, `test_flow.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Metrics Documentation`** (2 nodes): `core/metrics.py`, `Metrics (NRMSE, MSE, MAE)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Orchestrator Package`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `App Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Celery Worker`** (1 nodes): `celery_app.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Core Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Reservoirs Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Orchestrator CLI Docs`** (1 nodes): `orchestrator/cli.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Preparation Docs`** (1 nodes): `orchestrator/prepare_data.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Refactoring Stage 11`** (1 nodes): `Stage 11: documentation (README rewrite, docs/benchmark_protocol.md, docs/validation.md)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_experiment_task()` connect `Test Infrastructure` to `ESN Service Implementation`, `CLI & Data Pipeline`, `FHN Reservoir (ODE-based)`, `LSM Reservoir`, `Logistic Map Reservoir`?**
  _High betweenness centrality (0.183) - this node is a cross-community bridge._
- **Why does `get_data_for_experiment()` connect `CLI & Data Pipeline` to `Test Infrastructure`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `nrmse()` (e.g. with `select_alpha()` and `run_esn_experiment()`) actually correct?**
  _`nrmse()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `ExperimentStatus` (e.g. with `Base` and `ExperimentBase`) actually correct?**
  _`ExperimentStatus` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `run_lsm_experiment()` (e.g. with `run_experiment_task()` and `nrmse()`) actually correct?**
  _`run_lsm_experiment()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `run_fhn_experiment()` (e.g. with `run_experiment_task()` and `nrmse()`) actually correct?**
  _`run_fhn_experiment()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Initial tables  Revision ID: 4b02e7a79d21 Revises:  Create Date: 2026-01-19 05:4`, `Add User model  Revision ID: a253bc84cc78 Revises: 4b02e7a79d21 Create Date: 202`, `Add mse, val_nrmse to results; make owner_id NOT NULL  Revision ID: c7f3a1d9e042` to the rest of the system?**
  _72 weakly-connected nodes found - possible documentation gaps or missing edges._