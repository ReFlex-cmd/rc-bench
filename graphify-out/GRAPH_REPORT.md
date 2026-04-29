# Graph Report - .  (2026-04-29)

## Corpus Check
- 67 files · ~32,466 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 838 nodes · 1847 edges · 37 communities detected
- Extraction: 53% EXTRACTED · 47% INFERRED · 0% AMBIGUOUS · INFERRED: 874 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Reservoir Abstractions & Registry|Reservoir Abstractions & Registry]]
- [[_COMMUNITY_CLI Command Layer|CLI Command Layer]]
- [[_COMMUNITY_Benchmark Plan & Architecture|Benchmark Plan & Architecture]]
- [[_COMMUNITY_Reservoir Build & Transform|Reservoir Build & Transform]]
- [[_COMMUNITY_Dataset Generation|Dataset Generation]]
- [[_COMMUNITY_Experiment Execution|Experiment Execution]]
- [[_COMMUNITY_API & Test Infrastructure|API & Test Infrastructure]]
- [[_COMMUNITY_Config & Pipeline Settings|Config & Pipeline Settings]]
- [[_COMMUNITY_Metrics Module|Metrics Module]]
- [[_COMMUNITY_HPO & Multi-Seed|HPO & Multi-Seed]]
- [[_COMMUNITY_CLI Test Suite|CLI Test Suite]]
- [[_COMMUNITY_FastAPI REST API|FastAPI REST API]]
- [[_COMMUNITY_Visualization Outputs|Visualization Outputs]]
- [[_COMMUNITY_Prediction Plots|Prediction Plots]]
- [[_COMMUNITY_Reservoir State Interface|Reservoir State Interface]]
- [[_COMMUNITY_DB Migration Initial|DB Migration: Initial]]
- [[_COMMUNITY_DB Migration User Model|DB Migration: User Model]]
- [[_COMMUNITY_DB Migration Metrics Schema|DB Migration: Metrics Schema]]
- [[_COMMUNITY_DB Migration JSONB Results|DB Migration: JSONB Results]]
- [[_COMMUNITY_Integration Test Flow|Integration Test Flow]]
- [[_COMMUNITY_README Metrics Docs|README Metrics Docs]]
- [[_COMMUNITY_Orchestrator (Deprecated)|Orchestrator (Deprecated)]]
- [[_COMMUNITY_RC-Bench Package Init|RC-Bench Package Init]]
- [[_COMMUNITY_Celery App|Celery App]]
- [[_COMMUNITY_Core Package Init|Core Package Init]]
- [[_COMMUNITY_Reservoirs Package Init|Reservoirs Package Init]]
- [[_COMMUNITY_Orchestrator CLI Docs|Orchestrator CLI Docs]]
- [[_COMMUNITY_Orchestrator Prepare Docs|Orchestrator Prepare Docs]]
- [[_COMMUNITY_Refactoring Plan Stage 11|Refactoring Plan Stage 11]]
- [[_COMMUNITY_BaseReservoir Rationale A|BaseReservoir Rationale A]]
- [[_COMMUNITY_BaseReservoir Rationale B|BaseReservoir Rationale B]]
- [[_COMMUNITY_Readout Package Init|Readout Package Init]]
- [[_COMMUNITY_Runners Package Init|Runners Package Init]]
- [[_COMMUNITY_CLI Package Init|CLI Package Init]]
- [[_COMMUNITY_Protocol Package Init|Protocol Package Init]]
- [[_COMMUNITY_HPO Package Init|HPO Package Init]]
- [[_COMMUNITY_Reporting Package Init|Reporting Package Init]]

## God Nodes (most connected - your core abstractions)
1. `ExperimentSpec` - 54 edges
2. `ReadoutSpec` - 44 edges
3. `ReservoirSpec` - 40 edges
4. `DatasetSpec` - 39 edges
5. `ProtocolSpec` - 39 edges
6. `run_pipeline()` - 33 edges
7. `Graph Report` - 33 edges
8. `MetricsResult` - 31 edges
9. `ESNReservoir` - 31 edges
10. `FHNReservoir` - 30 edges

## Surprising Connections (you probably didn't know these)
- `_lsm()` --calls--> `LSMReservoir`  [INFERRED]
  tests/test_protocol.py → src/rc_bench/core/reservoirs/lsm_service.py
- `_logistic()` --calls--> `LogisticReservoir`  [INFERRED]
  tests/test_protocol.py → src/rc_bench/core/reservoirs/logistic_service.py
- `Tests for Stage 10: CLI/worker update.  Covers: - _load_predictions_from_artifac` --uses--> `Settings`  [INFERRED]
  tests/test_worker.py → src/rc_bench/config.py
- `Unit tests for main._load_predictions_from_artifact (no DB/config needed).` --uses--> `Settings`  [INFERRED]
  tests/test_worker.py → src/rc_bench/config.py
- `Verify that pipeline's npz output is readable by the plot endpoint helper.` --uses--> `Settings`  [INFERRED]
  tests/test_worker.py → src/rc_bench/config.py

## Hyperedges (group relationships)
- **run_experiment_task bridge across experiment stack** — run_experiment_task, esn_service_implementation, cli_data_pipeline, fhn_reservoir_ode_based, lsm_reservoir, logistic_map_reservoir, test_infrastructure [INFERRED 0.80]
- **Data access bridge around get_data_for_experiment** — get_data_for_experiment, cli_data_pipeline, test_infrastructure [INFERRED 0.75]
- **NARMA10 benchmark evaluation cycle** — narma10_benchmark_results, esn_prediction_visualization, nrmse [INFERRED 0.80]

## Communities

### Community 0 - "Reservoir Abstractions & Registry"
Cohesion: 0.07
Nodes (69): BaseReservoir, Return dict of {check_name: passed}. Override in subclasses., BaseModel, BaseReservoir, DeepESNReservoir, Deep Echo State Network: stacked leaky ESN layers (pure numpy).      Layer 0 rec, ESNReservoir, Run one step; reservoirpy step() expects a 1-D array. (+61 more)

### Community 1 - "CLI Command Layer"
Cohesion: 0.05
Nodes (47): aggregate(), list_datasets(), list_reservoirs(), _load_spec_file(), _metrics_table(), _print_result(), rcbench — RC-Bench command-line interface.  Commands -------- list-datasets, Print all registered reservoir types. (+39 more)

### Community 2 - "Benchmark Plan & Architecture"
Cohesion: 0.03
Nodes (80): Closed-Loop Rollout Protocol Mode, Dataset: Lorenz-63, Dataset: Mackey-Glass, Dataset: NARMA30, Module: rc_bench/datasets, Module: rc_bench/metrics, Module: rc_bench/protocol, Module: rc_bench/readout (+72 more)

### Community 3 - "Reservoir Build & Transform"
Cohesion: 0.04
Nodes (41): ABC, _build(), build_reservoir(), Создает резервуар, используя параметры из конфига., Подбор alpha для Ridge (как в твоем скрипте)., Основная точка входа.      Принимает данные и конфиг.     Возвращает словарь с м, run_esn_experiment(), select_alpha() (+33 more)

### Community 4 - "Dataset Generation"
Cohesion: 0.04
Nodes (28): prepare(), Подготовить данные для выбранной задачи (сейчас: NARMA10)., generate_lorenz63(), generate_mackey_glass(), generate_narma10(), generate_narma30(), get_data_for_experiment(), Lorenz-63 attractor via RK4.      Integration with fine ``dt``, subsampled by `` (+20 more)

### Community 5 - "Experiment Execution"
Cohesion: 0.05
Nodes (27): Reset reservoirpy node to its initial (zero) state.          reservoirpy initial, Use reservoirpy's native batch run for efficient warmup., _check_sanity(), run_experiment(), _scale_inputs(), build_fixed_horizon_targets(), build_one_step_targets(), closed_loop_predict() (+19 more)

### Community 6 - "API & Test Infrastructure"
Cohesion: 0.05
Nodes (46): API Models & Auth Schemas, App Configuration, Base, CLI & Data Pipeline, client(), db_session(), Создаём все таблицы перед тестами и дропаем после., Каждый тест получает сессию внутри транзакции, которая откатывается после теста. (+38 more)

### Community 7 - "Config & Pipeline Settings"
Cohesion: 0.06
Nodes (21): BaseSettings, Settings, Unified experiment pipeline: HPO → multi-seed → ResultSpec.  Both CLI ``rcbench, Execute the full pipeline described by *spec*.      Steps     -----     1. Optio, Execute the full pipeline described by *spec*.      Steps     -----     1. Optio, run_pipeline(), TestPipeline, TestPipelineArtifacts (+13 more)

### Community 8 - "Metrics Module"
Cohesion: 0.07
Nodes (16): mse(), nrmse_range(), nrmse_std(), nrmse_var(), prediction_horizon(), RMSE / (max - min) of y_true., MSE / var(y_true). Equals 1.0 for a constant (mean) predictor., Steps until |y_true[t] - y_pred[t]| / std(y_true) first exceeds threshold. (+8 more)

### Community 9 - "HPO & Multi-Seed"
Cohesion: 0.08
Nodes (19): Run the experiment ``n_seeds`` times, varying only the reservoir seed.      The, run_multi_seed(), from_metrics_list(), MultiSeedResult, params_from_trial(), Per-reservoir Optuna search spaces.  Each entry defines the hyperparameters to t, Reconstruct the structured params dict from a completed trial's flat param dict., Ask ``trial`` to suggest all hyperparameters for ``reservoir_type``.      Return (+11 more)

### Community 10 - "CLI Test Suite"
Cohesion: 0.11
Nodes (8): str, Smoke tests for the rcbench CLI. Uses Typer's CliRunner — no DB, no Celery, pure, Run n experiments and write their records to tmp_path., TestAggregate, TestListDatasets, TestListReservoirs, TestRun, TestValidateSpec

### Community 11 - "FastAPI REST API"
Cohesion: 0.1
Nodes (16): compare_experiments(), get_current_user(), _load_predictions_from_artifact(), login_for_access_token(), plot_experiment(), Load (y_test, y_pred) from the predictions.npz artifact saved by run_pipeline., register_user(), create_access_token() (+8 more)

### Community 12 - "Visualization Outputs"
Cohesion: 0.33
Nodes (12): Test NRMSE (Normalized Root Mean Square Error), NARMA10: test NRMSE by reservoir (bar chart), NARMA10 task (10th-order nonlinear autoregressive moving average), ESN NRMSE ~0.069 (lowest / best), FHN NRMSE ~0.164, Logistic NRMSE ~0.210 (highest / worst), LSM NRMSE ~0.171, RC-Bench reservoir computing benchmark suite (+4 more)

### Community 13 - "Prediction Plots"
Cohesion: 0.39
Nodes (9): Echo State Network (ESN), ESN on NARMA10: true vs predicted (first 500 steps), NARMA10 Task (Nonlinear Autoregressive Moving Average order 10), Output value range approximately [0.15, 0.75], Predicted output signal (ESN prediction, orange dashed line), Experiment run with random seed=42, Evaluation time horizon: 500 time steps, Good overall tracking quality with occasional peak underestimation (+1 more)

### Community 14 - "Reservoir State Interface"
Cohesion: 0.33
Nodes (3): Reset the step-state to initial conditions.          Must be called before the f, Advance the reservoir one step and return the state vector.          ``x_t`` : s, Warm up the reservoir step-state on ``X``; return H matrix.          Calls ``res

### Community 15 - "DB Migration: Initial"
Cohesion: 0.5
Nodes (1): Initial tables  Revision ID: 4b02e7a79d21 Revises:  Create Date: 2026-01-19 05:4

### Community 16 - "DB Migration: User Model"
Cohesion: 0.5
Nodes (1): Add User model  Revision ID: a253bc84cc78 Revises: 4b02e7a79d21 Create Date: 202

### Community 17 - "DB Migration: Metrics Schema"
Cohesion: 0.5
Nodes (1): Add mse, val_nrmse to results; make owner_id NOT NULL  Revision ID: c7f3a1d9e042

### Community 18 - "DB Migration: JSONB Results"
Cohesion: 0.5
Nodes (1): Replace fixed metric columns with result_data JSONB  Revision ID: d4e8f1a2b3c5 R

### Community 19 - "Integration Test Flow"
Cohesion: 1.0
Nodes (0): 

### Community 20 - "README Metrics Docs"
Cohesion: 1.0
Nodes (2): core/metrics.py, Metrics (NRMSE, MSE, MAE)

### Community 21 - "Orchestrator (Deprecated)"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "RC-Bench Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Celery App"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Core Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Reservoirs Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Orchestrator CLI Docs"
Cohesion: 1.0
Nodes (1): orchestrator/cli.py

### Community 27 - "Orchestrator Prepare Docs"
Cohesion: 1.0
Nodes (1): orchestrator/prepare_data.py

### Community 28 - "Refactoring Plan Stage 11"
Cohesion: 1.0
Nodes (1): Stage 11: documentation (README rewrite, docs/benchmark_protocol.md, docs/validation.md)

### Community 29 - "BaseReservoir Rationale A"
Cohesion: 1.0
Nodes (1): Initialize reservoir internals from config dict.

### Community 30 - "BaseReservoir Rationale B"
Cohesion: 1.0
Nodes (1): Map input sequence X [T] or [T,1] to reservoir states H [T, units].          Alw

### Community 31 - "Readout Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 32 - "Runners Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "CLI Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "Protocol Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "HPO Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Reporting Package Init"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **118 isolated node(s):** `Initial tables  Revision ID: 4b02e7a79d21 Revises:  Create Date: 2026-01-19 05:4`, `Add User model  Revision ID: a253bc84cc78 Revises: 4b02e7a79d21 Create Date: 202`, `Add mse, val_nrmse to results; make owner_id NOT NULL  Revision ID: c7f3a1d9e042`, `Подготовить данные для выбранной задачи (сейчас: NARMA10).`, `Генерация и сохранение данных для задачи NARMA10 в общем формате:       X_train` (+113 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Integration Test Flow`** (2 nodes): `test_full_flow()`, `test_flow.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `README Metrics Docs`** (2 nodes): `core/metrics.py`, `Metrics (NRMSE, MSE, MAE)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Orchestrator (Deprecated)`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `RC-Bench Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Celery App`** (1 nodes): `celery_app.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Core Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Reservoirs Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Orchestrator CLI Docs`** (1 nodes): `orchestrator/cli.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Orchestrator Prepare Docs`** (1 nodes): `orchestrator/prepare_data.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Refactoring Plan Stage 11`** (1 nodes): `Stage 11: documentation (README rewrite, docs/benchmark_protocol.md, docs/validation.md)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `BaseReservoir Rationale A`** (1 nodes): `Initialize reservoir internals from config dict.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `BaseReservoir Rationale B`** (1 nodes): `Map input sequence X [T] or [T,1] to reservoir states H [T, units].          Alw`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Readout Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Runners Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CLI Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Protocol Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `HPO Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Reporting Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_pipeline()` connect `Config & Pipeline Settings` to `Reservoir Abstractions & Registry`, `CLI Command Layer`, `Reservoir Build & Transform`, `Experiment Execution`, `HPO & Multi-Seed`?**
  _High betweenness centrality (0.187) - this node is a cross-community bridge._
- **Why does `run_experiment_task()` connect `Reservoir Build & Transform` to `Reservoir Abstractions & Registry`, `Dataset Generation`, `API & Test Infrastructure`, `Config & Pipeline Settings`, `CLI Test Suite`?**
  _High betweenness centrality (0.181) - this node is a cross-community bridge._
- **Why does `get_data_for_experiment()` connect `Dataset Generation` to `CLI Command Layer`, `Reservoir Build & Transform`, `Experiment Execution`?**
  _High betweenness centrality (0.141) - this node is a cross-community bridge._
- **Are the 51 inferred relationships involving `ExperimentSpec` (e.g. with `ExperimentCreate` and `ResultRead`) actually correct?**
  _`ExperimentSpec` has 51 INFERRED edges - model-reasoned connections that need verification._
- **Are the 42 inferred relationships involving `ReadoutSpec` (e.g. with `ExperimentCreate` and `ResultRead`) actually correct?**
  _`ReadoutSpec` has 42 INFERRED edges - model-reasoned connections that need verification._
- **Are the 38 inferred relationships involving `ReservoirSpec` (e.g. with `ExperimentCreate` and `ResultRead`) actually correct?**
  _`ReservoirSpec` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `DatasetSpec` (e.g. with `ExperimentCreate` and `ResultRead`) actually correct?**
  _`DatasetSpec` has 37 INFERRED edges - model-reasoned connections that need verification._