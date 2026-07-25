# Session handoff — 2026-07-25 (Claude session 2)

> **Начни следующую сессию с чтения этого файла, затем `docs/agent/` (PROJECT_CONTRACT, WORK_PLAN, BACKLOG, DECISIONS, AGENT_WORKFLOW).**

## Точка продолжения

- Ветка: `dev`, HEAD `0aad2fa` (+ этот docs-коммит поверх). `origin/dev` отстаёт (**ничего не запушено**).
- Рабочее дерево чистое после docs-коммита.
- Последний полный `bash scripts/verify.sh quick`: **427 passed, 1 deselected** (зелёный).
- Активных сабагентов и фоновых процессов нет. PROF-сабагент завершён и интегрирован; его worktree удалён.

## Что сделано в этой сессии (поверх `97d1537`, 15 коммитов)

Порядок next-steps #1–#4 + пред-условия + PROF + EXP-001 + reservoir-метрики:

- `b5ae127` schema WIP доведён (BaselineSpec, exactly-one, raw_sha256, selection_metric/seasonal_period, опц. JMLC-метрики, SelectionResult/EvaluationContext, ResultSpec traceability); legacy config_hash закреплён.
- `f247481` **ESN split isolation** (был blocker): `transform()` reset-ит reservoirpy state; regression `TestBatchTransformStatelessness`.
- `5c38ab6` MET-001: `seasonal_naive_mae_scale`, `mase`, `mae_skill` + edge-cases.
- `7f33fae` BASE-001 предикторы; `bcf8865` BASE-002 Ridge AR (+ **DEC-015**); `0866fe8` `run_baseline` (общий target-alignment, замок DEC-013).
- `d6f5bfc` **baseline dispatch + VSLICE-001**: `run_pipeline` → `_run_baseline_pipeline`; reservoir ResultSpec тоже с model_family/deterministic/evaluated_seeds/selection.
- `111c12d` reporting baseline-aware (`spec.model_type`); `873beac` reservoir alpha/HPO по `selection_metric` (DEC-013; argmin инвариантен, но `val_nrmse_std` теперь заполняется).
- `57fceaf` **PROF-001/002** (cherry-pick сабагента): `src/rc_bench/profiling/{latency,memory,hardware}.py` — generic utilities, 14 тестов. Санитизация DEC-008.
- `a87bceb` EXP-001: `runners/jmlc_matrix.py` + `scripts/run_jmlc_matrix.py` + `configs/jmlc/smoke.yaml`; `0aad2fa` фикс загрузки UCI-данных в раннере.
- `260b724` **reservoir MASE/MAE-skill**: `run_experiment` считает mase/mae_skill (при `seasonal_period` + fixed_horizon) на том же basis, что baselines → полный паритет fair-таблицы.

**DONE:** BOOT-001/002, COR-001/002/003, TEST-001, CI-001, DATA-001/002/003, ENERGY-001, EDA-001, DOC-002, BASE-001, BASE-002, MET-001, VSLICE-001, PROF-001, PROF-002, EXP-001.

## Проверено на реальных данных (проба, не закоммичена — вывод в scratchpad)

`run_matrix` на реальном UCI-окне, h=1: все cells `completed`, RunRecords+артефакты пишутся. Числа осмысленны:
persistence NRMSE_std=0.768 (skill 0.354), seasonal NRMSE_std=1.123 (self-skill **0.000** ✓), ridge_ar 0.676 (skill 0.388, лучший baseline), esn 0.704 (skill 0.354), leaky_esn 0.689 (skill 0.382). ~2 с/reservoir-cell (HPO 2 trials).

## Следующие шаги (по порядку)

1. **EXP-002 — полный smoke (14 cells).** Запустить:
   ```bash
   poetry run python scripts/run_jmlc_matrix.py \
     --config configs/jmlc/smoke.yaml --output reports/jmlc_2026/smoke
   ```
   Проверить: все 14 cells `completed`, отсутствие test-leakage, schema/артефакты, время. LSM может быть медленнее — при необходимости убавить reservoir units в smoke.yaml. Проба показала, что путь рабочий; полный прогон ещё НЕ делался.
2. **Wire PROF в evidence (P0, нужно для release gate).** Утилиты `rc_bench.profiling` пока НИГДЕ не вызываются. Нужен отдельный профилировочный проход (НЕ во время train — latency нельзя мешать с train-временем): для каждой модели построить inference-step callable → `measure_latency` (p50/p95/throughput), `measure_peak_rss_subprocess`, `serialized_model_bytes`, `working_state_bytes`; и один раз `get_hardware_profile()` → записать `reports/jmlc_2026/hardware_profile.json`. Это отдельный `scripts/`-скрипт или расширение матрицы отдельной фазой.
3. **EXP-003 — fair matrix (5 seeds, 20 trials).** Требует РЕАЛЬНОГО compute-времени → **согласовать тайминг с пользователем перед запуском**. Конфиг: копия smoke.yaml с `n_seeds: 5`, `hpo_budget: 20` (напр. `configs/jmlc/fair.yaml`). При нехватке времени — одинаково уменьшить до 10 trials (DEC-004); 5 seeds/оба горизонта/baselines НЕ сокращать.
4. EVID-001 (aggregation, traceability, `scripts/validate_evidence.py`) → PLOT-001 (Pareto quality-latency/quality-memory) → DOC-001 → DEMO-001 → PRES-001 → REL-001 (PR `dev → main`, без merge/force).

## Известные pending-элементы (обнаружены, НЕ сделаны)

- **`reports/jmlc_2026/dataset_manifest.json` отсутствует** — `verify.sh smoke` и `release` его требуют (manifest сейчас только в `configs/jmlc/dataset_manifest.json`). Скопировать/сгенерировать в evidence-каталог.
- **`scripts/validate_evidence.py` отсутствует** — требуется `verify.sh release` (EVID-001).
- **PROF не подключён** (см. шаг 2).
- **`main.py` REST API** (`create_experiment`) всё ещё завязан на `reservoir.type` — вне JMLC-пути, трек API-001 (P2).
- Reservoir alpha/HPO-селекция: argmin инвариантен к range/std (для фикс. val-набора обе = RMSE/const), поэтому DEC-013 «reservoir HPO по NRMSE_std» не меняет выбор конфигов — но `val_nrmse_std` и HPO-score теперь в headline-единицах.

## Ключевые команды

- Gate: `bash scripts/verify.sh quick` (unit, не-integration) — держать зелёным перед каждым commit.
- `verify.sh smoke` / `release` — см. `scripts/verify.sh` (нужны недостающие evidence-файлы выше).
- Матрица: `scripts/run_jmlc_matrix.py --config <yaml> --output <dir>`; `run_matrix(...)` в `rc_bench.runners.jmlc_matrix` принимает `cells=`/`horizons=` для подвыборки.
- Данные: raw UCI локально в `data/raw/` (gitignored); `get_data_for_experiment("uci_household_power", length=12000, train_frac=0.6, val_frac=0.2)`.

## Правила работы (подтверждено пользователем)

- Гибрид: сабагенты на тяжёлых независимых задачах (PROF-подобные, plots), cross-cutting (schema/pipeline/runners/backlog/decisions) — основной агент. Управление через скилл `subagent-driven-development`. **Замечание по worktree:** worktree сабагента может создаться от устаревшего базового коммита — интегрировать через `git cherry-pick <commit>` на актуальный `dev`, не merge; после — `git worktree remove --force` + `git branch -D`.
- Идти автономно по цепочке next-steps до блокера. Коммитить логическими единицами с зелёным gate; НЕ push без запроса; `main` напрямую не менять; force push запрещён.
- `gh auth status` валиден (`ReFlex-cmd`, scopes repo+workflow).

## Решения

DEC-001..015 в `docs/agent/DECISIONS.md`. Свежие/ключевые для baseline: DEC-012/013 (единый target set, per-split washout 200, fixed_horizon для UCI включая h=1, MASE train-only lag-24, MAE skill vs seasonal), DEC-014 (`BaselineSpec` deterministic, selection none/fixed_grid), **DEC-015** (Ridge AR: train-only scaler, alpha по val NRMSE_std, рефит train+val).
