# Session handoff — 2026-07-24 (updated, Claude session)

## Точка продолжения

- Ветка: `dev`, HEAD `0866fe8`; `origin/dev` отстаёт (не запушено).
- Рабочее дерево чистое, кроме незакоммиченного этого файла и (возможно) memory.
- Активных сабагентов и фоновых pytest нет.
- Последний полный quick gate: **402 passed, 1 deselected**.

## Сделано в этой сессии (поверх `97d1537`)

- `b5ae127` schema WIP доведён и закоммичен: `BaselineSpec`, exactly-one reservoir|baseline, `raw_sha256`, `selection_metric`/`seasonal_period`, опциональные JMLC-метрики (all-or-none aggregation), `SelectionResult`/`EvaluationContext`, traceability-поля `ResultSpec`; legacy config_hash закреплён.
- `f247481` **ESN split isolation исправлен** (blocker закрыт): `ESNReservoir.transform()` теперь reset-ит reservoirpy state. Regression `TestBatchTransformStatelessness`. Deep ESN/leaky/lsm/logistic уже stateless — подтверждено.
- `5c38ab6` MET-001: `seasonal_naive_mae_scale`, `mase`, `mae_skill` + edge-case тесты.
- `7f33fae` BASE-001 предикторы: `persistence_forecast`, `seasonal_persistence_forecast`, `ar_lag_features`.
- `bcf8865` BASE-002: `select_and_fit_ridge_ar` (train-only scaler, alpha по val NRMSE_std, рефит train+val). Записан **DEC-015**.
- `0866fe8` baseline runner: `run_baseline(data, spec)` — общий с reservoir target-alignment (`aligned_target_positions`/`target_offset`), MASE/MAE-skill, selection/evaluation metadata. Тест-замок DEC-013: baseline и reservoir дают идентичный наблюдаемый test-target set.

BOOT-001/002, COR-001/002/003, TEST-001, CI-001, DATA-001/002/003, ENERGY-001, EDA-001, DOC-002, **BASE-001, BASE-002, MET-001** — DONE.

## Следующие шаги

1. **Baseline dispatch + VSLICE-001**: подключить `run_baseline` в pipeline (`runners/pipeline.py`), собрать `ResultSpec`/`RunRecord` для `BaselineSpec` (model_family/deterministic/evaluated_seeds/selection/evaluation), диспетчеризация reservoir vs baseline. Затем end-to-end horizon-1 persistence + ESN smoke на реальных данных.
   - Прежде чем менять pipeline, изучить, как reservoir-путь строит `ResultSpec`/`RunRecord` (`runners/pipeline.py`, `reporting/run_record.py`), и где HPO/multi-seed развилка.
   - `run_baseline` уже возвращает готовые `metrics`/`selection`/`evaluation`/`preds`/`y_test` + `model_family="baseline"`, `deterministic=True`, `evaluated_seeds=[]`.
2. PROF-001/PROF-002 (latency p50/p95, isolated RSS/model bytes/state bytes) — хорошие кандидаты в сабагенты (heavy independent).
3. EXP-001/002 smoke matrix 14 cells → EXP-003 fair matrix (5 seeds, 20 trials) — требует compute.
4. EVID-001 → PLOT-001 → DOC-001 → DEMO-001 → PRES-001 → REL-001.

## Данные и окружение

- Raw UCI уже локально: `data/raw/household_power_consumption.txt` + `uci_235.zip` (gitignored). Manifest `configs/jmlc/dataset_manifest.json`.
- Окно `2006-12-16T17`—`2008-04-29T16`; observed/imputed `11935/65`; splits `7200/2400/2400`.
- Target counts (washout=200): h=1 train/val/test `6934/2199/2199`; h=24 `6911/2176/2176`.
- poetry 2.3.1; `bash scripts/verify.sh quick` — зелёный.

## Правила работы (подтверждено пользователем)

- Гибрид: сабагенты на тяжёлых независимых задачах; cross-cutting (schema/pipeline/backlog/decisions) — основной агент. Управление через скилл `subagent-driven-development`.
- Идти автономно по цепочке next-steps до блокера.
- Force push запрещён; `main` напрямую не менять; push только по запросу.

## Зафиксированные решения

- DEC-001..015. Ключевые для baseline: DEC-012/013 (единый target set, per-split washout 200, fixed_horizon для UCI включая h=1), DEC-014 (`BaselineSpec`, deterministic, selection none/fixed_grid), DEC-015 (Ridge AR: train-only scaler, alpha по val NRMSE_std, рефит train+val).

## Публикация

- `gh auth status` теперь валиден (`ReFlex-cmd`, scopes repo+workflow) — прежняя заметка про invalid token устарела. Ничего не запушено. Перед PR: `bash scripts/verify.sh release`, PR `dev → main` без merge, без force push.
