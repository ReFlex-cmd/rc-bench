# Audit Baseline

Дата фиксации: 24 июля 2026 года. База аудита: `main` на коммите `bb38f1a6`; исторический `dev` на `d9c4244`.

Этот файл — исходная гипотеза для последующей проверки, а не замена повторному аудиту актуальной ветки.

## Подтверждённое состояние

- Единый pipeline связывает `ExperimentSpec`, HPO, single/multi-seed evaluation и `ResultSpec`.
- Реализованы ESN, Leaky ESN, Deep ESN, LSM, FitzHugh-Nagumo, Logistic и QRC-inspired mean-field reservoir.
- Реализованы NARMA-10, стабилизированная NARMA-30, Mackey-Glass и одномерный Lorenz-63.
- Поддерживаются one-step, fixed-horizon и closed-loop.
- Есть Ridge readout, Optuna HPO, multi-seed aggregation, CLI, FastAPI, Celery, Redis, PostgreSQL, Alembic, Docker Compose и Nginx.
- RunRecord сохраняет git hash, версии библиотек, аппаратный профиль, HPO diagnostics и метрики.
- Изолированный unit-набор проходил: `289 passed, 1 integration test deselected`.

## Подтверждённые разрывы

- Реального датасета и отдельного EDA нет.
- Persistence, seasonal persistence и Ridge AR отсутствуют.
- Ресурсные показатели не соответствуют строгому Edge protocol: latency измеряется для полного rollout, `tracemalloc` не покрывает RSS и native allocations.
- Аппаратного energy backend и совместимого устройства нет.
- Сырые `reports/runs/*.json` и `reports/full_results.json`, на которые ссылается отчёт, не опубликованы.
- GitHub Actions отсутствует; badge тестов статический.
- Полный pytest из чистого клона требует обязательных переменных окружения; integration test требует PostgreSQL.
- В репозитории заявлен MIT, но файл лицензии на момент аудита не найден.

## Подтверждённые дефекты

1. CLI и Celery игнорируют пользовательские `train_frac` и `val_frac`, фактически используя `0.6/0.2`.
2. После HPO `config_hash` относится к настроенной конфигурации, но RunRecord сохраняет исходный spec вместо атомарной пары frozen/resolved spec.
3. При стандартном multi-seed predictions обычно не сохраняются, поэтому демонстрационный график результата недоступен.
4. В API не везде проверяется владелец эксперимента.

## Состояние веток на момент аудита

Исторический `dev` был предком `main` на четыре коммита. В `main` позднее удалили старые аудиты, планы, `graphify-out` и временные артефакты. Прямой PR из старого `dev` в `main` мог вернуть удалённые файлы. Перед началом разработки это отношение необходимо проверить заново.

Полезные материалы старого `dev` используются только как справка:

- `audit/architecture_map.md`;
- `audit/01_implementations.md` — `audit/05_test_results.md`;
- `CLAUDE_CODE_AUDIT_TASK.md`;
- `REFACTORING_PLAN.md`;
- `graphify-out/GRAPH_REPORT.md`.

Кэши и полный `graphify-out` в актуальную ветку не возвращаются.
