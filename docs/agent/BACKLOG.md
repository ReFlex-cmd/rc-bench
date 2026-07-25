# JMLC Backlog

Статусы: `READY`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `DROPPED`. Одновременно у одного агента может быть только одна задача `IN_PROGRESS`.

| ID | P | Задача | Зависит от | Оценка | Статус |
|---|---:|---|---|---:|---|
| BOOT-001 | P0 | Сохранить старый `dev` и синхронизировать рабочий `dev` с `main` | — | 0.5 ч | DONE |
| BOOT-002 | P0 | Установить harness и зафиксировать baseline | BOOT-001 | 0.5 ч | DONE |
| COR-001 | P0 | Исправить wiring `train_frac/val_frac` в CLI и Celery | BOOT-002 | 1 ч | DONE |
| COR-002 | P0 | Разделить frozen/resolved spec и исправить RunRecord после HPO | BOOT-002 | 1.5 ч | DONE |
| COR-003 | P0 | Сохранять predictions для demo без раздувания всех production runs | COR-002 | 0.5 ч | DONE |
| TEST-001 | P0 | Self-contained unit mode и актуальный integration marker | BOOT-002 | 1 ч | DONE |
| CI-001 | P0 | GitHub Actions: Python 3.12, install, unit gate, CLI smoke | TEST-001 | 1 ч | DONE |
| DATA-001 | P0 | Download command и dataset manifest с SHA-256 | BOOT-002 | 1 ч | DONE |
| DATA-002 | P0 | Hourly loader, masks, 12k window, causal preprocessing | DATA-001 | 2 ч | DONE |
| DATA-003 | P0 | Тесты временной оси, split, missing target и train-only scaler | DATA-002 | 1 ч | DONE |
| VSLICE-001 | P0 | End-to-end horizon-1 persistence + ESN smoke | COR-002, DATA-003 | 1 ч | DONE |
| EDA-001 | P0 | Воспроизводимый EDA report и графики | DATA-002 | 2 ч | DONE |
| BASE-001 | P0 | Persistence и seasonal persistence | DATA-003 | 1 ч | DONE |
| BASE-002 | P0 | Ridge AR и фиксированный alpha search | DATA-003 | 1 ч | DONE |
| MET-001 | P0 | MASE и MAE skill, edge-case tests | BASE-001 | 1 ч | DONE |
| PROF-001 | P0 | Latency p50/p95 и throughput protocol | VSLICE-001 | 1.5 ч | DONE |
| PROF-002 | P0 | Isolated peak RSS, model bytes, state bytes | VSLICE-001 | 1.5 ч | DONE |
| ENERGY-001 | P0 | Schema/status `unavailable`, без energy backend | COR-002 | 0.5 ч | DONE |
| EXP-001 | P0 | Сгенерировать frozen smoke matrix 14 cells | BASE-002, MET-001, PROF-002 | 0.5 ч | DONE |
| EXP-002 | P0 | Выполнить и проверить smoke matrix | EXP-001 | 1 ч + compute | DONE |
| EXP-003 | P0 | Выполнить fair matrix: 20 trials, 5 seeds | EXP-002 | 1 ч + compute | DONE |
| PROF-003 | P0 | Профилировочный проход: latency/RSS/размеры по cells | PROF-002, EXP-002 | 1.5 ч | IN_PROGRESS |
| EVID-001 | P0 | Evidence bundle, aggregation, traceability | EXP-003 | 1.5 ч | DONE |
| PLOT-001 | P0 | Quality-latency и quality-memory Pareto plots | EVID-001 | 1 ч | DONE (ждёт profiles для рендера) |
| DOC-001 | P0 | README, protocol, limitations, demo-config | EVID-001 | 1.5 ч | IN_PROGRESS |
| DOC-002 | P0 | `AI_USAGE.md` с ролями, проверками и вкладом автора | BOOT-002 | 1 ч | DONE |
| DEMO-001 | P0 | Воспроизводимый demo-сценарий и резервные готовые артефакты | DOC-001, PLOT-001 | 1 ч | READY |
| PRES-001 | P0 | Перенести подтверждённые числа и графики в презентацию | EVID-001, PLOT-001 | 1.5 ч | READY |
| REL-001 | P0 | Release gate и PR `dev → main` | PLOT-001, DOC-001, DOC-002, DEMO-001 | 1 ч | READY |
| PROXY-001 | P1 | Operation count и state sparsity | PROF-002 | 1.5 ч | READY |
| PROXY-002 | P1 | LSM spikes и synaptic events | PROF-002 | 1.5 ч | READY |
| REPO-001 | P1 | Добавить файл лицензии после подтверждения выбранной лицензии | BOOT-002 | 0.5 ч | BLOCKED |
| API-001 | P2 | Полная проверка ownership в API | REL-001 | 2 ч | READY |
| ENERGY-002 | P2 | RAPL/Jetson backend при появлении устройства | REL-001 | — | BLOCKED |

## Definition of Done по типу задачи

### Код

- тест воспроизводит контракт или дефект;
- минимальная реализация проходит тест;
- нет несвязанных изменений;
- `bash scripts/verify.sh quick` зелёный;
- публичные схемы и migration path описаны.

### Данные

- источник, лицензия, URL и SHA-256 в manifest;
- исходные данные не добавлены в Git;
- временная ось и пропуски тестируются;
- preprocessing детерминирован и каузален;
- результат повторяется из чистого каталога.

### Эксперимент

- frozen и resolved spec сохранены;
- test не участвовал в HPO;
- seed и commit SHA известны;
- raw и aggregate согласуются;
- failed runs не исчезают из отчёта;
- hardware profile и measurement protocol приложены.

### Документация

- каждое численное заявление ссылается на evidence artifact;
- planned, implemented и unavailable явно разделены;
- demo воспроизводится командами из README;
- локальные идентификаторы удалены.
