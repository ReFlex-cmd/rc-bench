# RC-Bench JMLC 2026 — состояние проекта и handoff

> Дата среза: 25 июля 2026. Ветка `dev`, 53 коммита поверх `main` (`bb38f1a`).
> Этот файл самодостаточен: он включает выжимку из `docs/agent/` и `docs/`, чтобы
> следующая сессия могла продолжить работу, не восстанавливая контекст по кускам.
> Полные документы: [docs/PROJECT_CONTRACT.md](docs/PROJECT_CONTRACT.md),
> [docs/DECISIONS.md](docs/DECISIONS.md), `docs/agent/WORK_PLAN.md`,
> `docs/agent/BACKLOG.md`, `docs/agent/AGENT_WORKFLOW.md`.

---

## 1. Что это за проект и зачем

`rc-bench` — воспроизводимый бенчмарк-сьют для сравнения типов резервуарных вычислений.
До июля 2026 в нём был только синтетический контур (NARMA-10/30, Mackey-Glass, Lorenz-63).
Работа последних трёх сессий — довести проект до защиты JMLC (27 июля 2026): добавить
контур на **реальных данных** с честным протоколом, сильными классическими baseline,
ресурсным профилем и проверяемым evidence-бандлом.

Ключевой тезис защиты: сравнение имеет смысл только тогда, когда все модели оценены на
одном и том же наборе целевых точек, а стоимость вывода измерена отдельно от обучения.

## 2. Контракт проекта (выжимка `docs/PROJECT_CONTRACT.md`)

**Датасет.** UCI Individual Household Electric Power Consumption. Почасовой ряд, окно
12 000 часов, хронологический split 60/20/20 без перемешивания. Источник пиннуется
манифестом с SHA-256 архива и сырого файла; сами данные в Git не попадают.

**Матрица.** 7 моделей × 2 горизонта = 14 cells (DEC-002):
baseline — persistence, seasonal persistence, Ridge AR(24);
reservoir — ESN, Leaky ESN, LSM, Logistic. Deep ESN, FHN и QRC остаются в синтетическом
контуре. Горизонты 1 и 24 часа.

**Fair protocol.** Равный HPO-бюджет для всех reservoir-моделей (DEC-004), селекция только
по validation, финальная оценка на test ровно один раз, 5 seeds для reservoir-моделей,
детерминированные baseline запускаются однократно и не получают фиктивных seed.

**Метрики.** Headline — NRMSE_std. Baseline-относительные: MASE (train-only seasonal scale,
лаг 24) и MAE skill относительно seasonal persistence на том же test target set.

**Ресурсный профиль.** hardware/software profile; single-step latency после warmup; p50 и
p95; throughput; peak RSS в изолированном процессе; сериализованный размер модели; размер
рабочего состояния. Протокол latency: один поток, ≥100 warmup, ≥1000 измеряемых шагов,
`perf_counter_ns`, сырые измерения сохраняются, train/HPO-время не смешивается с inference.

**Energy.** Аппаратного счётчика нет — energy во всех записях `unavailable`, прокси-метрики
вместо измерений не публикуются.

**Критерии результата.** Инженерия (self-contained tests, CI, CLI smoke, воспроизводимый
запуск); Data Science (реальный ряд, EDA, causal preprocessing, baselines, fair protocol);
AI-инструменты (`AI_USAGE.md`, разделение работы агентов, ручная проверка); продуктовое
мышление (Pareto quality-latency и quality-memory); честность (energy unavailable, planned
и implemented не смешиваются).

## 3. Текущее состояние

- Ветка `dev`, рабочее дерево чистое. Активных сабагентов и фоновых процессов нет.
- `bash scripts/verify.sh quick` → **489 passed, 1 deselected**.
- `bash scripts/verify.sh release` → **проходит целиком**: тесты, наличие обязательных
  файлов бандла, `validate_evidence.py` (OK, 14 cells), санитизация, проверка что сырой
  датасет не отслеживается Git.
- Evidence-бандл `reports/jmlc_2026/` собран и валиден.

### Закрытые задачи (`docs/agent/BACKLOG.md`)

| Блок | ID | Статус |
|---|---|---|
| Ветки и harness | BOOT-001/002 | DONE |
| Correctness | COR-001/002/003, TEST-001, CI-001 | DONE |
| Данные | DATA-001/002/003, EDA-001 | DONE |
| Vertical slice | VSLICE-001 | DONE |
| Baselines и метрики | BASE-001/002, MET-001 | DONE |
| Профилирование | PROF-001/002/003 | DONE |
| Energy | ENERGY-001 | DONE |
| Эксперименты | EXP-001/002/003 | DONE |
| Evidence и графики | EVID-001, PLOT-001 | DONE |
| Документация и демо | DOC-001, DOC-002, DEMO-001 | DONE |
| Презентация | PRES-001 | **BLOCKED** — файла презентации нет в репозитории |
| Релиз | REL-001 | PR открыт, merge за человеком |
| P1/P2 | PROXY-001/002, REPO-001, API-001, ENERGY-002 | не начаты / BLOCKED |

## 4. Что сделано по сессиям

### Сессии 1–2 (коммиты `4438154`…`8f6a6ce`)

Установка harness; исправление correctness-дефектов (реальное применение `train_frac/
val_frac`, разделение frozen/resolved spec, predictions для demo, self-contained unit
mode, CI на Python 3.12); загрузка и верификация UCI по манифесту; почасовой loader с
масками пропусков и каузальным препроцессингом; EDA-отчёт с графиками; изоляция состояния
ESN между split (был blocker); baseline-предикторы и Ridge AR; MASE и MAE skill;
baseline-диспетч в `run_pipeline`; профилировочные утилиты (latency/memory/hardware);
раннер матрицы `jmlc_matrix.py` и smoke-конфиг; `AI_USAGE.md`.

### Сессия 3 (12 коммитов, `1cc5b3b`…HEAD)

**Эксперименты.** Smoke 14/14 (17 с). Fair-матрица 14/14 за **1.9 минуты** — прошлая оценка
«нужно согласовать окно compute» не подтвердилась (самая тяжёлая LSM-ячейка на полном
бюджете 20 trials × 5 seeds заняла 21 с). Конфиг `configs/jmlc/fair.yaml`.

**Три дефекта достоверности evidence** (`46bd1f9`), найденные при проверке smoke:

1. В каждом RunRecord лежал `hostname` — нарушение DEC-008, и release gate упал бы на всём
   бандле. Поле убрано из модели, а не вычищается при сборке: то, что нужно санитизировать
   позже, рано или поздно утечёт.
2. `EvaluationContext` писали только baseline, у reservoir-моделей он был `null` — общий
   target set и общий MASE scale нельзя было проверить по самим записям. Теперь пишут оба
   семейства, включая multi-seed путь (DEC-018).
3. `dataset.raw_sha256` не заполнялся вообще. Теперь пиннуется в конфиге, а `run_matrix`
   **хеширует реальный файл** перед прогоном и отказывается считать при несовпадении
   (DEC-017) — привязка результатов к байтам стала фактом, а не заявлением.

**EVID-001** (`5f55d7d`). `scripts/validate_evidence.py` требовался release-гейтом, но не
существовал. Написан вместе с `src/rc_bench/reporting/evidence.py`: агрегация раw-записей в
`aggregates/matrix_table.{json,csv}` и перепроверка того, что бандл заявляет — статусы,
config hashes против frozen/resolved спеков, единый digest датасета, идентичный
`EvaluationContext` на каждом горизонте, объявленные seeds, равный HPO-бюджет, конечность
метрик, профиль на том же config hash. Таблица, отредактированная руками, валидацию не
проходит.

**PROF-003** (`340384d`, `b047e4d`, сабагент + cherry-pick). Профилировочный проход
`scripts/run_profiling.py`: latency, peak RSS, размеры модели и рабочего состояния по всем
14 cells, плюс `hardware_profile.json`.

**Ключевая находка — latency измеряла sklearn, а не модели** (`949b3d8`). Первый реальный
профиль выдал Ridge AR (87.4 мкс) «медленнее» ESN (73.1 мкс), что для 24-признакового
линейного предиктора физически бессмысленно. Замер примитивов: `Ridge.predict(1×300)` =
42.2 мкс, `Ridge.predict(1×24)` = 31.8 мкс, `StandardScaler.transform` = 38.3 мкс — против
1.05 мкс на `coef @ state` и 10.7 мкс на плотное обновление состояния 300×300. Накладной
расход фреймворка составлял **55–96 %** публикуемых чисел, то есть матрица ранжировала
модели по числу вызовов sklearn. Добавлено второе измерение `latency_deployable` (те же
коэффициенты, арифметика вместо per-sample API); равенство предсказаний обоих путей
проверяется с `rtol=1e-9` (DEC-019).

**Дефект гейта** (`9761c05`). `if rg ...; then fail; fi` при отсутствии `rg` даёт false —
обе проверки санитизации release-гейта молча пропускались; `rg` на машине разработчика
существует только как shell-функция, невидимая скрипту. Переведено на `grep`, проверено
подкладыванием `/home/...`: гейт падает и называет файл. Сам бандл был чист — проверки
просто никогда не выполнялись.

**PLOT-001, DOC-001, DEMO-001** (`9cd490a`, `36e5f81`, `eb32fd4`, `ada4b13`). Pareto-графики
quality-latency и quality-memory; секции реальных данных в README и полный README
evidence-бандла; `configs/jmlc/demo.yaml` (~13 с) и вывод MASE/MAE-skill в CLI.

**Реструктуризация документации** (этот коммит). `PROJECT_CONTRACT.md` и `DECISIONS.md`
перенесены из `docs/agent/` в `docs/`: это научные документы (протокол и журнал решений),
на которые ссылается опубликованный бандл, поэтому они должны быть в `main`, тогда как
процессные документы агентов остаются только в `dev`.

## 5. Результаты

Все числа — из `reports/jmlc_2026/aggregates/matrix_table.json` и
`reports/jmlc_2026/profiles/summary.json`.

**Качество (NRMSE_std, ниже лучше), h=1 / h=24:**

| Модель | Семейство | h=1 | h=24 | MAE skill h=1 | MAE skill h=24 |
|---|---|---|---|---|---|
| esn | reservoir | **0.6647 ± 0.0020** | **0.8705 ± 0.0048** | 0.400 | 0.150 |
| leaky_esn | reservoir | 0.6743 ± 0.0012 | 0.8910 ± 0.0194 | 0.397 | 0.130 |
| ridge_ar | baseline | 0.6760 | 0.9023 | 0.388 | 0.120 |
| logistic | reservoir | 0.7513 ± 0.0042 | 0.9656 ± 0.0108 | 0.309 | 0.060 |
| persistence | baseline | 0.7682 | 1.1273 | 0.354 | 0.000 |
| lsm | reservoir | 0.9449 ± 0.0029 | 0.9832 ± 0.0014 | 0.028 | −0.019 |
| seasonal_persistence | baseline | 1.1230 | 1.1273 | 0.000 | 0.000 |

**Стоимость (deployable p50, h=1):** persistence 0.27 мкс, seasonal 0.40 мкс, ridge_ar
3.37 мкс, logistic 9.59 мкс, esn 21.82 мкс, leaky_esn 22.09 мкс, lsm 43.05 мкс.
Рабочее состояние: 192 Б у ridge_ar против 2.3 КиБ у ESN; размер модели 1.5 КиБ против
75 КиБ.

**Вывод.** ESN лучший на обоих горизонтах, но покупает 1.7 % (h=1) и 3.5 % (h=24) точности
над Ridge AR ценой ~5× стоимости шага, 12× рабочего состояния и 50× размера модели. На
Pareto-фронте остаются persistence, Ridge AR и ESN; leaky ESN, logistic и LSM доминируются
(одновременно дороже и хуже). LSM на h=24 хуже сезонного наива (skill −0.019) — опубликован
как есть. На h=24 persistence и seasonal persistence совпадают до последнего знака, потому
что при `h = season = 24` это один и тот же предиктор; MAE skill сезонной модели равен нулю
по построению и работает как проверка корректности скоринга.

## 6. Решения (`docs/DECISIONS.md`)

DEC-001 рабочая ветка · DEC-002 сокращённая real-data матрица · DEC-003 реальный датасет ·
DEC-004 fair HPO · DEC-005 headline и baseline metrics · DEC-006 пропуски и ресемплинг ·
DEC-007 energy · DEC-008 публикация артефактов · DEC-009 приоритеты · DEC-010 frozen и
resolved спеки · DEC-011 фиксация UCI-источника · DEC-012 выравнивание горизонта на
границах split · DEC-013 общий evaluation protocol · DEC-014 явное представление baseline ·
DEC-015 Ridge AR: обучение и селекция · **DEC-016** RunRecord без machine identity ·
**DEC-017** pinned digest проверяется на реальных байтах · **DEC-018** EvaluationContext
обязателен для обоих семейств · **DEC-019** latency публикуется в deployable-виде ·
**DEC-020** peak RSS — footprint процесса, не модели.

## 7. Правила работы (`docs/agent/AGENT_WORKFLOW.md`, подтверждено пользователем)

- Гибрид: сабагенты на тяжёлых независимых задачах (профилирование, графики);
  cross-cutting (`schema.py`, `pipeline.py`, `run_record.py`, раннеры, BACKLOG, DECISIONS) —
  основной агент. Скилл `subagent-driven-development`.
- **Worktree сабагента** создаётся от устаревшего базового коммита: интегрировать
  `git cherry-pick <sha>`, не merge; после — `git worktree remove --force` + `git branch -D`.
  **Не удалять worktree, пока не решено, нужны ли доработки** — иначе сабагента не
  возобновить (в этой сессии так и вышло: доработку latency пришлось делать основному агенту).
- Идти автономно по цепочке next-steps до блокера. Коммитить логическими единицами с
  зелёным гейтом. **Не push без запроса**, `main` напрямую не менять, force push запрещён.
- `gh auth status` валиден (`ReFlex-cmd`, scopes repo+workflow).

## 8. Следующие шаги

1. **REL-001 — merge PR `dev → main`.** PR открыт; финальное решение о слиянии за человеком
   (по контракту это его зона ответственности).
2. **PRES-001 — BLOCKED:** файла презентации в репозитории нет. Нужен сам файл или решение
   делать слайды с нуля. Всё для переноса готово: `reports/jmlc_2026/README.md` и
   `plots/pareto_quality_{latency,memory}.png`.
3. **REPO-001 — вопрос к человеку:** README содержит бейдж MIT и ссылку `[MIT](LICENSE)`,
   но **файла `LICENSE` в репозитории нет**. Перед публичным релизом либо добавить файл
   выбранной лицензии, либо убрать заявление.

## 9. Известные pending-элементы и отклонения

- **Раскладка бандла отличается от контракта.** Контракт описывает `specs/frozen/`,
  `specs/resolved/`, `hpo/`, `runs/` как отдельные каталоги; фактически всё это лежит
  внутри каждого RunRecord (`fair/runs/*.json` содержит frozen spec, resolved spec,
  HPO-диагностику и selection), плюс добавлен `profiles/`. Информация не потеряна, но если
  проверяющий будет сверяться с контрактом буквально — это расхождение нужно объяснить.
- `main.py` REST API (`create_experiment`) всё ещё завязан на `reservoir.type` — вне
  JMLC-пути, трек API-001 (P2).
- PROXY-001/002 (operation count, sparsity, LSM events) — P1, не начаты; по WORK_PLAN
  сокращаются первыми.
- `model_profiles.py` при локальном fit reservoir-моделей игнорирует
  `target_observed_mask_*`. На latency и память это не влияет (точность там не измеряется),
  но код нельзя переиспользовать для accuracy-чувствительных задач без правки.
- `peak_rss_bytes` измеряется в forked-процессе и наследует память родителя (DEC-020):
  87–244 МБ — это footprint процесса, не модели. Для настоящей изоляции нужен spawn-процесс,
  собирающий модель с нуля.
- Три docstring в `src/rc_bench/profiling/` ссылаются на `docs/PROJECT_CONTRACT.md` —
  после переноса файла ссылки корректны в обеих ветках.

## 10. Ключевые команды

```bash
bash scripts/verify.sh quick     # держать зелёным перед каждым коммитом
bash scripts/verify.sh release   # полный гейт: тесты + evidence + санитизация

poetry run python scripts/download_jmlc_data.py                    # данные (gitignored)
poetry run python scripts/run_jmlc_matrix.py \
    --config configs/jmlc/fair.yaml --output reports/jmlc_2026/fair
poetry run python scripts/run_profiling.py \
    --config configs/jmlc/fair.yaml --runs reports/jmlc_2026/fair/runs \
    --output reports/jmlc_2026/profiles \
    --hardware-output reports/jmlc_2026/hardware_profile.json
poetry run python scripts/validate_evidence.py reports/jmlc_2026 --write
poetry run python scripts/plot_pareto.py --bundle reports/jmlc_2026 --output reports/jmlc_2026/plots

poetry run rcbench run configs/jmlc/demo.yaml \
    --output reports/demo/run.json --artifacts reports/demo/artifacts   # демо, ~13 с
```

`reports/jmlc_2026/smoke/` намеренно gitignored — это артефакт гейта, каждый прогон
переписывает timestamps. Публикуется fair-матрица.
