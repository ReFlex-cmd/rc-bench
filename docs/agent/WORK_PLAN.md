# Исполнимый план до защиты JMLC

Дата старта: 24 июля 2026 года. Дата защиты: 27 июля 2026 года. Бюджет: не менее 20 часов активной работы плюс машинное время.

## Общий принцип

Сначала устраняются дефекты достоверности и фиксируется контракт, затем строится минимальный вертикальный real-data pipeline, после smoke-прогона выполняется полная матрица, и только по полученным артефактам обновляются README и презентация.

## Этап 0. Безопасная подготовка веток

Проверить отношение `origin/dev` и `origin/main`. Если `dev` по-прежнему отстаёт:

```bash
git fetch origin
git status --short --branch
git branch archive/dev-pre-jmlc-20260724 origin/dev
git push origin archive/dev-pre-jmlc-20260724
git switch -C dev origin/dev
git merge --ff-only origin/main
git push origin dev
```

Если рабочее дерево не чистое, ветки разошлись или имя архивной ветки уже занято другим SHA, остановиться и разрешить ситуацию явно. Не использовать force push.

Результат: актуальный `dev` основан на `main`, полезные старые материалы доступны через архивную ветку, будущий PR не возвращает удалённые кэши.

## Этап 1. Harness и correctness gate

Добавить repository-native файлы из этого пакета. Затем исправить:

- фактическое применение `train_frac/val_frac` в CLI и Celery;
- сохранение frozen и resolved spec после HPO;
- доступность predictions для demo-запусков;
- self-contained unit test mode без PostgreSQL;
- минимальный GitHub Actions workflow.

Сначала тесты на каждый дефект, затем код. Этап завершается зелёным `bash scripts/verify.sh quick`.

Оценка: 3-4 часа.

## Этап 2. Минимальный real-data vertical slice

Реализовать download/manifest, loader, почасовой ряд, causal preprocessing и один минимальный spec:

- horizon 1;
- persistence;
- ESN;
- 1 seed;
- без HPO или с 2 smoke trials.

Цель — доказать прохождение полного пути `download → preprocess → split → model → metrics → RunRecord`, не строя сразу всю матрицу.

Оценка: 3-4 часа.

## Этап 3. EDA и baselines

EDA должен одной командой создавать отчёт и графики:

- диапазон и покрытие;
- missingness до и после ресемплинга;
- распределение target;
- выбросы;
- суточный и недельный профиль;
- ACF на разумном числе лагов;
- сравнение распределений train/val/test;
- явную проверку временной утечки.

Добавить persistence, seasonal persistence и Ridge AR через общий результатный контракт. Добавить MASE и MAE skill с тестами на простых массивах.

Оценка: 4-5 часов.

## Этап 4. Edge-oriented profiling

Реализовать изолированный resource profile: p50/p95 latency, throughput, peak RSS, model bytes и state bytes. Сохранить measurement protocol и hardware profile. Energy записывается как `unavailable`.

Operation count, state sparsity и LSM event statistics выполняются только после P0, если остаётся время.

Оценка: 2-3 часа.

## Этап 5. Smoke и основная матрица

Smoke:

- 14 model-horizon cells;
- reservoir: 1 seed, 2 trials;
- deterministic baselines: 1 запуск;
- проверка schema, артефактов, времени и отсутствия test leakage.

После успешного smoke оценить машинное время. Если полный прогон укладывается в окно, запустить:

- 4 reservoir × 2 horizons × 20 trials;
- fixed best config × 5 seeds;
- 3 baseline × 2 horizons;
- общий aggregation и Pareto plots.

Если машинное время слишком велико, сначала уменьшить HPO до 10 trials для всех reservoir-моделей одинаково. Пять seed, baselines, оба горизонта и evidence bundle не сокращать.

Оценка активной работы: 2-3 часа плюс машинное время.

## Этап 6. Evidence и защита

Сформировать `reports/jmlc_2026/`, проверить traceability и sanitization. Обновить:

- README;
- `AI_USAGE.md`;
- методику эксперимента и ограничения;
- demo-config и сценарий демонстрации;
- таблицы и Pareto-графики для презентации.

В материалы защиты переносить только значения из агрегатов. Energy plot удалить; вместо него показать quality-latency и quality-memory и явно объяснить отсутствие аппаратного счётчика.

Оценка: 3-4 часа.

## Этап 7. Freeze и релиз

После `bash scripts/verify.sh release`:

- проверить diff относительно `main`;
- убедиться, что датасет, secrets, hostname, пути и временные файлы не попали в Git;
- проверить воспроизводимость demo по README;
- создать один PR `dev → main`;
- не добавлять новые функции после freeze без критической причины.

Оценка: 1-2 часа.

## Рекомендуемая последовательность по времени

| Окно | Результат |
|---|---|
| 24 июля | ветки, harness, correctness, CI |
| 24-25 июля | real-data vertical slice |
| 25 июля | EDA, baselines, quality metrics |
| 25-26 июля | profiling, smoke, полный прогон |
| 26 июля | evidence, README, AI_USAGE, demo, слайды |
| 27 июля | freeze, финальная проверка, PR и репетиция |

## Несокращаемый минимум

При дефиците времени сохранить: manifest, causal preprocessing, EDA, три baseline, оба горизонта, пять seed reservoir-моделей, raw RunRecord, latency/RSS profile и честные ограничения.

Сокращать в порядке:

1. activity proxies;
2. число HPO trials одинаково для всех reservoir-моделей;
3. декоративные графики и вторичные отчёты.

Запрещено сокращать качество доказательств: нельзя использовать test для HPO, скрывать пропуски, подменять energy прокси-метриками или публиковать числа без RunRecord.
