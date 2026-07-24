# Decision Log

Новые записи добавляются в конец. Старые решения не переписываются: изменение оформляется новой записью со ссылкой на заменённое решение.

## DEC-001 — Рабочая ветка

Дата: 24.07.2026. Статус: accepted.

Рабочая база — `dev`; `main` изменяется одним итоговым PR. Исторический `dev` сохраняется в архивной ветке до синхронизации с `main`.

## DEC-002 — Сокращённая real-data матрица

Дата: 24.07.2026. Статус: accepted.

На реальном датасете сравниваются persistence, seasonal persistence, Ridge AR, ESN, Leaky ESN, LSM и Logistic на горизонтах 1 и 24. Deep ESN, FHN и QRC остаются в синтетическом контуре.

Причина: законченный и проверяемый эксперимент важнее широкой, но незавершённой матрицы.

## DEC-003 — Реальный датасет

Дата: 24.07.2026. Статус: accepted.

Используется UCI Individual Household Electric Power Consumption, `Global_active_power`, почасовая частота и детерминированное окно 12 000 часов.

## DEC-004 — Fair HPO

Дата: 24.07.2026. Статус: accepted.

Smoke использует 1 seed и 2 trials. Основной прогон использует 20 trials на каждую reservoir-модель и горизонт, затем фиксированный resolved spec оценивается на seed `42-46`. Если машинного времени недостаточно, бюджет одинаково уменьшается до 10 trials.

## DEC-005 — Headline и baseline metrics

Дата: 24.07.2026. Статус: accepted.

Headline — `NRMSE_std`. Рядом публикуются MAE, RMSE, seasonal MASE и MAE skill относительно seasonal persistence.

## DEC-006 — Пропуски и ресемплинг

Дата: 24.07.2026. Статус: accepted.

Час считается наблюдаемым при минимум 30 валидных минутных значениях. Входной ряд заполняется forward fill без будущих данных. Восстановленные targets исключаются из обучения readout и основных val/test метрик.

## DEC-007 — Energy

Дата: 24.07.2026. Статус: accepted.

Полноценные energy measurements исключены: совместимого аппаратного счётчика нет. Поле energy сохраняется со статусом `unavailable`. TDP и время CPU не используются как оценка энергии. Activity proxies допускаются только как отдельный P1-блок.

## DEC-008 — Публикация артефактов

Дата: 24.07.2026. Статус: accepted.

Публикуются dataset manifest, frozen/resolved specs, HPO diagnostics, сырые RunRecord, агрегаты, hardware profile и итоговые графики. Исходный датасет не публикуется. Hostname, username и абсолютные пути удаляются.

## DEC-009 — Приоритет перед защитой

Дата: 24.07.2026. Статус: accepted.

P0: correctness, real data, EDA, baselines, fair protocol, latency/memory, evidence, CI и demo. P1: operation count, sparsity, LSM events. P2: API hardening, energy backend и расширение матрицы.

## DEC-010 — Frozen и resolved спецификации

Дата: 24.07.2026. Статус: accepted.

`ResultSpec.config_hash` всегда относится к фактически оценённому resolved spec. `ResultSpec` дополнительно сохраняет `frozen_config_hash` и `resolved_spec`, а `RunRecord` хранит исходный `spec` как frozen и отдельный `resolved_spec`. Старые RunRecord без `resolved_spec` загружаются совместимо, используя исходный `spec` в обеих ролях.

## DEC-011 — Фиксация UCI-источника

Дата: 24.07.2026. Статус: accepted.

Канонический manifest версии 1 хранится в `configs/jmlc/dataset_manifest.json` и фиксирует официальный UCI URL, размер и SHA-256 архива и извлечённого файла. Поскольку UCI не публикует подписанный SHA-256, закреплённый архив дополнительно сверяется с официальным legacy endpoint. Изменение байтов upstream считается ошибкой и требует отдельного проверяемого обновления manifest.

## DEC-012 — Выравнивание горизонта на границах split

Дата: 24.07.2026. Статус: accepted.

Сначала фиксируются непересекающиеся хронологические блоки 60/20/20, затем внутри каждого блока строятся пары `X[:-h] → y[h:]`. Контекст из предыдущего блока не переносится в validation или test. Одинаковая семантика применяется к reservoir-моделям и baseline, чтобы ни одна группа не получала дополнительные первые `h` targets на границе split.

## DEC-013 — Общий JMLC evaluation protocol

Дата: 24.07.2026. Статус: accepted.

Для всех 14 real-data cells используется `fixed_horizon`, в том числе при горизонте 1; `one_step` с unshifted UCI-рядом запрещён как target leakage. Общий per-split warmup равен 200 часам, поэтому оцениваемые target indices для горизонта `h` начинаются с `200 + h`. Persistence, seasonal persistence, Ridge AR и reservoir-модели обязаны использовать один и тот же набор наблюдаемых target timestamps.

Ridge baseline фиксируется как AR(24) с train-only z-score входных lag-признаков, нескалированным target, intercept и общей для обоих горизонтов alpha-grid `[0.001, 0.01, 0.1, 1.0, 10.0]`. Validation selection для Ridge и reservoir HPO выполняется по `NRMSE_std`; legacy synthetic runs сохраняют прежний default `NRMSE_range`.

MASE использует train-only seasonal scale с лагом 24, а MAE skill — seasonal persistence на точно том же test target set. Маска применяется к текущему target; causally forward-filled прошлое значение допустимо как input. Пустой набор или нулевой denominator считается явной ошибкой протокола, а не публикуемым `NaN`/`Infinity`.

## DEC-014 — Явное представление baseline

Дата: 24.07.2026. Статус: accepted.

Persistence, seasonal persistence и Ridge AR представляются отдельным `BaselineSpec`, а не типами reservoir registry. Они дают deterministic single-result, не проходят через multi-seed и не получают фиктивный seed. Persistence-модели имеют selection `none`; Ridge AR сохраняет `fixed_grid` selection отдельно от Optuna HPO. Расширение схемы аддитивно: старые reservoir specs и RunRecord продолжают загружаться, а новые результаты явно сохраняют model family, deterministic status, реально оценённые seeds и selection metadata.
