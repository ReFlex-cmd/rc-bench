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
