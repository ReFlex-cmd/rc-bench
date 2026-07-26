# Project Contract: RC-Bench JMLC 2026

Статус: утверждённый рабочий контракт. Дата: 24 июля 2026 года. Защита: 27 июля 2026 года.

## Цель

К защите `rc-bench` должен демонстрировать воспроизводимое сравнение резервуарных моделей на реальном временном ряде с единым fair-протоколом, простыми baseline, проверяемыми артефактами и честным профилем качества, задержки и памяти.

Проект не должен изображать готовый energy benchmark без аппаратных измерений. Итоговая формулировка: reproducible reservoir-computing benchmark with edge-oriented resource profiling.

## Ветвление и релиз

- Рабочая база: `dev`.
- Старый `dev` сохраняется как архив до синхронизации.
- Feature-ветки создаются от актуального `dev`.
- `main` изменяется единственным итоговым PR `dev → main`.
- Массовое восстановление старых audit/cache/graphify файлов запрещено.

## Датасет

Источник: UCI Individual Household Electric Power Consumption.

Обязательные параметры:

| Поле | Значение |
|---|---|
| Target | `Global_active_power` |
| Частота | 1 час |
| Размер окна | 12 000 последовательных часов |
| Горизонты | 1 и 24 часа |
| Split | хронологический 60/20/20 |
| Перемешивание | запрещено |
| Scaler | обучается только на train |
| Исходный файл в Git | запрещён |

Пайплайн обработки:

1. Скачать данные только через отдельную воспроизводимую команду.
2. Проверить SHA-256 исходного файла по manifest.
3. Разобрать дату и время, отсортировать временную ось, удалить точные дубликаты с явным логом.
4. Преобразовать `?` в missing values.
5. Агрегировать `Global_active_power` почасовым средним; час считается наблюдаемым при наличии минимум 30 валидных минут.
6. Сохранить маску наблюдаемых и восстановленных часов.
7. Заполнить входной ряд только каузально: forward fill; начальный префикс без прошлого наблюдения удалить.
8. Выбрать первое детерминированное окно из 12 000 часов после формирования регулярной шкалы.
9. Не включать примеры с восстановленным target в train loss и основные val/test метрики.
10. Обучить scaler только на train и применить его без переоценки к val/test.

Если этот алгоритм приходится изменить из-за фактической структуры данных, изменение сначала фиксируется в `DECISIONS.md`, затем покрывается тестом и отражается в manifest.

## Экспериментальная матрица

На каждом горизонте выполняются:

| Группа | Модели |
|---|---|
| Baseline | persistence, seasonal persistence с лагом 24, Ridge AR |
| Reservoir | ESN, Leaky ESN, LSM, Logistic |

Всего: 14 сочетаний модель–горизонт.

Deep ESN, FHN и QRC остаются в синтетической части и не входят в обязательную real-data матрицу.

## Fair protocol

- Seeds стохастических моделей: `42, 43, 44, 45, 46`.
- Smoke: 1 seed и 2 HPO trials.
- Основной прогон: 5 seeds и 20 HPO trials для каждой настраиваемой reservoir-модели на каждом горизонте.
- Все reservoir-модели получают одинаковый бюджет.
- Persistence и seasonal persistence не получают искусственный HPO.
- Ridge AR использует заранее зафиксированный train/validation search по единой сетке alpha; это отмечается отдельно от Optuna budget.
- HPO выполняется один раз на train/validation. Найденный resolved spec фиксируется и оценивается на пяти seed.
- Test используется ровно для финальной оценки.
- Best-effort режим не входит в обязательный прогон и не смешивается с fair-таблицей.

## Метрики качества

Headline: `NRMSE_std`.

Обязательные сопутствующие метрики:

- MAE;
- RMSE;
- MASE с seasonal-naive denominator, рассчитанным только по train с лагом 24;
- MAE skill относительно seasonal persistence: `1 - MAE_model / MAE_seasonal`;
- mean, sample std и per-seed values для стохастических моделей.

Детерминированные baseline запускаются один раз и помечаются как deterministic; одинаковые копии под разными seed не создаются.

## Ресурсный профиль

Обязательные измерения на одном компьютере:

- hardware/software profile;
- single-step latency после warmup;
- p50 и p95;
- throughput, samples/s;
- peak RSS в изолированном процессе;
- сериализованный размер модели;
- размер рабочего состояния.

Протокол latency:

- один поток, если библиотека позволяет его контролировать;
- минимум 100 warmup steps;
- минимум 1 000 измеряемых steps либо больше до устойчивого времени;
- `perf_counter_ns`;
- сырые измерения или достаточная агрегированная статистика сохраняются;
- train/HPO time не смешивается с inference latency.

## Energy

Счётчик RAPL доступен и используется (DEC-021). Измерение идёт по
package-домену `/sys/class/powercap/intel-rapl:N`, с вычитанием базовой линии
простоя и окном не короче 2 с; подробности протокола — в `reports/jmlc_2026/README.md`.

`intel-rapl` в пути — имя control type в powercap-иерархии ядра, а не
требование к вендору CPU: драйвер `intel_rapl_msr` с Linux 5.11 обслуживает и
AMD Zen, читая RAPL-регистры AMD, и регистрируется под тем же именем. Бандл
снят на AMD Ryzen; разбор — в `reports/jmlc_2026/README.md`, раздел «Энергия».

Обязательное представление при наличии счётчика:

```json
{
  "energy": {
    "status": "measured",
    "backend": "intel_rapl",
    "domains": ["package-0"],
    "window_target_met": true,
    "net_energy_per_inference_mj": 0.1297,
    "net_samples_per_joule": 7708.0,
    "energy_delay_product_j_s": 2.77e-12
  }
}
```

При отсутствии счётчика, недоступности файла счётчика (root-only с
CVE-2020-8694) или окне короче периода обновления счётчика — обязательный
явный отказ с причиной:

```json
{
  "energy": {
    "status": "unavailable",
    "reason": "<конкретная причина>",
    "backend": null
  }
}
```

Схема (`EnergyResult`) запрещает промежуточные состояния в обе стороны:
`measured` без чисел и числа при `unavailable` не проходят валидацию, и
профиль прогоняет свой блок через неё перед записью.

Запрещено оценивать энергию через TDP, CPU time или паспортную мощность.
Operation count, state sparsity, spikes и synaptic events публикуются только
как отдельно подписанные activity proxies в блоке `activity` и никогда не
пересчитываются в джоули (DEC-022).

## Evidence bundle

Итоговый каталог:

```text
reports/jmlc_2026/
├── README.md
├── dataset_manifest.json
├── hardware_profile.json
├── fair/
│   ├── runs/            # RunRecord каждой ячейки
│   ├── artifacts/       # предсказания и таргеты
│   └── summary.json
├── best_effort/         # второй контур, та же раскладка
├── profiles/            # ресурсный профиль + activity + energy (только fair)
├── aggregates/          # matrix_table[_best_effort].{json,csv}, eda_*
├── plots/               # pareto_quality_{latency,memory,energy}.png, eda_*
└── selection.md         # выбор модели под ограничения устройства
```

**Расхождение с раскладкой выше по тексту, которое проверяющему нужно знать.**
Отдельных каталогов `specs/frozen/`, `specs/resolved/` и `hpo/` в бандле нет:
frozen-спека, resolved-спека и траектория HPO физически лежат внутри каждого
RunRecord (`fair/runs/*.json`, поля `spec`, `resolved_spec`,
`result.selection.candidates`). Информация не потеряна — она собрана в одном
файле на ячейку, чтобы результат нельзя было отделить от спеки, которая его
породила. Каталог `runs/` находится не в корне бандла, а внутри контура
режима, потому что контуров теперь два и сводить их в один каталог значило бы
смешивать результаты, полученные по разным правилам (DEC-023).

Каждый результат должен позволять восстановить:

- commit SHA;
- frozen и resolved spec;
- dataset manifest и checksum;
- seed;
- версии библиотек;
- hardware profile;
- HPO trajectory;
- quality и resource metrics.

Перед публикацией удаляются hostname, username, абсолютные пути, токены и другие локальные идентификаторы.

## Критерии результата для JMLC

- Инженерия: self-contained unit tests, CI, CLI smoke, воспроизводимый запуск.
- Data Science: реальный ряд, отдельный EDA, causal preprocessing, baselines и fair protocol.
- AI-инструменты: `AI_USAGE.md`, разделение работы агентов, ручная проверка, тесты и evidence.
- Продуктовое мышление: Pareto quality-latency, quality-memory и quality-energy, сценарий выбора модели под ограничение устройства (`selection.md`, `rcbench select`).
- Честность: energy unavailable, planned и implemented результаты не смешиваются.
