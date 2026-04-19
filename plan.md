Нужно доработать проект rc-bench. Работай как поэтапный инженерный рефакторинг с сохранением работоспособности на каждом шаге. Не делай косметику без пользы. Цель — превратить rc-bench в единый воспроизводимый benchmark-suite для reservoir computing: исправить старое, убрать дублирование, интегрировать все существующие reservoir'ы в один контур, затем добавить новые архитектуры, бенчмарки, метрики и протокол честного сравнения.

Главные проблемы текущего состояния, которые нужно исправить в первую очередь:
1. В проекте два расходящихся контура: offline через orchestrator/* и online через src/rc_bench/*; логика данных, метрик и запуска задублирована.
2. Worker поддерживает только ESN, хотя в репозитории уже есть локальные реализации LSM, logistic map и FHN.
3. Сейчас фактически поддерживается только dataset NARMA10.
4. Схемы результатов и названия метрик не унифицированы: локальные скрипты и worker пишут разные ключи.
5. NRMSE нужно явно формализовать: текущий код использует нормировку по range(target), но стенд должен уметь и явно именованные альтернативы.
6. CLI, worker и core должны использовать одну библиотечную прослойку, а не копии логики.
7. Нужно ввести единый protocol layer: split, washout, forecasting mode, HPO budget, multi-seed, reproducibility manifest.

Ожидаемый результат:
- один core library API для всех экспериментов;
- единый ExperimentSpec и ResultSpec;
- единый модуль datasets, metrics, protocol, readout, reservoirs;
- полная интеграция ESN, LSM, logistic, FHN в общий runner;
- добавлены leaky-ESN и DeepESN;
- QRC оформлен как stub/minimal simulated baseline, без претензии на hardware advantage;
- добавлены NARMA30, Mackey-Glass, Lorenz-63; KS, spiking/event benchmark и real multivariate dataset — как следующий слой;
- единые агрегированные отчёты mean±std по multi-seed;
- документация и smoke/integration tests.

Сделай работу в таком порядке.

Этап 1. Свести архитектуру проекта к одному ядру.
1. Убери дублирование между orchestrator/* и src/rc_bench/*.
2. Собери библиотечное ядро со следующей структурой:
   - rc_bench/schema.py
   - rc_bench/datasets/*
   - rc_bench/protocol/*
   - rc_bench/reservoirs/*
   - rc_bench/readout/*
   - rc_bench/metrics/*
   - rc_bench/runners/*
   - rc_bench/reporting/*
3. CLI и FastAPI/Celery должны стать тонкими оболочками над одним и тем же core API.
4. Исправь packaging/imports так, чтобы проект нормально устанавливался как пакет и не зависел от src.*-импортов.
5. Синхронизируй требования Python в README, pyproject и runtime.
6. В конце этапа должен быть один минимальный smoke test: установить пакет, запустить один experiment spec на NARMA10 + ESN и получить валидный result JSON.

Этап 2. Формализовать контракты данных и результатов.
1. Введи pydantic-модели:
   - ExperimentSpec
   - DatasetSpec
   - ReservoirSpec
   - ProtocolSpec
   - ReadoutSpec
   - ResultSpec
   - MetricsSpec
2. В ExperimentSpec должны быть как минимум:
   - dataset.name
   - dataset.params
   - reservoir.type
   - reservoir.params
   - protocol.split
   - protocol.washout
   - protocol.forecasting_mode
   - protocol.multi_seed
   - protocol.hpo_budget
   - readout.type
   - readout.params
   - random_seed / seed_list
3. ResultSpec должен содержать:
   - status
   - config hash
   - per-seed metrics
   - aggregated metrics
   - timings
   - memory stats
   - manifest / environment info
   - artifact paths
4. Убери любые ad hoc JSON/CSV схемы, которые не соответствуют ResultSpec.

Этап 3. Унифицировать datasets.
1. Перенеси генерацию данных в rc_bench/datasets.
2. Сначала поддержи:
   - NARMA10
   - NARMA30
   - Mackey-Glass
   - Lorenz-63
3. Для каждого датасета зафиксируй:
   - canonical parameters
   - split convention
   - washout convention
   - target alignment
   - supported forecasting modes
4. Реализуй golden tests:
   - NARMA10 и NARMA30: сравнение первых K значений на фиксированном input sequence
   - Mackey-Glass и Lorenz-63: фиксированный solver + dt + checkpoint short segment after transient
5. Дальше как optional extensions подготовь каркас для:
   - Kuramoto-Sivashinsky
   - one spiking/event benchmark
   - one real multivariate time-series benchmark

Этап 4. Унифицировать protocol layer.
1. Вынеси protocol как отдельный модуль.
2. Поддержи три режима:
   - one-step teacher forcing
   - fixed-horizon multi-step
   - closed-loop rollout
3. Явно опиши и реализуй:
   - train/val/test contiguous split
   - washout exclusion
   - state collection window
   - readout fit window
   - metric evaluation window
4. Добавь protocol conformance tests:
   - маленькая synthetic task, где teacher forcing и closed-loop дают заведомо разные результаты
   - проверка, что reported mode совпадает с фактическим режимом расчёта

Этап 5. Унифицировать метрики.
1. Сделай один модуль metrics.
2. Базовые метрики:
   - mse
   - rmse
   - mae
   - nrmse_range
3. Дополнительные:
   - nrmse_std
   - nrmse_var
   - prediction_horizon / valid prediction time
   - trajectory_error
4. Cost metrics:
   - train_time
   - inference_latency
   - throughput
   - peak_memory
   - energy_proxy
5. Не оставляй двусмысленного названия nrmse. Если есть несколько нормировок, названия должны быть явные.
6. Добавь deterministic metric tests на hand-crafted arrays, где точные значения известны заранее.

Этап 6. Интегрировать существующие reservoir'ы в общий интерфейс.
1. Сделай единый интерфейс для reservoir modules, например:
   - fit_reservoir(...)
   - transform(...)
   - fit_transform(...)
   - reset_state(...)
2. Под этот интерфейс интегрируй:
   - ESN
   - LSM
   - LogisticMapRC
   - FitzHughNagumoRC
3. Убери модельные run.py-скрипты как отдельные несвязанные пайплайны; если они нужны, пусть становятся thin wrappers над core runners.
4. Все reservoir types должны запускаться и через CLI, и через worker, и возвращать одинаковый ResultSpec.
5. Для каждой архитектуры добавь sanity checks:
   - ESN: bounded states, spectral radius sanity, leak/sr sensitivity smoke test
   - LSM: spiking occurs, no dead reservoir, stable state extraction dimensionality
   - Logistic map: chaotic regime sanity, correct input mixing
   - FHN: stable numerical integration, bounded oscillatory regime, single oscillator smoke test

Этап 7. Добавить новые архитектуры.
1. Leaky-ESN — обязательно.
   - Реализуй explicit leak parameter как canonical ESN variant.
   - Документируй update convention.
   - Добавь HPO grid по leak rate.
   - Сравни с текущим ESN при одинаковом readout/HPO budget.
2. DeepESN — обязательно.
   - Реализуй stacked reservoirs без обучения внутренних слоёв.
   - Поддержи concat-to-readout и last-layer-to-readout как явные режимы.
   - Поддержи per-layer leak, per-layer spectral scaling.
   - Не добавляй trainable inter-layer mappings в core variant.
3. QRC — ограниченно.
   - Сделай minimal simulated QRC baseline / stub.
   - Зафиксируй одну reference formulation.
   - Не заявляй quantum advantage.
   - Если полноценно не помещается, оформи как experimental/future-work module, но с единым интерфейсом.

Этап 8. Ввести единый readout и честный HPO.
1. Для regression core protocol используй ridge regression.
2. Для classification используй regularized linear classifier.
3. Запрети нелинейные readout'ы в core benchmark.
4. Введи одинаковый HPO budget для всех архитектур:
   - одинаковое число trials
   - одинаковый split usage
   - одинаковая схема val selection
5. Введи multi-seed как обязательный режим:
   - минимум 5 seeds
   - агрегация mean ± std
   - опционально CI/bootstrap

Этап 9. Сделать reporting и reproducibility first-class частью стенда.
1. Каждый запуск должен сохранять:
   - full spec
   - resolved hyperparameters
   - seed list
   - package versions
   - platform info
   - artifact manifest
2. Добавь агрегированные отчёты:
   - per-seed table
   - aggregated table
   - best config table
   - quality-vs-cost summary
3. Добавь export в JSON + CSV + markdown report.
4. Подготовь reproducibility command:
   - rerun from saved manifest/spec

Этап 10. Обновить CLI и worker.
1. CLI должен уметь:
   - list datasets
   - list reservoirs
   - validate spec
   - run one experiment
   - run multi-seed experiment
   - aggregate results
2. Worker/FastAPI должны уметь тот же ExperimentSpec без специальных веток под ESN only.
3. Убери места, где reservoir_type == "esn" зашит как единственный supported case.
4. Добавь async-safe result lifecycle:
   - pending
   - running
   - completed
   - failed
   - artifact paths

Этап 11. Документация.
1. Полностью перепиши README так, чтобы он отражал реальное состояние проекта.
2. Добавь:
   - architecture overview
   - benchmark protocol
   - metric definitions
   - dataset definitions
   - reservoir definitions
   - examples of ExperimentSpec
   - reproducibility guide
3. Отдельно сделай docs/benchmark_protocol.md и docs/validation.md.

Этап 12. Минимальный acceptance suite.
Считай задачу выполненной только если проходят следующие acceptance criteria:
1. Один и тот же ExperimentSpec запускается через CLI и worker без изменения логики.
2. Все четыре текущие архитектуры работают через единый runner API.
3. Leaky-ESN и DeepESN добавлены и покрыты smoke tests.
4. NARMA10, NARMA30, Mackey-Glass, Lorenz-63 реализованы и покрыты generator tests.
5. Метрики считаются из одного модуля и имеют единые имена.
6. Multi-seed aggregation работает.
7. Есть хотя бы один end-to-end benchmark report по:
   - NARMA10
   - Mackey-Glass или Lorenz-63
   - ESN vs leaky-ESN vs LSM vs logistic vs FHN
8. README не врёт про поддерживаемые архитектуры, Python version и datasets.

Формат работы:
- Делай изменения небольшими PR-логическими пачками, но в одной ветке можешь группировать связанные шаги.
- После каждого крупного этапа обновляй changelog.
- Если где-то придётся выбрать между backward compatibility и чистым benchmark design, выбирай чистый benchmark design, но оставляй migration note.

Приоритеты:
P0:
- unify core library
- fix packaging/imports/version mismatch
- unify datasets/metrics/protocol/results
- integrate LSM/logistic/FHN into worker/common runner
- add multi-seed
P1:
- add NARMA30, MG, Lorenz-63
- add leaky-ESN
- add DeepESN
- add cost metrics and reporting
P2:
- add KS / event benchmark / real multivariate dataset
- add QRC stub
- add hardware-oriented energy profiling hooks

Сначала покажи proposed file tree and migration plan, затем приступай к реализации.