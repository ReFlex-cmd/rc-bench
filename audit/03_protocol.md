# Этап 3. Аудит протокола эксперимента

Дата: 2026-05-13
Файлы:
- `src/rc_bench/core/schema.py` — `ExperimentSpec`, `ProtocolSpec`, `RunRecord`
- `src/rc_bench/runners/experiment_runner.py` — основной runner
- `src/rc_bench/runners/multi_seed.py` — multi-seed обёртка
- `src/rc_bench/protocol/{splitter,state_collector,forecasting}.py`
- `src/rc_bench/reporting/run_record.py`

Severity: **H** / **M** / **L** — как раньше.

---

## 3.1 Соответствие 6 критериям Wringe et al. 2024

| # | Критерий | Статус | Комментарий |
|---|---|---|---|
| 1 | Равный HPO-бюджет | ⚠️ partial | `hpo_budget` задаётся per-experiment, но дефолт = 20, ТЗ требует 50–100. Технически унификация на стороне пользователя. **Действие:** дефолт → 100, документировать. |
| 2 | Единый Ridge readout | ✅ | `Ridge(alpha=…, fit_intercept=True)` через sklearn для всех моделей; α тюнится в HPO. |
| 3 | Multi-seed контроль | ⚠️ partial | `MultiSeedResult` есть; `ProtocolSpec.n_seeds` дефолт = 1. ТЗ требует ≥ 5 (целевое 10). **Действие:** дефолт → 10. |
| 4 | Единые разбиения | ✅ | `TimeSeriesSplitter(train=0.6, val=0.2)` + `washout` в `ProtocolSpec`. Forecasting mode явный. |
| 5 | Единые метрики с явной нормировкой | ✅ | `nrmse_range`, `nrmse_std`, `nrmse_var` все три считаются — пользователь выбирает в отчёте. Основная метрика HPO — `nrmse_range`. |
| 6 | Полное логирование | ⚠️ partial | `RunRecord` содержит `git_hash, hostname, python_version, timestamp, rc_bench_version`. **Нет:** `lib_versions` (numpy, scipy, sklearn, optuna, reservoirpy), **нет** `hardware_profile` (cpu_model, n_cores, ram_gb), **нет** `hpo_diagnostics` (n_pruned, n_completed). |

---

## 3.2 ExperimentSpec / ProtocolSpec

```python
class ProtocolSpec(BaseModel):
    washout: int = 200
    train_frac: float = 0.6
    val_frac: float = 0.2
    forecasting_mode: Literal["one_step", "fixed_horizon", "closed_loop"] = "one_step"
    horizon: int = 1
    use_hpo: bool = False
    hpo_budget: int = 20
    n_seeds: int = 1
```

| # | S | Находка | Исправление |
|---|---|---|---|
| 3.2.1 | M | Дефолт `washout=200`. Для NARMA-10 это ≥ 10× memory horizon — **OK**. Для NARMA-30 — `washout=200` это ~7× — допустимо. Для Mackey-Glass `τ=17` — норма. Для Lorenz-63 (closed-loop) washout — это seed window. **OK для всех задач.** | Документировать в README/ВКР. |
| 3.2.2 | M | Дефолт `n_seeds=1`. ТЗ §3: «целевое 10, минимум 5». | `n_seeds: int = 10` (или 5 для feasibility, 10 для P0/P1). |
| 3.2.3 | M | Дефолт `hpo_budget=20`. ТЗ §2: 50–100. | `hpo_budget: int = 100` (P0/P1) / 50 (P2/P3 в коде проще задавать вручную). |
| 3.2.4 | — | `forecasting_mode` поддерживает `closed_loop`. Реализован в `forecasting.py:closed_loop_predict`. **OK** для решения по Q3 (Lorenz-63). | — |
| 3.2.5 | L | `train_frac + val_frac < 1.0` валидация в splitter. test_frac неявный. **OK.** | — |
| 3.2.6 | M | `use_hpo: bool = False`. CLI/runner должен явно включать. ТЗ предполагает HPO **всегда** для финальных ячеек. | Зафиксировать в pipeline что для P0/P1 `use_hpo=True` обязательно. |

---

## 3.3 Multi-seed

`runners/multi_seed.py:run_multi_seed`:
- Использует `seeds = [spec.seed + i for i in range(n_seeds)]` — детерминистично.
- Только `seed` варьируется; данные, гиперпараметры, splits — фиксированы.
- Возвращает `mean ± std` (population std, `ddof=0`) по всем `MetricsResult` полям.

| # | S | Находка | Исправление |
|---|---|---|---|
| 3.3.1 | — | Логика корректная: на каждой итерации создаётся новая `reservoir` с новым `seed`. Run использует тот же `data` и `spec`. **OK.** | — |
| 3.3.2 | M | `std` использует `ddof=0` (population). Для отчёта n_seeds=10 разница с `ddof=1` (sample) минимальна (~5%), но **методологически в литературе чаще sample std (ddof=1)** для оценки variability эстиматора. | Поменять на `ddof=1`. Документировать в `audit/03_protocol.md`. |
| 3.3.3 | M | Multi-seed запускается **после** HPO с фиксированными гиперпараметрами. Это «mode 1» из Wringe et al. (HPO один раз → N seeds на лучших). Альтернатива «mode 2» (HPO для каждого seed отдельно) — методологически правильнее, но в N раз дороже. | Зафиксировать как «mode 1, justified by HPO cost». В тексте ВКР упомянуть, что вариативность гиперпараметров → variability оценки **не учтена**. |
| 3.3.4 | L | `metrics_per_seed` сохраняется полным списком — это позволяет потом построить boxplot (ТЗ §5.2). **OK.** | — |

---

## 3.4 RunRecord — полнота логирования

```python
class RunRecord(BaseModel):
    spec: ExperimentSpec
    result: ResultSpec
    timestamp: str = ""
    git_hash: Optional[str] = None
    hostname: str = ""
    python_version: str = ""
    rc_bench_version: str = "0.1.0"
```

| # | S | Находка | Исправление |
|---|---|---|---|
| 3.4.1 | **H** | **Нет** `lib_versions`. ТЗ §3 критерий 6 явно требует «параметры, версии библиотек, git hash, hardware profile». | Добавить `lib_versions: Dict[str, str] = {}`. Заполнять в `RunRecord.make()` через `importlib.metadata.version("numpy")` и т.д. для `numpy, scipy, scikit-learn, optuna, reservoirpy, pydantic`. |
| 3.4.2 | **H** | **Нет** `hardware_profile`. См. Q4 (решение пользователя — да, добавить). | Добавить `hardware_profile: Dict[str, Any] = {}`. Заполнять из `psutil` (`cpu_count`, `virtual_memory().total`) и `platform.processor()` / `/proc/cpuinfo`. |
| 3.4.3 | M | **Нет** `hpo_diagnostics`. См. 2.0.5. | Добавить `hpo_diagnostics: Optional[Dict[str, int]]` с полями `n_trials, n_completed, n_pruned, n_failed, best_trial_number`. Заполнять в `tuner.run_hpo` и пробрасывать через `MultiSeedResult` или новое поле `ResultSpec`. |
| 3.4.4 | L | `rc_bench_version` хардкод "0.1.0". | Читать из `pyproject.toml` через `importlib.metadata.version("rc-bench")`. |
| 3.4.5 | — | `git_hash` через `subprocess.run('git rev-parse --short HEAD')` с timeout=2 — **OK.** | — |
| 3.4.6 | — | `timestamp` UTC ISO — **OK.** | — |

---

## 3.5 Forecasting protocols

`protocol/forecasting.py`:
- `one_step` — identity alignment H[t]→y[t]. **OK.**
- `fixed_horizon` — H[t]→y[t+h], корректная синхронизация. **OK.**
- `closed_loop` — обучается teacher-forced (как one_step), при инференсе — `closed_loop_predict` с `warmup` на seed window и feedback. **OK.**

| # | S | Находка | Исправление |
|---|---|---|---|
| 3.5.1 | — | Все три режима реализованы корректно. **OK.** | — |
| 3.5.2 | M | `closed_loop_predict` использует `washout` как seed window (`X_test_seed = X_test_s[:washout]`). При `washout=200` это даёт большое окно для прогрева. Для Lorenz-63 — норма. **OK.** | Зафиксировать в `audit/03_protocol.md`. |
| 3.5.3 | M | `closed_loop` feedback: `x_in = np.array([[y_t]])` — предполагается **скалярный** выход, который становится скалярным входом следующего шага. Это работает только для univariate задач. Если в будущем добавятся multi-variate datasets — расширить. | Out-of-scope для текущей задачи (зафиксировать). |
| 3.5.4 | L | `closed_loop` train identical to one_step — нет teacher-forcing schedule (curriculum). Это стандарт RC. **OK.** | — |

---

## 3.6 StateCollector / Splitter

`state_collector.py` — `collect()` применяет washout, `trim()` обрезает y. Используется в `experiment_runner` корректно (`collector.trim(y_train)` etc.).

`splitter.py` — `train_frac=0.6, val_frac=0.2` фиксированы; test_frac = 0.2 неявно.

| # | S | Находка | Исправление |
|---|---|---|---|
| 3.6.1 | — | Логика корректная. **OK.** | — |
| 3.6.2 | M | В `experiment_runner` сейчас **не используется** `TimeSeriesSplitter` — splits приходят уже готовыми из `data_provider.get_data_for_experiment` (см. `data_provider.py:153`), который дублирует логику splitter. Один lithic из двух мест. **Не противоречит результатам, но дублирование.** | Out-of-scope. |
| 3.6.3 | M | `data_provider` применяет `StandardScaler` к X (только X!). y **не нормируется**. Для метрики `nrmse_range` это OK (нормировка идёт внутри метрики). Но для NARMA-30 (диапазон y ~[-1, 1] но с тяжёлыми хвостами) и Mackey-Glass (y ~[0.4, 1.4]) непонятно, насколько y centered. | Документировать. NRMSE уже нормирована, поэтому корректно. |
| 3.6.4 | M | `experiment_runner._scale_inputs` повторно применяет `StandardScaler` если `spec.reservoir.params.scaler == "zscore"`, причём `data_provider` **уже** применил scaler. Это значит, что для моделей с `DEFAULT_SCALER='zscore'` (LSM, FHN, Logistic) X нормируется **дважды** (повторный StandardScaler на уже нормированных данных — идемпотентен с точностью до численных ошибок, но логически бессмысленно). | Проверить и убрать одно из двух мест. Скорее — убрать из `data_provider` (передавать raw, scaling — задача runner). Этот дубль — **technical debt, severity M** для прозрачности. |

---

## 3.7 ResultSpec / итоговая структура результата

```python
class ResultSpec(BaseModel):
    status: Literal["completed", "failed"]
    config_hash: str
    metrics: Optional[MetricsResult] = None
    multi_seed_result: Optional[MultiSeedResult] = None
    hpo_best_params: Optional[Dict[str, Any]] = None
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
```

| # | S | Находка | Исправление |
|---|---|---|---|
| 3.7.1 | — | Покрывает все нужные поля для отчёта. `hpo_best_params` есть. **OK.** | — |
| 3.7.2 | M | `multi_seed_result.metrics_per_seed` хранит полный список → boxplot OK. **OK.** | — |
| 3.7.3 | M | Нет поля `convergence_history` (HPO best vs trial). Для графика `hpo_convergence_<model>_<task>.png` (ТЗ §5.2) это нужно. | Добавить `hpo_convergence: Optional[List[float]]` (best_score после каждого trial). Заполнять в `tuner.run_hpo` через callback. |

---

## 3.8 Решение по `n_seeds` (фиксация для P0/P1/P2/P3)

Согласовано с ТЗ §4 «Прогон экспериментов»:
- **P0 (ESN, Leaky ESN):** 100 trials HPO + 10 seeds.
- **P1 (Deep ESN, LSM, FHN):** 100 trials HPO + 10 seeds.
- **P2 (Logistic):** 50 trials + 5 seeds.
- **P3 (QRC):** 50 trials + 5 seeds.

Корректировка по решению Q3 (Lorenz-63 closed-loop, только для ESN/Leaky/Deep ESN):
- Lorenz-63 запускается только для P0 (ESN, Leaky) + P1 (Deep ESN). 3 ячейки.
- Остальные ячейки задач из ТЗ §4 таблицы — без изменений.

---

## 3.9 Что итого править в schema.py / runners

Сводная таблица для этапа правок:

| # | Файл | Изменение |
|---|---|---|
| П1 | `schema.py:ProtocolSpec` | `n_seeds: int = 10` (default) |
| П2 | `schema.py:ProtocolSpec` | `hpo_budget: int = 100` (default) |
| П3 | `schema.py:ResultSpec` | Добавить `hpo_convergence: Optional[List[float]] = None` |
| П4 | `schema.py:ResultSpec` | Добавить `hpo_diagnostics: Optional[Dict[str, int]] = None` |
| П5 | `reporting/run_record.py:RunRecord` | Добавить `lib_versions: Dict[str, str]`, `hardware_profile: Dict[str, Any]` |
| П6 | `reporting/run_record.py:RunRecord.make` | Заполнить эти два поля через `importlib.metadata` + `psutil`/`platform` |
| П7 | `runners/multi_seed.py` | Заменить `ddof=0` на `ddof=1` (sample std) |
| П8 | `hpo/tuner.py` | Callback для `convergence_history`; счётчики pruned/completed/failed |
| П9 | `runners/experiment_runner.py` или `data_provider.py` | Убрать двойное масштабирование X (см. 3.6.4) |
| П10 | `runners/pipeline.py` (если есть) | Для Lorenz-63 переключать `forecasting_mode='closed_loop'` автоматически или в spec preset |

---

## 3.10 Нерешённые методологические вопросы (записаны в `05_open_questions.md`)

Все четыре вопроса этапа 1 решены пользователем 2026-05-13. Новых блокирующих вопросов из этапа 3 нет.

Возможные следующие вопросы (для обсуждения после прогонов):
- Q5. Использовать ли `RidgeCV` вместо ручного `select_alpha` (см. OOS-4)?
- Q6. Multi-seed mode 1 vs mode 2 (HPO один раз vs HPO на каждом seed)? — текущий выбор mode 1 зафиксирован выше.
