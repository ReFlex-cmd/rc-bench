# Состояние проекта rc_bench (post-audit, 2026-05-13)

## 1. Что реализовано и протестировано

| Компонент | Статус | Файл/папка | Тесты |
|---|---|---|---|
| 7 классов резервуаров (ESN, Leaky, Deep ESN, LSM, FHN, Logistic, QRC) | ✅ прошли аудит, исправлены | `src/rc_bench/core/reservoirs/` | `tests/test_reservoirs.py` (35 тестов) |
| Унифицированный протокол (washout, splits, forecasting modes) | ✅ | `src/rc_bench/protocol/` | `tests/test_protocol.py` |
| HPO (Optuna TPE + MedianPruner + feasibility gate) | ✅ + per-layer Deep ESN | `src/rc_bench/hpo/` | `tests/test_hpo.py` (20 тестов) |
| Multi-seed runner (sample std, ddof=1) | ✅ | `src/rc_bench/runners/multi_seed.py` | `tests/test_hpo.py::TestRunMultiSeed` |
| Unified Ridge readout с тюнингом α | ✅ | `src/rc_bench/readout/ridge.py` | — |
| RunRecord (lib_versions, hardware_profile, git_hash, hpo_convergence, diagnostics) | ✅ | `src/rc_bench/reporting/run_record.py` | `tests/test_reporting.py` |
| Plot generation | ✅ | `scripts/make_plots.py` | — |
| 4 датасета (NARMA-10/30, Mackey-Glass, Lorenz-63) | ✅ | `src/rc_bench/core/data_provider.py` | `tests/test_data_provider.py` |
| CLI (rcbench run, list-datasets, list-reservoirs, ...) | ✅ | `src/rc_bench/cli/app.py` | `tests/test_cli.py` |

**Итого тестов:** 289 passed (исключая `test_flow.py`, требующий Postgres).

## 2. Что сделано в этой итерации (2026-05-13)

См. `audit/01_implementations.md`, `audit/02_hpo_spaces.md`, `audit/03_protocol.md`. Кратко:

**Аудит:**
- Найдено 5 H + 12 M + 8 L расхождений в реализациях моделей.
- Найдено 6 H + 19 M расхождений в HPO-пространствах.
- Найдено 3 H + 8 M расхождений в протоколе.

**Исправления (применены):**
- ESN: `lr` убран из HPO (был превращающим в Leaky), `input_connectivity=1.0` (Jaeger-style).
- FHN: `dt=0.01` фиксирован, удалён из HPO; параметризация через `epsilon`; добавлены `a, b, coupling_strength` в HPO.
- LSM: добавлен рефрактерный период, расширены HPO-диапазоны (input_scale, w_rec_scale ↑, v_th ↓), дефолты сделаны spike-friendly.
- Logistic: convex combination → аддитивная форма, мягкий клип `[0, 1]`.
- Deep ESN: per-layer `sr_l, leak_rate_l` (custom Optuna suggest).
- HPO `readout_alpha`: `[1e-4, 10] → [1e-6, 1e+2]` (по результату smoke-теста).
- Sparsity diapazons: верхние границы 0.5 → 0.20.
- RunRecord обогащён `lib_versions`, `hardware_profile`, `hpo_convergence`, `hpo_diagnostics`.
- Protocol: `n_seeds` дефолт 1 → 10, `hpo_budget` 20 → 100; `multi_seed` `ddof=1`.
- Убран двойной `StandardScaler` в `data_provider`/`experiment_runner`.
- **Off-by-one в `closed_loop_predict`** (warmup-step + iter-step одинаковый input) — исправлено.
- **Feasibility gate в `_objective`**: `states_std < 1e-6` → `TrialPruned`. Без него LSM/FHN HPO часто залипали в режим без спайков/без активности.

**Эмпирический прогон:**
- Smoke-test (5 trials × 2 seeds × 19 ячеек) — 19/19 ОК после двух итераций фиксов.
- Полный прогон (100 trials × 10 seeds для P0/P1, 50 × 5 для P2/P3) — **18.4 минуты, 19/19 ОК**.
- 43 графика в `reports/plots/`.
- Сводка в `reports/empirical_summary.md`.

## 3. Что осталось как known limitation

### 3.1 Lorenz-63 closed-loop горизонт ≈ 0.26 λ_max·t

Гэп с Pathak et al. 2018 (~8 λ_max·t) объясняется отсутствием curriculum learning / noise injection / multi-output (x, y, z) Lorenz / большего N. Это **направление дальнейшего развития**, не дефект текущей системы. Подробно — `reports/empirical_summary.md` §4.1.

### 3.2 NARMA-30 со стабилизирующим β/3

Сознательное отклонение от Atiya & Parlos 2000, документировано в docstring `generate_narma30()`. Числа NARMA-30 не сравнимы напрямую с публикациями.

### 3.3 QRC = mean-field Ising

Зафиксировано в `audit/notes_qrc.md`. В тексте ВКР — называть «quantum-inspired reservoir» / «mean-field Ising surrogate».

### 3.4 Multi-seed mode 1

HPO один раз → 10 seeds на лучших. Variability гиперпараметров не учтена (mode 2 был бы в 10× дороже). Зафиксировано в `audit/03_protocol.md` §3.3.3.

### 3.5 Размерности резервуара не унифицированы (LSM/FHN меньше)

LSM=400, FHN=200 vs ESN/Leaky=300, Logistic=500. Причина — стоимость Python-цикла. Зафиксировано в `audit/04_out_of_scope.md` OOS-1.

### 3.6 Двойное scaling X решено, но `data_provider.scaler_name` остался как backward-compat noop

Параметр игнорируется, scaling делает только `experiment_runner` по `DEFAULT_SCALER` модели. Documented в `data_provider.py`.

### 3.7 Reservoirpy ESN — обёртка, не нативная реализация

ESN использует `reservoirpy.nodes.Reservoir`. Формулы в библиотеке. Версия зафиксирована в `RunRecord.lib_versions` для воспроизводимости. Если в ВКР нужна полная прозрачность — переписать на чистый numpy (есть LeakyESN как образец, ~70 строк).

## 4. TODO для главы 4 ВКР

Для каждой модели в разделе 4.3 «Реализованные модели»:

- **ESN** (1 раздел): математическая формула Jaeger-style, упомянуть reservoirpy v0.4.1, ссылка Jaeger 2001.
- **Leaky ESN** (1 раздел): уравнение с α, объяснить смысл leak rate, ссылка Jaeger & Lukoševičius 2009.
- **Deep ESN** (1 раздел): стек слоёв, per-layer hyperparameters, `n_layers ∈ [2,5]`, ссылка Gallicchio & Micheli 2017.
- **LSM** (1 раздел): LIF с рефрактером, синаптический след, direct-current encoding, ссылки Maass 2002, Verstraeten 2007.
- **FHN** (1 раздел): двумерная система, RK4, режим возбудимости (a=0.7, b=0.8, ε≈0.08), ссылки FitzHugh 1961, Nagumo 1962.
- **Logistic** (1 раздел): аддитивная форма, диверсифицированные `r_i ∈ [3.7, 3.99]`, без сетевой связи (как known limitation).
- **QRC** (1 раздел): обязательно дисклеймер «mean-field Ising surrogate», virtual nodes (Fujii & Nakajima 2017).

Раздел 4.4 «Датасеты и протокол генерации»:
- NARMA-10 (canonical Atiya & Parlos), NARMA-30 (с стабилизирующим β/3 — явно указать!), Mackey-Glass τ=17, Lorenz-63 (RK4, dt=0.01, subsample=10, x-component).

Раздел 4.5 «Унифицированный протокол сравнения»:
- Wringe et al. 2024 6 критериев + наша таблица соответствия (см. `audit/03_protocol.md` §3.1).
- Multi-seed mode 1 как осознанный выбор.

Раздел 4.6 «Результаты»:
- Использовать таблицу из `reports/empirical_summary.md` §1.
- Графики `nrmse_by_task_<task>.png`, `hpo_convergence_*.png`, `seeds_distribution_*.png`, `prediction_horizon_lorenz63.png`.
- Раздел про Lorenz-63 — обязательно с дисклеймером про exposure bias и отличие от Pathak setup.

## 5. Полезные команды

```bash
# Полный прогон (~18 минут)
python scripts/full_grid.py

# Перегенерация графиков (~10 секунд)
python scripts/make_plots.py

# Smoke (~1 минута, 19 ячеек)
python scripts/smoke_grid.py

# Полная батарея тестов
.venv/bin/python -m pytest tests/ --ignore=tests/test_flow.py -q
```

## 6. Метаданные

- Branch: `main`
- Test suite: 289 passed (без DB-зависимого test_flow)
- Last commit before this iteration: `10701ec`
- Изменённые файлы (для последующего разделения на коммиты): см. финальное сообщение Claude в чате 2026-05-13.
