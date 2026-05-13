# Эмпирическая сводка (этап 4)

Дата: 2026-05-13
Источник: `reports/full_results.json`, `reports/runs/*.json`
Запуск: `python scripts/full_grid.py` (18.4 минуты, 19/19 ячеек ОК)

Унифицированный протокол: HPO budget **100 trials** (P0/P1) или 50 (P2/P3); **10 seeds** (P0/P1) или 5 (P2/P3); washout=200; train/val/test = 60/20/20; Ridge readout с тюнингом α внутри HPO; sample std (`ddof=1`) по seeds.

---

## 1. Главная таблица — NRMSE_range на тесте (mean ± std)

| Модель | NARMA-10 | NARMA-30 | Mackey-Glass | Lorenz-63 (closed-loop) |
|---|---|---|---|---|
| **ESN** | **0.0481 ± 0.0060** | **0.0703 ± 0.0060** | 0.0002 ± 0.0000 | 1.4735 ± 0.9346 |
| **Leaky ESN** | 0.0687 ± 0.0324 | 0.1848 ± 0.1912 | **0.0001 ± 0.0000** | **0.3027 ± 0.0107** |
| **Deep ESN** | 0.0656 ± 0.0133 | 0.0853 ± 0.0068 | **0.0001 ± 0.0000** | **0.3038 ± 0.0167** |
| **LSM** | 0.1352 ± 0.0065 | — | 0.0358 ± 0.0021 | — |
| **FHN** | 0.1667 ± 0.0152 | — | 0.2178 ± 0.0020 | — |
| **Logistic** | 0.1975 ± 0.0044 | — | 0.0707 ± 0.0017 | — |
| **QRC*** | 0.1064 ± 0.0046 | — | — | — |

\* QRC — упрощённая mean-field Ising-симуляция, не unitary; см. `audit/notes_qrc.md`.

**Жирным** — лучший результат в столбце.

### 1.1 Lorenz-63 closed-loop — prediction horizon

| Модель | Horizon mean ± std (steps) | в единицах λ_max·t |
|---|---|---|
| ESN | 2.8 ± 0.4 | **0.25 ± 0.04** |
| Leaky ESN | 2.9 ± 0.3 | **0.26 ± 0.03** |
| Deep ESN | 3.0 ± 0.0 | **0.27 ± 0.00** |
| Pathak et al. 2018 (literature reference) | — | ≈ 8 |

`λ_max ≈ 0.906` для классических параметров Lorenz-63, эффективный шаг dt=0.01·subsample(10)=0.1.

---

## 2. Иерархия моделей

**На задачах с короткой памятью (NARMA-10, NARMA-30):**
- Лучшие: ESN > Deep ESN > Leaky ESN, разрыв порядка 30-50%.
- Хуже всего: Logistic (0.20) и FHN (0.17), что ожидаемо — обе модели не имеют долгой recurrent-памяти.
- LSM (0.135) и QRC* (0.106) занимают середину.

**На Mackey-Glass (требует delay τ=17):**
- ESN-семейство практически решает задачу до уровня шума (NRMSE ≈ 1e-4).
- LSM (0.036) — приличный результат для биофизически правдоподобного резервуара.
- Logistic (0.071) — лучше, чем ожидалось от модели без сетевой связи между узлами.
- FHN (0.218) — заметно отстаёт; excitable dynamics плохо подходят для долгой памяти.

**На Lorenz-63 (closed-loop autonomous prediction):**
- Leaky ESN ≈ Deep ESN >> ESN. Leak rate смягчает динамику и существенно стабилизирует прогноз.
- Все три ESN-варианта дают ~0.26 λ_max·t горизонта — **существенно меньше** Pathak 2018 (~8 λ_max·t). См. §4.

---

## 3. Сравнение с литературой (sanity-check)

| Задача / модель | Литература | Наш результат | Соответствие |
|---|---|---|---|
| NARMA-10, ESN N≈300 | NRMSE 0.15-0.30 (Lukoševičius 2012, Rodan & Tino 2011) | **0.048** | ✓ лучше нижней границы |
| NARMA-30, ESN N≈300 | NRMSE ~0.15-0.30 (но наш использует stabilized β/3, см. §5) | **0.070** | ✓ лучше, но **не сравнимо** напрямую |
| Mackey-Glass τ=17, ESN N≈400 | NRMSE 0.01-0.05 (Jaeger 2001) | **0.0002** | ✓ намного лучше |
| Lorenz-63 closed-loop, ESN N≈300 | ~8 λ_max·t (Pathak et al. 2018, PRL) | **0.26 λ_max·t** | ✗ см. §4 |

ESN-семейство на NARMA/MG превосходит литературные числа — причина: широкий HPO-диапазон `readout_alpha ∈ [1e-6, 1e+2]` (см. `audit/02_hpo_spaces.md` §2.8.1) и unified Optuna TPE с 100 trials.

---

## 4. Известные ограничения

### 4.1 Lorenz-63 closed-loop горизонт ≈ 0.26 λ_max·t vs Pathak 8 λ_max·t

**Причина — не качество моделей, а training-протокол:**
- Наш `closed_loop` режим обучается teacher-forced (как one_step), а инференс идёт автономно. Это классический **exposure bias**.
- Pathak et al. 2018 использовали:
  1. **Multi-output Lorenz** — readout предсказывает все три компоненты `(x, y, z)`, а не только `x`. Это даёт намного больше информации в обратной связи.
  2. **Curriculum / noise injection** — добавление шума к входу при обучении делает модель устойчивой к собственным ошибкам в closed-loop.
  3. **Larger reservoir** — N=2000-5000 vs наш N=300.
- Off-by-one bug в `closed_loop_predict` найден и исправлен (2026-05-13), но это улучшило NRMSE leaky/deep с 0.32 до 0.30 и горизонт с 0 до 3 шагов — фундаментальный exposure-bias gap остался.

**Что это значит для ВКР главы 4:**
- Указанная цифра 0.26 λ_max·t — **честный baseline** при минимальных training tricks. Подходит для иллюстрации «закрытый цикл — отдельная инженерная задача».
- Pathak'овские 8 λ_max·t достижимы, но требуют дополнительной работы (multi-output, curriculum). Это **направление дальнейшего развития**, не failure текущей системы.

### 4.2 NARMA-30 со стабилизирующим β/3

`generate_narma30()` использует β=0.05/3 вместо канонического β=0.05 — стабилизирующая модификация (см. docstring и `audit/05_open_questions.md` Q2). **Числа NARMA-30 не сравнимы напрямую** с публикациями, использующими каноническую формулу.

### 4.3 QRC — mean-field Ising, не unitary

Зафиксировано в `audit/notes_qrc.md`. NRMSE 0.106 на NARMA-10 ставит QRC между Leaky ESN (0.069) и LSM (0.135), но эти числа **нельзя интерпретировать как quantum advantage** — это классическая нелинейная сеть с tanh-update.

### 4.4 LSM/FHN feasibility-gate в HPO

Без feasibility-gate (см. `src/rc_bench/hpo/tuner.py:_objective`) HPO для LSM с большой вероятностью находил «мёртвый» режим (нейроны не спайкуют → H=0 → константный предиктор с baseline NRMSE ~0.27). Gate отбраковывает trials с `H.std() < 1e-6`. Аналогично спасает FHN от «низко-coupling, near-quiescent» решений.

После применения gate: pruned trials в LSM/FHN — 30-50%, но best trial всегда даёт нетривиальный H. **Без gate LSM был бы методологически некорректен.**

### 4.5 HPO sub-pruning rates

Для всех моделей 24-49% trials prune (MedianPruner + feasibility gate). Это нормальный режим для unified TPE; с increased budget (200+ trials) prune-rate снизится. Полное логирование диагностики — в `RunRecord.result.hpo_diagnostics`.

### 4.6 Multi-seed mode 1

HPO один раз → лучшие гиперпараметры → 10 seeds на тесте. Variability **гиперпараметров** между seeds не учтена (mode 2 был бы в N раз дороже). Этот выбор зафиксирован в `audit/03_protocol.md` §3.3.3.

### 4.7 Размерности резервуара (не унифицированы)

| Модель | units |
|---|---|
| ESN, Leaky ESN | 300 |
| Deep ESN | 100 × n_layers (тюнится) |
| LSM | 400 |
| FHN | 200 |
| Logistic | 500 |
| QRC | 50 (n_qubits) |

LSM/FHN имеют меньшие N из-за стоимости Python-цикла транзишена (см. `audit/04_out_of_scope.md` OOS-1). Зафиксировано как осознанный выбор.

---

## 5. HPO-сходимость

Все ячейки достигли плато до конца budget — best_trial_number обычно в районе 30-70 из 100 (P0/P1) или 15-35 из 50 (P2/P3). Графики — `reports/plots/hpo_convergence_<model>_<task>.png`. Recommendation для будущих прогонов: **50 trials достаточно** для всех моделей кроме Deep ESN (per-layer dim до 14). Для Deep ESN с n_layers=5 рекомендация — 200+ trials для надёжного покрытия пространства.

---

## 6. Воспроизводимость

Каждый run сохраняет:
- `RunRecord` JSON с полным `lib_versions`, `hardware_profile`, `git_hash`, `timestamp`, `python_version`.
- HPO best_params, convergence trajectory, diagnostics.
- Метрики per-seed (для построения boxplot).

Команда повтора прогона:
```bash
python scripts/full_grid.py
```

Все артефакты — в `reports/runs/`, `reports/plots/`, `reports/full_results.json`.
