# Полное соответствие репозитория описанию проекта (03_Project_Description_rc-bench.pdf)

> **For agentic workers:** REQUIRED SUB-SKILL: используйте `superpowers:subagent-driven-development`
> (рекомендуется) или `superpowers:executing-plans`, чтобы исполнять план задача за задачей.
> Шаги размечены чекбоксами (`- [ ]`).

**Goal:** привести репозиторий `rc-bench` в состояние, при котором каждое заявление
описания проекта (PDF, JMLC 2026) либо подтверждается артефактом в репозитории, либо явно
помечено как planned/unavailable — без деклараций, за которыми ничего не стоит.

**Architecture:** три новых профилировочных модуля (`energy.py`, `activity.py`,
`selection.py`) подключаются к существующему проходу `run_profiling_pass` и evidence-гейту;
режим `best_effort` добавляется как поле протокола, попадающее в каждый RunRecord, и как
второй прогон матрицы рядом с `fair`; сервисный контур и обложка репозитория доводятся до
того, что о них говорит PDF.

**Tech Stack:** Python 3.12, Poetry, pydantic v2, numpy/scipy, reservoirpy, Optuna,
matplotlib, pytest, FastAPI + Celery + Redis + PostgreSQL, Docker Compose, GitHub Actions,
Linux powercap/RAPL (`/sys/class/powercap/intel-rapl:*`).

## Global Constraints

- Рабочая ветка — `dev` (DEC-001). `main` напрямую не менять, force push запрещён, push
  только по запросу человека.
- `bash scripts/verify.sh quick` зелёный перед каждым коммитом; `bash scripts/verify.sh release`
  зелёный перед PR.
- TDD: сначала падающий тест, потом минимальная реализация. Тесты self-contained
  (`-m "not integration"` не требует PostgreSQL/Redis).
- Разделение honest/planned: реализованное, запланированное и недоступное нигде не
  смешиваются (DEC-007, требование §2 и §8 PDF).
- Прокси-метрики активности **никогда** не публикуются как энергия и не пересчитываются в
  джоули через TDP (§5 PDF, DEC-007).
- В evidence-бандл не попадают hostname, домашние пути, сырой датасет (DEC-008/DEC-016);
  release-гейт проверяет это `grep`, а не `rg` (см. `9761c05`).
- Cross-cutting файлы (`core/schema.py`, `runners/pipeline.py`, `reporting/run_record.py`,
  `docs/agent/BACKLOG.md`, `docs/DECISIONS.md`) меняет только основной агент
  (`docs/agent/AGENT_WORKFLOW.md`).
- Каждое число в README/бандле ссылается на артефакт; таблицы, отредактированные руками,
  не проходят `scripts/validate_evidence.py`.

---

## Результат сверки: PDF ↔ репозиторий (25 июля 2026, `dev` @ `de6bd3a`)

Подтверждено без доработок:

| Заявление PDF | Чем подтверждается |
|---|---|
| §4: семь архитектур (ESN, Leaky ESN, Deep ESN, LSM, FHN, Logistic, QRC mean-field) | `src/rc_bench/core/reservoirs/*_service.py`, `registry.py` |
| §4: QRC — классическая упрощённая симуляция, не квантовый резервуар | `qrc_service.py`, README §Результаты, `empirical_summary.md` §1 |
| §3: NARMA-10, стабилизированная NARMA-30, Mackey-Glass, Lorenz-63; split по времени | `core/data_provider.py`, `protocol/splitter.py` |
| §3: UCI, почасовой ряд, окно 12 000 ч, горизонты 1 и 24, каузальный препроцессинг | `data/jmlc.py`, `configs/jmlc/fair.yaml`, DEC-003/006/012 |
| §3: отдельный EDA (пропуски, выбросы, сезонность, ACF, сдвиг между split) | `reports/jmlc_2026/eda_report.md` (§Missingness, §outliers — Tukey 1.5 IQR: 223, §ACF, §split comparison) |
| §4: Ridge-выход, Optuna, агрегация по инициализациям | `readout/ridge.py`, `hpo/tuner.py`, `runners/multi_seed.py` |
| §4: запуск сохраняет конфиг, версии библиотек, git commit, параметры оборудования | `reporting/run_record.py` (`git_hash`, `lib_versions`, `python_version`), `profiling/hardware.py` |
| §4: CLI, YAML/JSON-спеки, отчёты в Markdown/CSV/PNG | `cli/app.py`, `configs/jmlc/*.yaml`, `reporting/report.py`, `reports/**` |
| §5: MSE, NRMSE, MAE, RMSE, MASE, improvement над baseline | `core/metrics.py`, `MetricsResult` |
| §5: baseline persistence, seasonal naive, Ridge AR | `core/baselines.py`, `runners/baseline_runner.py`, DEC-014/015 |
| §5: p50/p95 задержки, throughput, peak RSS, размер модели и рабочего состояния | `profiling/latency.py`, `memory.py`, `model_profiles.py`, `reports/jmlc_2026/profiles/` |
| §6: числа синтетической сетки (0.0481 / 0.0703 / ~1e-4 / 0.30 против 1.47, 19 ячеек, 0.25–0.27 λt) | `reports/empirical_summary.md` §1, §1.1 |
| §8: строка «Инженерия» — CI, CLI smoke, demo-config | `.github/workflows/ci.yml`, `scripts/verify.sh`, `configs/jmlc/demo.yaml` |
| §9: публичный репозиторий, AI_USAGE.md, Pareto качество-задержка и качество-память | `github.com/ReFlex-cmd/rc-bench` (PUBLIC), `AI_USAGE.md`, `reports/jmlc_2026/plots/pareto_quality_{latency,memory}.png` |

Расхождения, которые закрывает этот план:

| # | Заявление PDF | Фактическое состояние | Задача |
|---|---|---|---|
| 1 | §9 «Публичный репозиторий»; README: бейдж MIT и ссылка `[MIT](LICENSE)` | Файла `LICENSE` нет, GitHub отдаёт `licenseInfo: null` | REPO-001 |
| 2 | §5 «При отсутствии аппаратного счётчика публикуются только прокси: число операций, разреженность состояния» | Не реализовано (PROXY-001 READY) | PROXY-001 |
| 3 | §5 «…а для LSM — число спайков и синаптических событий» | Не реализовано (PROXY-002 READY) | PROXY-002 |
| 4 | §5 «Edge-профиль: измеренная энергия через Intel RAPL…, mJ/inference, samples/J и energy-delay product» | `EnergyResult.status` захардкожен в `unavailable`; RAPL на машине **есть** (`intel-rapl:0`, `package-0`), но `energy_uj` читается только root | ENERGY-002, ENERGY-003 |
| 5 | §2 «Режим best-effort позволит отдельно оценить практически лучший результат каждой архитектуры»; §8 строка «Сравнение» | В коде только fair; слова `best_effort` нет нигде, кроме docstring про threadpoolctl | MODE-001, MODE-002 |
| 6 | §5 «Ресурсный профиль: **время обучения**, p50/p95…» | `train_time` есть в `MetricsResult`, но не выведено ни в `profiles/summary.json`, ни в `aggregates/matrix_table.*` | PROF-004 |
| 7 | §9 «единый отчёт, по которому можно найти Pareto-оптимальную архитектуру под ограничения устройства» | Есть графики Pareto, но нет ни выбора под ограничения, ни отчёта выбора | SELECT-001 |
| 8 | §4 «Сервисный контур построен на FastAPI, Celery, Redis и PostgreSQL» | `create_experiment` читает `experiment_data.reservoir.type` → падает на baseline-спеках; `list_experiments`/`compare` не фильтруют по `owner_id` | API-001 |
| 9 | §9 «Pareto-диаграммы качество-задержка, качество-память и **качество-энергия**» | Третьей диаграммы нет | ENERGY-003 (шаг 8) |
| 10 | §2 «Планируемые функции отделены от уже реализованных» | README не размечает planned/implemented явно; описание репозитория на GitHub устарело (перечисляет 4 модели, ни слова о платформе, реальных данных и fair-протоколе) | DOC-003 |
| 11 | §9 «воспроизводимость … каждый эксперимент фиксирует параметры запуска» | Раскладка бандла отличается от `docs/PROJECT_CONTRACT.md` (`specs/frozen/`, `specs/resolved/`, `hpo/`, `runs/` лежат внутри RunRecord) — расхождение известно, но не описано в самом контракте | DOC-003 (шаг 5) |

---

## Предусловие (выполняет человек, один раз)

Доступ к счётчику RAPL. Сейчас `/sys/class/powercap/intel-rapl:0/energy_uj` имеет режим
`0400 root:root` — это защита от side-channel атаки CVE-2020-8694, а не поломка.

```bash
# Вариант A (на время сессии, сбрасывается при перезагрузке):
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj \
               /sys/class/powercap/intel-rapl:0:0/energy_uj

# Вариант B (переживает перезагрузку):
echo 'SUBSYSTEM=="powercap", ACTION=="add", RUN+="/bin/chmod a+r /sys%p/energy_uj"' \
  | sudo tee /etc/udev/rules.d/99-rapl-readable.rules
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=powercap
```

Проверка: `cat /sys/class/powercap/intel-rapl:0/energy_uj` печатает число.

**Если доступ не выдан** — ENERGY-002/003 не блокируют остальной план: backend вернёт
`status: "unavailable"`, `reason: "intel-rapl counter is not readable by this user"`,
энергетические поля не появятся, а §5 PDF будет закрыт прокси-метриками (PROXY-001/002).
Это ровно та развилка, которую PDF описывает словами «при отсутствии аппаратного счётчика».

---

## Структура файлов

Создаются:

| Файл | Ответственность |
|---|---|
| `LICENSE` | Текст лицензии MIT |
| `src/rc_bench/profiling/energy.py` | Обнаружение и чтение RAPL-доменов, протокол энергоизмерения, `EnergyProfile` |
| `src/rc_bench/profiling/activity.py` | Прокси активности: разреженность состояния, аналитический счёт операций, спайковая статистика |
| `src/rc_bench/reporting/selection.py` | Выбор Pareto-оптимальной модели под ограничения устройства + отчёт `selection.md` |
| `configs/jmlc/best_effort.yaml` | Шаблон матрицы в режиме best-effort с индивидуальными HPO-бюджетами |
| `tests/test_energy.py`, `tests/test_activity.py`, `tests/test_selection.py`, `tests/test_repo_metadata.py`, `tests/test_api_ownership.py` | Тесты новых модулей |

Изменяются:

| Файл | Что меняется |
|---|---|
| `src/rc_bench/core/schema.py` | `ProtocolSpec.mode`, расширение `EnergyResult`, стабильность `config_hash()` |
| `src/rc_bench/core/reservoirs/base.py` | Необязательный хук `step_operation_counts()` |
| `src/rc_bench/core/reservoirs/{esn,leaky_esn,lsm,logistic}_service.py` | Реализация хука; в LSM — счётчики спайков |
| `src/rc_bench/profiling/model_profiles.py` | Подключение energy и activity в проход профилирования, `train_time` в summary |
| `src/rc_bench/profiling/hardware.py` | Блок energy: реальный backend вместо жёсткого `unavailable` |
| `src/rc_bench/runners/jmlc_matrix.py` | Режимы fair/best_effort, индивидуальные бюджеты |
| `src/rc_bench/reporting/evidence.py` | Mode-aware валидация, `train_time`, energy/прокси в агрегатах |
| `src/rc_bench/reporting/pareto.py` | Ось стоимости `energy` |
| `src/rc_bench/cli/app.py` | Команда `rcbench select` |
| `src/rc_bench/main.py` | Приём baseline-спеков, ownership-фильтрация |
| `scripts/validate_evidence.py`, `scripts/run_profiling.py`, `scripts/verify.sh` | Новые артефакты в гейте |
| `README.md`, `reports/jmlc_2026/README.md`, `AI_USAGE.md`, `docs/PROJECT_CONTRACT.md`, `docs/DECISIONS.md`, `docs/agent/BACKLOG.md` | Документация и решения |

---

### Task 1: REPO-001 — файл лицензии

**Files:**
- Create: `LICENSE`
- Create: `tests/test_repo_metadata.py`
- Modify: `README.md` (секция «Лицензия», строка 271)

**Interfaces:**
- Consumes: ничего.
- Produces: файл `LICENSE` в корне; на него ссылается README и (после релиза) GitHub.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_repo_metadata.py
"""Проверки обложки репозитория: то, что README обещает читателю, должно
существовать в дереве. Бейдж лицензии без файла лицензии — заявление без
артефакта, ровно то, что запрещает Definition of Done для документации."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_license_file_exists_and_is_mit():
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.is_file(), "README ссылается на LICENSE, файла нет"
    text = license_path.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "2026" in text
    assert "Dmitry Komarov" in text


def test_readme_license_links_resolve():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\((LICENSE[^)]*)\)", readme):
        assert (REPO_ROOT / target).is_file(), f"битая ссылка на {target}"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `poetry run pytest tests/test_repo_metadata.py -v`
Expected: FAIL — `README ссылается на LICENSE, файла нет`.

- [ ] **Step 3: Создать `LICENSE`**

Канонический текст MIT, первые строки:

```text
MIT License

Copyright (c) 2026 Dmitry Komarov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
```

(полный текст — https://opensource.org/license/mit, без сокращений)

- [ ] **Step 4: Прогнать тест**

Run: `poetry run pytest tests/test_repo_metadata.py -v`
Expected: 2 passed.

- [ ] **Step 5: Проверить, что README не расходится с файлом**

Run: `grep -n "MIT\|LICENSE" README.md`
Expected: бейдж и секция «Лицензия» говорят MIT и ссылаются на существующий `LICENSE`.
Если формулировка расходится — поправить README, а не файл лицензии.

- [ ] **Step 6: Зелёный гейт и коммит**

```bash
bash scripts/verify.sh quick
git add LICENSE tests/test_repo_metadata.py README.md
git commit -m "chore(repo): add the MIT license file the README has been claiming"
```

---

### Task 2: PROXY-001 — число операций и разреженность состояния

**Files:**
- Create: `src/rc_bench/profiling/activity.py`
- Create: `tests/test_activity.py`
- Modify: `src/rc_bench/core/reservoirs/base.py`
- Modify: `src/rc_bench/core/reservoirs/esn_service.py`, `leaky_esn_service.py`, `lsm_service.py`, `logistic_service.py`

**Interfaces:**
- Consumes: `BaseReservoir` (`step`, `reset_state`), `RidgeReadout.coefficients() -> (np.ndarray, float)`.
- Produces:
  - `BaseReservoir.step_operation_counts() -> Optional[Dict[str, int]]` — базовая реализация возвращает `None`;
  - `activity.state_sparsity(h: np.ndarray, *, near_zero_atol: float = 1e-6) -> Dict[str, float]`;
  - `activity.count_step_operations(reservoir: BaseReservoir, *, readout_units: int) -> Dict[str, Any]`;
  - ключ `"activity"` в словаре ячейки профиля (используется Task 5 и Task 8).

- [ ] **Step 1: Написать падающие тесты разреженности и счёта операций**

```python
# tests/test_activity.py
"""Прокси активности — аналитические оценки, а не измерения. Тесты фиксируют
именно это: числа должны выводиться из формы весов и состояния, быть
воспроизводимыми и никогда не превращаться в джоули."""
import numpy as np
import pytest

from rc_bench.core.reservoirs.registry import get_reservoir
from rc_bench.profiling.activity import count_step_operations, state_sparsity


def test_state_sparsity_counts_exact_and_near_zeros():
    h = np.array([0.0, 0.0, 1e-9, 0.5, -0.25])
    stats = state_sparsity(h, near_zero_atol=1e-6)
    assert stats["n_units"] == 5
    assert stats["zero_fraction"] == pytest.approx(2 / 5)
    assert stats["near_zero_fraction"] == pytest.approx(3 / 5)
    assert stats["mean_abs"] == pytest.approx((0.0 + 0.0 + 1e-9 + 0.5 + 0.25) / 5)


def test_state_sparsity_rejects_empty_state():
    with pytest.raises(ValueError):
        state_sparsity(np.array([]))


def test_leaky_esn_operation_count_matches_dense_arithmetic():
    reservoir = get_reservoir("leaky_esn", {"units": 50, "density": 0.1, "seed": 7})
    counts = count_step_operations(reservoir, readout_units=50)
    nnz = int(np.count_nonzero(reservoir._W_rec))
    # W_rec @ x (nnz MAC) + W_in * u (units MAC) + leak-смешение (2*units)
    assert counts["backend"] == "analytic"
    assert counts["reservoir_macs"] == nnz + 50
    assert counts["reservoir_nonlinearities"] == 50
    assert counts["readout_macs"] == 50
    assert counts["total_macs"] == nnz + 50 + 50


def test_logistic_operation_count_has_no_recurrent_matrix():
    reservoir = get_reservoir("logistic", {"units": 32, "seed": 7})
    counts = count_step_operations(reservoir, readout_units=32)
    # Узлы не связаны между собой: рекуррентной матрицы нет вовсе.
    assert counts["reservoir_nonzero_recurrent_weights"] == 0
    assert counts["reservoir_macs"] == 3 * 32
    assert counts["total_macs"] == 3 * 32 + 32


def test_esn_operation_count_uses_sparse_nnz():
    reservoir = get_reservoir("esn", {"n_units": 40, "rc_connectivity": 0.2, "seed": 7})
    reservoir.reset_state()  # reservoirpy инициализирует веса лениво
    counts = count_step_operations(reservoir, readout_units=40)
    assert counts["reservoir_nonzero_recurrent_weights"] == reservoir._res.W.nnz
    assert counts["reservoir_macs"] > 0


def test_unknown_reservoir_reports_unavailable_instead_of_guessing():
    reservoir = get_reservoir("fhn", {"units": 20, "seed": 7})
    counts = count_step_operations(reservoir, readout_units=20)
    assert counts["backend"] == "unavailable"
    assert "not implemented" in counts["reason"]
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `poetry run pytest tests/test_activity.py -v`
Expected: FAIL — `ModuleNotFoundError: rc_bench.profiling.activity`.

- [ ] **Step 3: Добавить хук в базовый класс**

```python
# src/rc_bench/core/reservoirs/base.py — добавить в BaseReservoir
    def step_operation_counts(self) -> Optional[Dict[str, int]]:
        """Аналитическая стоимость одного ``step()`` в MAC-операциях.

        Возвращает ``None``, если модель не описала свою арифметику: лучше
        честное «недоступно», чем правдоподобная выдумка. Ключи:
        ``reservoir_macs`` (умножения-сложения обновления состояния),
        ``reservoir_nonlinearities`` (число вызовов нелинейности),
        ``reservoir_nonzero_recurrent_weights`` (ненулевые элементы W_rec).
        """
        return None
```

(добавить `Optional` в импорт `typing`)

- [ ] **Step 4: Реализовать хук в четырёх моделях real-data матрицы**

```python
# leaky_esn_service.py
    def step_operation_counts(self) -> Dict[str, int]:
        nnz = int(np.count_nonzero(self._W_rec))
        return {
            # W_rec @ x — по одному MAC на ненулевой вес; W_in * u — units MAC;
            # (1-a)*x + a*pre — ещё 2*units, учтены как смешение.
            "reservoir_macs": nnz + self._units,
            "reservoir_nonlinearities": self._units,
            "reservoir_nonzero_recurrent_weights": nnz,
        }

# logistic_service.py
    def step_operation_counts(self) -> Dict[str, int]:
        units = int(self._r.shape[0])
        return {
            # r*x*(1-x): 2 MAC на узел; + coupling*(w_in*u): ещё 1.
            "reservoir_macs": 3 * units,
            "reservoir_nonlinearities": 0,     # логистическое отображение — сама арифметика
            "reservoir_nonzero_recurrent_weights": 0,   # межузловых связей нет
        }

# lsm_service.py
    def step_operation_counts(self) -> Dict[str, int]:
        units = int(self._W_rec.shape[0])
        nnz = int(np.count_nonzero(self._W_rec))
        return {
            # W_rec @ s (nnz) + W_in*u (units) + мембранный распад (units)
            # + синаптический распад (units).
            "reservoir_macs": nnz + 3 * units,
            "reservoir_nonlinearities": units,  # пороговое сравнение на нейрон
            "reservoir_nonzero_recurrent_weights": nnz,
        }

# esn_service.py
    def step_operation_counts(self) -> Dict[str, int]:
        if not self._res.initialized:
            self._res.run(np.zeros((1, 1)))
        w_nnz = int(self._res.W.nnz)
        win_nnz = int(self._res.Win.nnz)
        units = int(self._res.output_dim)
        return {
            "reservoir_macs": w_nnz + win_nnz,
            "reservoir_nonlinearities": units,
            "reservoir_nonzero_recurrent_weights": w_nnz,
        }
```

- [ ] **Step 5: Реализовать `activity.py`**

```python
"""Прокси активности вычислений (PROXY-001/PROXY-002).

Это АНАЛИТИЧЕСКИЕ оценки: они выводятся из формы весов и наблюдаемого
состояния, а не измеряются счётчиком. Они не являются энергией и не
пересчитываются в джоули — ни через TDP, ни как-либо ещё (DEC-007, §5 PDF).
Публикуются рядом с энергией, но в отдельном блоке ``activity``.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rc_bench.core.reservoirs.base import BaseReservoir


def state_sparsity(h: np.ndarray, *, near_zero_atol: float = 1e-6) -> Dict[str, float]:
    """Разреженность вектора рабочего состояния.

    ``zero_fraction`` — доля точных нулей (для LSM это молчащие нейроны);
    ``near_zero_fraction`` — доля элементов с |h| < ``near_zero_atol``
    (для tanh-резервуаров точных нулей почти не бывает, а численно
    незначимые компоненты есть).
    """
    flat = np.asarray(h, dtype=float).reshape(-1)
    if flat.size == 0:
        raise ValueError("state vector is empty; nothing to characterise")
    abs_h = np.abs(flat)
    return {
        "n_units": int(flat.size),
        "zero_fraction": float(np.count_nonzero(abs_h == 0.0) / flat.size),
        "near_zero_fraction": float(np.count_nonzero(abs_h < near_zero_atol) / flat.size),
        "near_zero_atol": float(near_zero_atol),
        "mean_abs": float(abs_h.mean()),
        "max_abs": float(abs_h.max()),
    }


def count_step_operations(reservoir: BaseReservoir, *, readout_units: int) -> Dict[str, Any]:
    """Аналитическая стоимость одного шага вывода: резервуар + Ridge-выход."""
    counts = reservoir.step_operation_counts()
    if counts is None:
        return {
            "backend": "unavailable",
            "reason": (
                f"{type(reservoir).__name__}.step_operation_counts() is not "
                "implemented; an operation count is not guessed"
            ),
        }
    readout_macs = int(readout_units)
    return {
        "backend": "analytic",
        **{k: int(v) for k, v in counts.items()},
        "readout_macs": readout_macs,
        "total_macs": int(counts["reservoir_macs"]) + readout_macs,
        "note": "analytic MAC estimate from weight shapes; not an energy measurement",
    }
```

- [ ] **Step 6: Прогнать тесты**

Run: `poetry run pytest tests/test_activity.py -v`
Expected: 6 passed.

- [ ] **Step 7: Коммит**

```bash
bash scripts/verify.sh quick
git add src/rc_bench/profiling/activity.py src/rc_bench/core/reservoirs/ tests/test_activity.py
git commit -m "feat(profiling): analytic operation counts and state sparsity proxies (PROXY-001)"
```

---

### Task 3: PROXY-002 — спайки и синаптические события LSM

**Files:**
- Modify: `src/rc_bench/core/reservoirs/lsm_service.py`
- Modify: `src/rc_bench/profiling/activity.py`
- Modify: `tests/test_activity.py`

**Interfaces:**
- Consumes: `LSMReservoir.step`, `LSMReservoir._W_rec`.
- Produces:
  - `LSMReservoir.spike_stats() -> Dict[str, float]` со счётчиками, накопленными с последнего `reset_state()`;
  - `activity.spiking_activity(reservoir) -> Optional[Dict[str, float]]`.

- [ ] **Step 1: Написать падающий тест**

```python
# добавить в tests/test_activity.py
from rc_bench.profiling.activity import spiking_activity


def test_lsm_counts_spikes_and_synaptic_events():
    reservoir = get_reservoir("lsm", {"units": 60, "density": 0.1, "seed": 3})
    reservoir.reset_state()
    rng = np.random.default_rng(0)
    for _ in range(200):
        reservoir.step(rng.normal(0.0, 1.0, (1, 1)))

    stats = spiking_activity(reservoir)
    assert stats["steps"] == 200
    assert stats["total_spikes"] > 0, "LSM с этими параметрами обязан спайковать"
    assert stats["spikes_per_step"] == pytest.approx(stats["total_spikes"] / 200)
    # Синаптическое событие — доставка спайка по исходящей ненулевой связи.
    fanout = np.count_nonzero(reservoir._W_rec) / reservoir._W_rec.shape[0]
    assert stats["synaptic_events_per_step"] == pytest.approx(
        stats["spikes_per_step"] * fanout, rel=0.35
    )
    assert 0.0 <= stats["mean_firing_rate_hz"] < 1000.0


def test_reset_state_clears_spike_counters():
    reservoir = get_reservoir("lsm", {"units": 40, "seed": 3})
    reservoir.reset_state()
    for _ in range(50):
        reservoir.step(np.array([[1.0]]))
    assert spiking_activity(reservoir)["steps"] == 50
    reservoir.reset_state()
    assert spiking_activity(reservoir)["steps"] == 0


def test_non_spiking_reservoir_has_no_spiking_activity():
    assert spiking_activity(get_reservoir("leaky_esn", {"units": 10, "seed": 3})) is None
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `poetry run pytest tests/test_activity.py -k spik -v`
Expected: FAIL — `ImportError: cannot import name 'spiking_activity'`.

- [ ] **Step 3: Добавить счётчики в LSM**

В `_build` дочитать `dt`, чтобы считать частоту в герцах:

```python
        self._dt_ms = dt   # уже прочитан выше как float(config.get("dt", 1.0))
```

В `reset_state` обнулить счётчики; в `step` — накапливать:

```python
    def reset_state(self) -> None:
        units = self._W_rec.shape[0]
        self._step_v = np.zeros(units)
        self._step_s = np.zeros(units)
        self._step_refrac = np.zeros(units, dtype=np.int32)
        # Счётчики активности (PROXY-002): растут только внутри step(),
        # обнуляются вместе с состоянием, чтобы статистика всегда относилась
        # к одному и тому же прогону.
        self._spike_steps = 0
        self._spike_total = 0.0
        self._synaptic_events_total = 0.0
        # Исходящий fan-out каждого нейрона: сколько ненулевых связей он
        # активирует, когда спайкует. Столбец j матрицы W_rec — вклад
        # нейрона j во все остальные.
        self._out_degree = np.count_nonzero(self._W_rec, axis=0).astype(float)

    def step(self, x_t: np.ndarray) -> np.ndarray:
        ...
        s = self._alpha_syn * s + spikes
        self._step_v, self._step_s, self._step_refrac = v, s, refrac
        self._spike_steps += 1
        self._spike_total += float(spikes.sum())
        self._synaptic_events_total += float(self._out_degree @ spikes)
        return s

    def spike_stats(self) -> Dict[str, float]:
        """Спайковая статистика с момента последнего reset_state()."""
        steps = int(getattr(self, "_spike_steps", 0))
        units = int(self._W_rec.shape[0])
        if steps == 0:
            return {
                "steps": 0, "units": units, "total_spikes": 0.0,
                "spikes_per_step": 0.0, "synaptic_events_per_step": 0.0,
                "mean_firing_rate_hz": 0.0,
            }
        spikes_per_step = self._spike_total / steps
        return {
            "steps": steps,
            "units": units,
            "total_spikes": float(self._spike_total),
            "spikes_per_step": spikes_per_step,
            "synaptic_events_per_step": self._synaptic_events_total / steps,
            # dt задан в миллисекундах симуляции; частота — спайков на нейрон в секунду.
            "mean_firing_rate_hz": (spikes_per_step / units) * (1000.0 / self._dt_ms),
        }
```

- [ ] **Step 4: Добавить `spiking_activity` в `activity.py`**

```python
def spiking_activity(reservoir: BaseReservoir) -> Optional[Dict[str, float]]:
    """Спайковая статистика для моделей, которые её ведут (сейчас — LSM).

    ``None`` для не-спайковых резервуаров: у них нет спайков, и подставлять
    нули было бы враньём другого рода.
    """
    stats_fn = getattr(reservoir, "spike_stats", None)
    if stats_fn is None:
        return None
    return {k: float(v) for k, v in stats_fn().items()}
```

(добавить `Optional` в импорт `typing`)

- [ ] **Step 5: Прогнать тесты**

Run: `poetry run pytest tests/test_activity.py -v`
Expected: 9 passed.

- [ ] **Step 6: Проверить, что счётчики не сломали существующее поведение LSM**

Run: `poetry run pytest tests/test_reservoirs.py -v`
Expected: все прежние тесты LSM зелёные (детерминизм `transform`/`step` не изменился —
счётчики только читают `spikes`).

- [ ] **Step 7: Коммит**

```bash
bash scripts/verify.sh quick
git add src/rc_bench/core/reservoirs/lsm_service.py src/rc_bench/profiling/activity.py tests/test_activity.py
git commit -m "feat(profiling): LSM spike and synaptic-event counters (PROXY-002)"
```

---

### Task 4: ENERGY-002 — backend Intel RAPL и протокол энергоизмерения

**Files:**
- Create: `src/rc_bench/profiling/energy.py`
- Create: `tests/test_energy.py`

**Interfaces:**
- Consumes: `time.perf_counter_ns`, файловую систему `/sys/class/powercap`.
- Produces:
  - `energy.discover_rapl_domains(root: Path = RAPL_ROOT) -> List[RaplDomain]`
  - `energy.energy_backend_status(root: Path = RAPL_ROOT) -> Dict[str, Any]`
  - `energy.measure_energy(step_fn, *, p50_ns, min_duration_s=2.0, min_steps=1000, root=RAPL_ROOT) -> Dict[str, Any]`
  - Ключи результата: `status`, `backend`, `domains`, `n_steps`, `duration_s`,
    `total_energy_j`, `idle_energy_j`, `net_energy_j`, `energy_per_inference_mj`,
    `net_energy_per_inference_mj`, `samples_per_joule`, `net_samples_per_joule`,
    `energy_delay_product_j_s`.

- [ ] **Step 1: Написать падающие тесты на fake sysfs**

```python
# tests/test_energy.py
"""Энергия измеряется, а не оценивается. Тесты работают на подставном sysfs:
настоящий RAPL нельзя воспроизвести в CI, а вот арифметику окна, обработку
переполнения счётчика и честный отказ при отсутствии доступа — можно и нужно."""
from pathlib import Path

import pytest

from rc_bench.profiling import energy as energy_mod


def _fake_rapl(tmp_path: Path, *, energy_uj: int = 1_000_000, readable: bool = True) -> Path:
    root = tmp_path / "powercap"
    domain = root / "intel-rapl:0"
    domain.mkdir(parents=True)
    (domain / "name").write_text("package-0\n")
    (domain / "max_energy_range_uj").write_text("262143328850\n")
    counter = domain / "energy_uj"
    counter.write_text(f"{energy_uj}\n")
    if not readable:
        counter.chmod(0o000)
    return root


def test_discover_returns_named_domains(tmp_path):
    root = _fake_rapl(tmp_path)
    domains = energy_mod.discover_rapl_domains(root)
    assert [d.name for d in domains] == ["package-0"]
    assert domains[0].max_range_uj == 262143328850


def test_status_is_unavailable_without_powercap(tmp_path):
    status = energy_mod.energy_backend_status(tmp_path / "nothing-here")
    assert status["status"] == "unavailable"
    assert "powercap" in status["reason"]


def test_status_is_unavailable_when_counter_is_not_readable(tmp_path):
    root = _fake_rapl(tmp_path, readable=False)
    status = energy_mod.energy_backend_status(root)
    assert status["status"] == "unavailable"
    assert "readable" in status["reason"]


def test_status_is_available_when_counter_reads(tmp_path):
    status = energy_mod.energy_backend_status(_fake_rapl(tmp_path))
    assert status["status"] == "available"
    assert status["backend"] == "intel_rapl"
    assert status["domains"] == ["package-0"]


def test_counter_wraparound_is_not_reported_as_negative_energy(tmp_path):
    root = _fake_rapl(tmp_path)
    domain = energy_mod.discover_rapl_domains(root)[0]
    # Счётчик переполнился: конец меньше начала.
    delta_uj = energy_mod.counter_delta_uj(
        start_uj=domain.max_range_uj - 1_000, end_uj=2_000, domain=domain
    )
    assert delta_uj == pytest.approx(3_001)


def test_measure_energy_reports_unavailable_instead_of_failing(tmp_path):
    result = energy_mod.measure_energy(
        lambda: None, p50_ns=1_000.0, min_duration_s=0.01, min_steps=10,
        root=tmp_path / "nothing-here",
    )
    assert result["status"] == "unavailable"
    assert "energy_per_inference_mj" not in result


def test_measure_energy_derives_window_from_measured_latency(tmp_path, monkeypatch):
    root = _fake_rapl(tmp_path)
    # Счётчик, растущий на 1 мДж за каждое чтение, даёт предсказуемую арифметику.
    reads = {"n": 0}
    real_read = energy_mod.read_domain_uj

    def fake_read(domain):
        reads["n"] += 1
        return 1_000 * reads["n"]

    monkeypatch.setattr(energy_mod, "read_domain_uj", fake_read)
    calls = {"n": 0}

    def step_fn():
        calls["n"] += 1

    result = energy_mod.measure_energy(
        step_fn, p50_ns=1_000_000.0, min_duration_s=0.05, min_steps=10, root=root
    )
    assert result["status"] == "measured"
    # min_duration_s / p50 = 0.05с / 1мс = 50 шагов, что больше min_steps=10.
    assert result["n_steps"] == 50
    assert calls["n"] >= 50
    assert result["energy_per_inference_mj"] > 0
    assert result["samples_per_joule"] > 0
    assert result["energy_delay_product_j_s"] > 0
    assert real_read is not energy_mod.read_domain_uj  # sanity: патч применился
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `poetry run pytest tests/test_energy.py -v`
Expected: FAIL — `ModuleNotFoundError: rc_bench.profiling.energy`.

- [ ] **Step 3: Реализовать `energy.py`**

```python
"""Измерение энергии вывода через Intel RAPL (ENERGY-002).

Протокол (docs/PROJECT_CONTRACT.md, «Energy»):

- источник — powercap-интерфейс ядра, ``/sys/class/powercap/intel-rapl:*``;
  берутся домены верхнего уровня (package), а не их поддомены, чтобы не
  сложить одну и ту же энергию дважды;
- окно измерения не короче ``min_duration_s`` (по умолчанию 2 с) И не меньше
  ``min_steps`` шагов: разрешение счётчика ~61 мкДж и период обновления ~1 мс
  сопоставимы с длительностью одного шага (десятки микросекунд), поэтому
  измерять один вызов бессмысленно;
- переполнение счётчика обрабатывается по ``max_energy_range_uj``;
- отдельно измеряется энергия простоя той же длительности, и публикуются
  обе величины: полная (пакет целиком) и net (за вычетом простоя). Без
  вычитания простоя число говорит в основном о статическом потреблении
  процессора, а не о модели;
- при отсутствии или недоступности счётчика возвращается
  ``status="unavailable"`` с причиной. Прокси-метрики сюда не подставляются
  никогда: energy и activity — разные блоки (DEC-007).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

RAPL_ROOT = Path("/sys/class/powercap")


@dataclass(frozen=True)
class RaplDomain:
    name: str
    energy_path: Path
    max_range_uj: int


def discover_rapl_domains(root: Path = RAPL_ROOT) -> List[RaplDomain]:
    """Домены верхнего уровня (``intel-rapl:N``), без поддоменов ``:N:M``."""
    root = Path(root)
    if not root.is_dir():
        return []
    domains: List[RaplDomain] = []
    for entry in sorted(root.glob("intel-rapl:*")):
        if entry.name.count(":") != 1:
            continue
        name_file, energy_file = entry / "name", entry / "energy_uj"
        range_file = entry / "max_energy_range_uj"
        if not (name_file.is_file() and energy_file.is_file()):
            continue
        try:
            max_range = int(range_file.read_text().strip())
        except (OSError, ValueError):
            max_range = 2**32 - 1
        domains.append(
            RaplDomain(name_file.read_text().strip(), energy_file, max_range)
        )
    return domains


def read_domain_uj(domain: RaplDomain) -> int:
    return int(domain.energy_path.read_text().strip())


def counter_delta_uj(*, start_uj: int, end_uj: int, domain: RaplDomain) -> int:
    """Разность с учётом кольцевого переполнения счётчика."""
    if end_uj >= start_uj:
        return end_uj - start_uj
    return (domain.max_range_uj - start_uj) + end_uj + 1


def energy_backend_status(root: Path = RAPL_ROOT) -> Dict[str, Any]:
    domains = discover_rapl_domains(root)
    if not domains:
        return {
            "status": "unavailable",
            "backend": None,
            "reason": f"no powercap intel-rapl domains under {root}",
        }
    try:
        for domain in domains:
            read_domain_uj(domain)
    except OSError as exc:
        return {
            "status": "unavailable",
            "backend": None,
            "reason": (
                f"intel-rapl counter {exc.filename} is not readable by this user "
                "(root-only since CVE-2020-8694); see the plan's prerequisite"
            ),
        }
    return {
        "status": "available",
        "backend": "intel_rapl",
        "domains": [d.name for d in domains],
    }


def _sum_uj(domains: List[RaplDomain]) -> List[int]:
    return [read_domain_uj(d) for d in domains]


def _elapsed_uj(domains: List[RaplDomain], start: List[int], end: List[int]) -> float:
    return float(
        sum(
            counter_delta_uj(start_uj=s, end_uj=e, domain=d)
            for d, s, e in zip(domains, start, end)
        )
    )


def measure_energy(
    step_fn: Callable[[], Any],
    *,
    p50_ns: float,
    min_duration_s: float = 2.0,
    min_steps: int = 1000,
    root: Path = RAPL_ROOT,
) -> Dict[str, Any]:
    """Энергия одного шага вывода ``step_fn``.

    ``p50_ns`` — уже измеренная медианная задержка шага: по ней выбирается
    число шагов, чтобы окно измерения было длиннее периода обновления
    счётчика. Она же используется для energy-delay product.
    """
    status = energy_backend_status(root)
    if status["status"] != "available":
        return status

    domains = discover_rapl_domains(root)
    n_steps = max(int(min_steps), int((min_duration_s * 1e9) // max(p50_ns, 1.0)))

    # Простой той же длительности — базовая линия статического потребления.
    idle_start = _sum_uj(domains)
    t0 = time.perf_counter_ns()
    time.sleep(min_duration_s)
    idle_duration_s = (time.perf_counter_ns() - t0) / 1e9
    idle_uj = _elapsed_uj(domains, idle_start, _sum_uj(domains))

    start = _sum_uj(domains)
    t0 = time.perf_counter_ns()
    for _ in range(n_steps):
        step_fn()
    duration_s = (time.perf_counter_ns() - t0) / 1e9
    total_uj = _elapsed_uj(domains, start, _sum_uj(domains))

    idle_scaled_uj = idle_uj * (duration_s / idle_duration_s) if idle_duration_s > 0 else 0.0
    net_uj = max(total_uj - idle_scaled_uj, 0.0)

    total_j, net_j = total_uj / 1e6, net_uj / 1e6
    per_inference_mj = (total_uj / 1e3) / n_steps
    net_per_inference_mj = (net_uj / 1e3) / n_steps
    return {
        "status": "measured",
        "backend": "intel_rapl",
        "domains": [d.name for d in domains],
        "n_steps": n_steps,
        "duration_s": duration_s,
        "idle_duration_s": idle_duration_s,
        "total_energy_j": total_j,
        "idle_energy_j": idle_scaled_uj / 1e6,
        "net_energy_j": net_j,
        "energy_per_inference_mj": per_inference_mj,
        "net_energy_per_inference_mj": net_per_inference_mj,
        "samples_per_joule": (n_steps / total_j) if total_j > 0 else None,
        "net_samples_per_joule": (n_steps / net_j) if net_j > 0 else None,
        # EDP на один вывод: (net энергия одного шага, Дж) × (p50 задержка, с).
        "energy_delay_product_j_s": (net_per_inference_mj / 1e3) * (p50_ns / 1e9),
        "note": (
            "package-domain energy; idle baseline of equal duration subtracted "
            "for the net values. Not a per-model isolated measurement."
        ),
    }
```

- [ ] **Step 4: Прогнать тесты**

Run: `poetry run pytest tests/test_energy.py -v`
Expected: 7 passed.

- [ ] **Step 5: Проверить backend на реальном железе**

Run:
```bash
poetry run python -c "
from rc_bench.profiling.energy import energy_backend_status, measure_energy
print(energy_backend_status())
print(measure_energy(lambda: sum(range(50)), p50_ns=2000.0, min_duration_s=1.0))
"
```
Expected: `{'status': 'available', 'backend': 'intel_rapl', 'domains': ['package-0']}` и
измерение с положительными `energy_per_inference_mj`/`samples_per_joule`.
Если `unavailable` — предусловие плана не выполнено; записать это в handoff и идти дальше
(остальные задачи не блокируются).

- [ ] **Step 6: Коммит**

```bash
bash scripts/verify.sh quick
git add src/rc_bench/profiling/energy.py tests/test_energy.py
git commit -m "feat(profiling): Intel RAPL energy backend with idle-corrected per-inference energy (ENERGY-002)"
```

---

### Task 5: ENERGY-003 — energy и activity в проходе профилирования и схеме

**Files:**
- Modify: `src/rc_bench/core/schema.py` (`EnergyResult`) — **основной агент**
- Modify: `src/rc_bench/profiling/model_profiles.py`
- Modify: `src/rc_bench/profiling/hardware.py`
- Modify: `src/rc_bench/reporting/pareto.py`
- Modify: `tests/test_model_profiles.py`, `tests/test_jmlc_schema.py`, `tests/test_pareto.py`

**Interfaces:**
- Consumes: `energy.measure_energy`, `activity.state_sparsity`, `activity.count_step_operations`, `activity.spiking_activity`.
- Produces: в каждой ячейке профиля появляются ключи `"energy"` и `"activity"`;
  в `profiles/summary.json` — `net_energy_per_inference_mj`, `net_samples_per_joule`,
  `energy_delay_product_j_s`, `total_macs`, `state_near_zero_fraction`,
  `spikes_per_step`, `synaptic_events_per_step` (последние два — `None` для не-LSM);
  в `pareto.COST_AXES` — ось `"energy"`.

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_jmlc_schema.py
def test_energy_result_accepts_a_measured_backend():
    from rc_bench.core.schema import EnergyResult
    measured = EnergyResult(
        status="measured",
        backend="intel_rapl",
        net_energy_per_inference_mj=0.0123,
        net_samples_per_joule=81300.0,
        energy_delay_product_j_s=2.7e-10,
        domains=["package-0"],
    )
    assert measured.status == "measured"
    # Значение по умолчанию не меняется: без счётчика запись остаётся честной.
    assert EnergyResult().status == "unavailable"
    assert EnergyResult().backend is None


def test_energy_result_rejects_measured_without_numbers():
    import pytest
    from pydantic import ValidationError
    from rc_bench.core.schema import EnergyResult
    with pytest.raises(ValidationError):
        EnergyResult(status="measured", backend="intel_rapl")
```

```python
# tests/test_model_profiles.py
def test_profile_cell_carries_activity_and_energy_blocks(monkeypatch, tmp_path):
    """Профиль ячейки обязан нести блок activity всегда и блок energy —
    с явным статусом. Отсутствующий ключ невозможно отличить от забытого."""
    from rc_bench.profiling import model_profiles

    monkeypatch.setattr(
        model_profiles, "measure_energy",
        lambda *a, **k: {"status": "unavailable", "backend": None, "reason": "test"},
    )
    cell = model_profiles._profile_cell_body(
        "reservoir", "leaky_esn", _leaky_esn_spec(), _tiny_data(), n_steps=20, warmup=5
    )
    assert cell["energy"]["status"] == "unavailable"
    assert cell["activity"]["operations"]["backend"] == "analytic"
    assert 0.0 <= cell["activity"]["state_sparsity"]["near_zero_fraction"] <= 1.0
    assert cell["activity"]["spiking"] is None


def test_lsm_profile_carries_spiking_activity(monkeypatch):
    from rc_bench.profiling import model_profiles
    monkeypatch.setattr(
        model_profiles, "measure_energy",
        lambda *a, **k: {"status": "unavailable", "backend": None, "reason": "test"},
    )
    cell = model_profiles._profile_cell_body(
        "reservoir", "lsm", _lsm_spec(), _tiny_data(), n_steps=20, warmup=5
    )
    assert cell["activity"]["spiking"]["spikes_per_step"] >= 0.0
    assert cell["activity"]["spiking"]["synaptic_events_per_step"] >= 0.0


def test_summary_entry_exposes_energy_and_activity():
    from rc_bench.profiling.model_profiles import _summary_entry
    entry = _summary_entry(_completed_cell_with_energy())
    assert entry["net_energy_per_inference_mj"] == pytest.approx(0.0123)
    assert entry["total_macs"] > 0
    assert entry["energy_status"] == "measured"
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `poetry run pytest tests/test_jmlc_schema.py tests/test_model_profiles.py -k "energy or activity" -v`
Expected: FAIL — `EnergyResult` не принимает `status="measured"`, ключа `activity` нет.

- [ ] **Step 3: Расширить `EnergyResult` (основной агент, `core/schema.py`)**

```python
class EnergyResult(BaseModel):
    """Энергия вывода: измерена аппаратным счётчиком или явно недоступна.

    Прокси-метрики активности сюда не попадают ни при каких условиях — они
    живут в блоке ``activity`` профиля (DEC-007). ``unavailable`` остаётся
    значением по умолчанию, поэтому все прежние записи читаются без миграции.
    """

    status: Literal["measured", "unavailable"] = "unavailable"
    reason: Optional[str] = "No supported hardware energy counter available"
    backend: Optional[Literal["intel_rapl"]] = None
    domains: List[str] = Field(default_factory=list)
    net_energy_per_inference_mj: Optional[float] = None
    net_samples_per_joule: Optional[float] = None
    energy_delay_product_j_s: Optional[float] = None

    @model_validator(mode="after")
    def _measured_requires_numbers(self) -> "EnergyResult":
        if self.status == "measured":
            missing = [
                name
                for name in (
                    "backend",
                    "net_energy_per_inference_mj",
                    "net_samples_per_joule",
                    "energy_delay_product_j_s",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"status='measured' requires {', '.join(missing)}"
                )
        return self
```

- [ ] **Step 4: Подключить energy и activity в `_profile_cell_body`**

В `model_profiles.py` добавить импорты и собрать блоки после измерения latency:

```python
from rc_bench.profiling.activity import (
    count_step_operations, spiking_activity, state_sparsity,
)
from rc_bench.profiling.energy import measure_energy
```

В `_profile_reservoir_cell` после `deployable_latency` и `h_sample`:

```python
    readout_units = int(np.asarray(h_sample).reshape(-1).size)
    activity = {
        "operations": count_step_operations(reservoir, readout_units=readout_units),
        "state_sparsity": state_sparsity(np.asarray(h_sample)),
        "spiking": spiking_activity(reservoir),
        "note": "analytic/observed proxies; never converted into energy",
    }
    reservoir.reset_state()
    energy_step_fn = build_reservoir_compute_step_fn(reservoir, readout, X_test_s)
    energy = measure_energy(energy_step_fn, p50_ns=deployable_latency.p50_ns)
```

и добавить `"energy": energy, "activity": activity` в возвращаемый словарь.

Для baseline-ячеек (`_profile_ridge_ar_cell`, `_profile_persistence_family_cell`)
блок activity описывает только аналитическую стоимость выхода — резервуара там нет:

```python
    activity = {
        "operations": {
            "backend": "analytic",
            "reservoir_macs": 0,
            "reservoir_nonlinearities": 0,
            "reservoir_nonzero_recurrent_weights": 0,
            "readout_macs": readout_macs,   # tau для ridge_ar, 0 для persistence
            "total_macs": readout_macs,
            "note": "analytic MAC estimate; not an energy measurement",
        },
        "state_sparsity": None,   # рабочее состояние — окно входа, не состояние модели
        "spiking": None,
    }
    energy = measure_energy(deployable_step_fn_for_energy, p50_ns=deployable_latency.p50_ns)
```

- [ ] **Step 5: Вывести новые поля в `_summary_entry`**

```python
    energy = cell.get("energy") or {"status": "unavailable"}
    activity = cell.get("activity") or {}
    operations = activity.get("operations") or {}
    sparsity = activity.get("state_sparsity") or {}
    spiking = activity.get("spiking") or {}
    base.update(
        ...,
        energy_status=energy.get("status"),
        net_energy_per_inference_mj=energy.get("net_energy_per_inference_mj"),
        net_samples_per_joule=energy.get("net_samples_per_joule"),
        energy_delay_product_j_s=energy.get("energy_delay_product_j_s"),
        total_macs=operations.get("total_macs"),
        state_near_zero_fraction=sparsity.get("near_zero_fraction"),
        spikes_per_step=spiking.get("spikes_per_step"),
        synaptic_events_per_step=spiking.get("synaptic_events_per_step"),
    )
```

- [ ] **Step 6: Заменить жёсткий energy-блок в `hardware.py`**

Найти место, где `_hardware_artifact` пишет energy-блок, и подставить реальный статус:

```python
from rc_bench.profiling.energy import energy_backend_status
...
        "energy": energy_backend_status(),
```

- [ ] **Step 7: Добавить ось energy в Pareto**

```python
# reporting/pareto.py
COST_AXES: Dict[str, Tuple[str, str, float]] = {
    "latency": ("deployable_p50_ns", "Задержка одного шага, p50 (мс, log)", 1e-6),
    "memory": ("working_state_bytes", "Рабочее состояние (КиБ, log)", 1 / 1024),
    "energy": ("net_energy_per_inference_mj", "Энергия одного вывода (мДж, log)", 1.0),
}
```

`generate_pareto_plots` должен пропускать ось energy, если хотя бы у одной ячейки
`net_energy_per_inference_mj is None`, и возвращать причину пропуска — график с дырами
хуже отсутствующего графика:

```python
def generate_pareto_plots(bundle_dir, output_dir) -> Dict[str, str]:
    written: Dict[str, str] = {}
    for cost in COST_AXES:
        try:
            points = load_points(bundle_dir, cost)
        except MissingCostAxis as exc:      # новое исключение вместо ValueError
            written[cost] = f"skipped: {exc}"
            continue
        ...
```

- [ ] **Step 8: Прогнать тесты**

Run: `poetry run pytest tests/test_model_profiles.py tests/test_jmlc_schema.py tests/test_pareto.py tests/test_profiling.py -v`
Expected: все зелёные, включая новые.

- [ ] **Step 9: Коммит**

```bash
bash scripts/verify.sh quick
git add src/rc_bench/core/schema.py src/rc_bench/profiling/ src/rc_bench/reporting/pareto.py tests/
git commit -m "feat(profiling): publish measured energy and activity proxies per cell (ENERGY-003)"
```

---

### Task 6: MODE-001 — режим best-effort в спецификации и раннере матрицы

**Files:**
- Modify: `src/rc_bench/core/schema.py` (`ProtocolSpec.mode`, `config_hash`) — **основной агент**
- Modify: `src/rc_bench/runners/jmlc_matrix.py`
- Create: `configs/jmlc/best_effort.yaml`
- Modify: `tests/test_jmlc_schema.py`, `tests/test_jmlc_matrix.py`

**Interfaces:**
- Consumes: `ExperimentSpec`, `build_cell_spec`, `run_matrix`.
- Produces:
  - `ProtocolSpec.mode: Literal["fair", "best_effort"] = "fair"` — попадает во frozen и resolved spec каждой ячейки, значит и в каждый RunRecord;
  - ключ шаблона `matrix: {mode: ..., budgets: {<model>: <int>}}`, читаемый `build_cell_spec`;
  - `jmlc_matrix.resolve_mode(template) -> str` и `jmlc_matrix.resolve_budget(template, model) -> int`.

- [ ] **Step 1: Написать падающий тест на стабильность config_hash**

Это главный риск задачи: новое поле протокола меняет `config_hash` каждой ячейки, и весь
опубликованный fair-бандл перестаёт сходиться сам с собой.

```python
# tests/test_jmlc_schema.py
def test_default_mode_does_not_change_existing_config_hashes():
    """Значение по умолчанию выбрасывается из payload, как уже сделано для
    selection_metric и seasonal_period — иначе добавление поля переименует
    каждую опубликованную ячейку и порвёт traceability бандла."""
    from rc_bench.core.schema import ExperimentSpec
    spec = ExperimentSpec.model_validate({
        "dataset": {"name": "uci_household_power", "length": 12000},
        "reservoir": {"type": "esn", "params": {}},
        "protocol": {"forecasting_mode": "fixed_horizon", "horizon": 1,
                     "use_hpo": True, "hpo_budget": 20, "n_seeds": 5,
                     "selection_metric": "nrmse_std", "seasonal_period": 24},
        "readout": {"alpha_grid": [0.001, 0.01, 0.1, 1.0, 10.0]},
        "seed": 42,
    })
    assert spec.protocol.mode == "fair"
    # Хеш из опубликованного бандла: reports/jmlc_2026/fair/runs/reservoir_esn_h1.json
    assert spec.config_hash() == PUBLISHED_ESN_H1_CONFIG_HASH


def test_best_effort_mode_changes_the_config_hash():
    from rc_bench.core.schema import ExperimentSpec
    base = _fair_spec_dict()
    fair = ExperimentSpec.model_validate(base)
    best = ExperimentSpec.model_validate(
        {**base, "protocol": {**base["protocol"], "mode": "best_effort"}}
    )
    assert fair.config_hash() != best.config_hash()
```

`PUBLISHED_ESN_H1_CONFIG_HASH` брать из
`reports/jmlc_2026/fair/runs/reservoir_esn_h1.json` (`result.config_hash`) — прочитать
файл в тесте, а не хардкодить строку.

- [ ] **Step 2: Убедиться, что тест падает**

Run: `poetry run pytest tests/test_jmlc_schema.py -k mode -v`
Expected: FAIL — `ProtocolSpec` не имеет поля `mode`.

- [ ] **Step 3: Добавить поле и сохранить хеш**

```python
# ProtocolSpec
    # Режим сравнения (§2 описания проекта): fair — единый бюджет и правила
    # для всех reservoir-моделей; best_effort — индивидуальная настройка,
    # результаты двух режимов никогда не смешиваются в одной таблице.
    mode: Literal["fair", "best_effort"] = "fair"
```

```python
# ExperimentSpec.config_hash — рядом с существующими выбрасываниями умолчаний
        if protocol.get("mode") == "fair":
            protocol.pop("mode", None)
```

- [ ] **Step 4: Прогнать тест хеша**

Run: `poetry run pytest tests/test_jmlc_schema.py -k mode -v`
Expected: 2 passed — опубликованный хеш не изменился.

- [ ] **Step 5: Написать падающий тест раннера**

```python
# tests/test_jmlc_matrix.py
def test_best_effort_template_gives_each_model_its_own_budget():
    from rc_bench.runners.jmlc_matrix import build_cell_spec, resolve_budget, resolve_mode
    template = {
        **_fair_template(),
        "matrix": {"mode": "best_effort", "budgets": {"esn": 60, "lsm": 40}},
    }
    assert resolve_mode(template) == "best_effort"
    assert resolve_budget(template, "esn") == 60
    assert resolve_budget(template, "lsm") == 40
    # Модель без индивидуального бюджета наследует бюджет шаблона.
    assert resolve_budget(template, "logistic") == template["protocol"]["hpo_budget"]

    spec = build_cell_spec(template, "reservoir", "lsm", 24)
    assert spec.protocol.mode == "best_effort"
    assert spec.protocol.hpo_budget == 40


def test_fair_template_ignores_per_model_budgets():
    """В fair-режиме индивидуальный бюджет — нарушение DEC-004, а не опция."""
    import pytest
    from rc_bench.runners.jmlc_matrix import resolve_budget
    template = {**_fair_template(), "matrix": {"mode": "fair", "budgets": {"esn": 60}}}
    with pytest.raises(ValueError, match="fair"):
        resolve_budget(template, "esn")


def test_baseline_cells_ignore_mode_budgets():
    from rc_bench.runners.jmlc_matrix import build_cell_spec
    template = {**_fair_template(), "matrix": {"mode": "best_effort", "budgets": {"esn": 60}}}
    spec = build_cell_spec(template, "baseline", "persistence", 1)
    assert spec.protocol.use_hpo is False
    assert spec.protocol.mode == "best_effort"   # режим фиксируется и у детерминированных
```

- [ ] **Step 6: Убедиться, что тесты падают**

Run: `poetry run pytest tests/test_jmlc_matrix.py -k "best_effort or mode" -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_mode'`.

- [ ] **Step 7: Реализовать режимы в `jmlc_matrix.py`**

```python
def resolve_mode(template: Dict[str, Any]) -> str:
    """Режим матрицы: ``fair`` (по умолчанию) или ``best_effort``."""
    matrix = template.get("matrix") or {}
    mode = matrix.get("mode", "fair")
    if mode not in ("fair", "best_effort"):
        raise ValueError(f"unknown matrix mode {mode!r}; expected fair or best_effort")
    return mode


def resolve_budget(template: Dict[str, Any], model: str) -> int:
    """HPO-бюджет ячейки.

    В fair-режиме бюджет один на всех (DEC-004), поэтому индивидуальный
    бюджет — не переопределение, а ошибка конфигурации: она означает, что
    автор хотел best_effort и забыл переключить режим.
    """
    template_budget = int(template["protocol"].get("hpo_budget", 100))
    budgets = (template.get("matrix") or {}).get("budgets") or {}
    if not budgets:
        return template_budget
    if resolve_mode(template) == "fair":
        raise ValueError(
            "per-model hpo budgets are only allowed in best_effort mode; "
            "fair mode gives every reservoir model the same budget (DEC-004)"
        )
    return int(budgets.get(model, template_budget))
```

В `build_cell_spec` — прокинуть режим в обе ветки и бюджет в reservoir-ветку:

```python
    protocol["mode"] = resolve_mode(template)
    ...
    if family == "reservoir":
        protocol["hpo_budget"] = resolve_budget(template, model)
```

- [ ] **Step 8: Прогнать тесты**

Run: `poetry run pytest tests/test_jmlc_matrix.py -v`
Expected: все зелёные.

- [ ] **Step 9: Создать `configs/jmlc/best_effort.yaml`**

```yaml
# JMLC best-effort matrix config (§2 описания проекта).
#
# Fair-режим отвечает на вопрос «какая архитектура лучше при равных условиях».
# Best-effort отвечает на другой: «чего каждая архитектура достигает, когда её
# настраивают ради результата». Числа двух режимов НИКОГДА не попадают в одну
# таблицу — бандл держит их в отдельных каталогах и отдельных агрегатах.
#
# Бюджеты подобраны по стоимости ячейки в fair-прогоне (самая тяжёлая, LSM,
# заняла 21 с на 20 trials × 5 seeds), так что весь прогон остаётся в пределах
# ~15 минут машинного времени.
dataset:
  name: uci_household_power
  length: 12000
  raw_sha256: 4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7
reservoir:
  type: esn
  params: {}
matrix:
  mode: best_effort
  budgets:
    esn: 60
    leaky_esn: 60
    logistic: 60
    lsm: 40
protocol:
  washout: 200
  train_frac: 0.6
  val_frac: 0.2
  forecasting_mode: fixed_horizon
  horizon: 1
  use_hpo: true
  hpo_budget: 60          # значение по умолчанию для моделей без своей строки
  n_seeds: 5
  selection_metric: nrmse_std
  seasonal_period: 24
readout:
  alpha_grid: [0.001, 0.01, 0.1, 1.0, 10.0]
seed: 42
```

- [ ] **Step 10: Проверить, что шаблон валиден как ExperimentSpec**

Run: `poetry run rcbench validate-spec configs/jmlc/best_effort.yaml`
Expected: OK (ключ `matrix` не входит в `ExperimentSpec` и игнорируется валидацией —
это же поведение уже используется шаблонами `fair.yaml`/`smoke.yaml`).

- [ ] **Step 11: Коммит**

```bash
bash scripts/verify.sh quick
git add src/rc_bench/core/schema.py src/rc_bench/runners/jmlc_matrix.py configs/jmlc/best_effort.yaml tests/
git commit -m "feat(protocol): add the best-effort matrix mode alongside fair (MODE-001)"
```

---

### Task 7: MODE-002 — evidence-гейт понимает два режима

**Files:**
- Modify: `src/rc_bench/reporting/evidence.py`
- Modify: `scripts/validate_evidence.py`
- Modify: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `ProtocolSpec.mode` из RunRecord, `CellRow`.
- Produces:
  - `CellRow.mode: str` и колонка `mode` в `matrix_table.csv/json`;
  - `validate_bundle(bundle_dir, ..., modes=("fair",))` проверяет каждый подкаталог режима отдельно;
  - `aggregates/matrix_table_best_effort.{json,csv}` для второго режима.

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_evidence.py
def test_equal_budget_rule_applies_only_to_fair_mode():
    """DEC-004 требует равного бюджета в fair. В best_effort равные бюджеты
    означали бы, что режим не сделал того, ради чего существует."""
    fair = [_reservoir_record(model="esn", budget=20), _reservoir_record(model="lsm", budget=40)]
    problems = validate_records(fair)
    assert any("hpo budget" in p for p in problems)

    best = [
        _reservoir_record(model="esn", budget=60, mode="best_effort"),
        _reservoir_record(model="lsm", budget=40, mode="best_effort"),
    ]
    assert not [p for p in validate_records(best) if "hpo budget" in p]


def test_mixed_modes_in_one_directory_are_rejected():
    records = [_reservoir_record(model="esn", budget=20),
               _reservoir_record(model="lsm", budget=40, mode="best_effort")]
    problems = validate_records(records)
    assert any("mixes protocol modes" in p for p in problems)


def test_best_effort_budgets_must_not_be_smaller_than_fair():
    records = [_reservoir_record(model="esn", budget=5, mode="best_effort")]
    problems = validate_records(records, fair_budget=20)
    assert any("smaller than the fair budget" in p for p in problems)


def test_aggregates_carry_the_mode_column():
    rows = build_rows(_fair_records(), runs_dir="reports/jmlc_2026/fair/runs")
    assert {row.mode for row in rows} == {"fair"}
    written = write_aggregates(rows, out_dir=tmp_path)
    header = Path(written["csv"]).read_text().splitlines()[0]
    assert "mode" in header.split(",")
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `poetry run pytest tests/test_evidence.py -k mode -v`
Expected: FAIL — `CellRow` не имеет поля `mode`.

- [ ] **Step 3: Реализовать mode-aware валидацию**

- добавить `mode: str` в `CellRow` и в заголовок CSV/JSON (`build_rows`, `write_aggregates`);
- в `validate_records`:
  - собрать `{record.resolved_spec.protocol.mode}`; если множество больше одного —
    `"bundle directory mixes protocol modes: ..."`;
  - правило равенства бюджетов применять только при `mode == "fair"`;
  - при `mode == "best_effort"` и переданном `fair_budget` требовать
    `budget >= fair_budget` (иначе «best» получился хуже честного) и требовать,
    чтобы бюджеты были записаны в каждой записи;
- в `validate_bundle` — параметр `modes: Sequence[str] = ("fair",)`, для каждого режима
  свой подкаталог `runs/` и свой файл агрегатов
  (`matrix_table.*` для fair, `matrix_table_<mode>.*` для остальных).

- [ ] **Step 4: Обновить `scripts/validate_evidence.py`**

Скрипт принимает `--modes fair,best_effort` (по умолчанию — только те подкаталоги, что
реально существуют в бандле), проверяет каждый и пишет соответствующие агрегаты.
`--write` по-прежнему обязателен для перезаписи агрегатов.

- [ ] **Step 5: Прогнать тесты**

Run: `poetry run pytest tests/test_evidence.py -v`
Expected: все зелёные.

- [ ] **Step 6: Проверить на опубликованном бандле**

Run: `poetry run python scripts/validate_evidence.py reports/jmlc_2026`
Expected: `OK, 14 cells` для режима fair — существующий бандл не сломался.

- [ ] **Step 7: Коммит**

```bash
bash scripts/verify.sh quick
git add src/rc_bench/reporting/evidence.py scripts/validate_evidence.py tests/test_evidence.py
git commit -m "feat(evidence): validate fair and best-effort matrices as separate modes (MODE-002)"
```

---

### Task 8: PROF-004 — время обучения в агрегатах

**Files:**
- Modify: `src/rc_bench/reporting/evidence.py`
- Modify: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `MetricsResult.train_time` / `MetricsSummary.train_time`.
- Produces: колонки `train_time_s` и `train_time_s_sd` в `matrix_table.{json,csv}`.

- [ ] **Step 1: Написать падающий тест**

```python
def test_matrix_table_reports_training_time():
    """§5 описания проекта ставит время обучения первым пунктом ресурсного
    профиля. Оно измеряется в каждом прогоне, но до таблицы не доходило."""
    rows = build_rows(_fair_records(), runs_dir="reports/jmlc_2026/fair/runs")
    esn = next(r for r in rows if r.model == "esn" and r.horizon == 1)
    assert esn.train_time_s > 0
    assert esn.train_time_s_sd is not None   # multi-seed ячейка
    persistence = next(r for r in rows if r.model == "persistence" and r.horizon == 1)
    assert persistence.train_time_s >= 0
    assert persistence.train_time_s_sd is None   # детерминированный baseline
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `poetry run pytest tests/test_evidence.py -k training_time -v`
Expected: FAIL — `AttributeError: 'CellRow' object has no attribute 'train_time_s'`.

- [ ] **Step 3: Добавить поля в `CellRow` и `build_rows`**

Брать `train_time` из `mean`/`std` для multi-seed ячеек и из `metrics` для
детерминированных; `sd` оставлять `None`, когда seeds нет.

- [ ] **Step 4: Прогнать тест и перегенерировать агрегаты**

Run:
```bash
poetry run pytest tests/test_evidence.py -v
poetry run python scripts/validate_evidence.py reports/jmlc_2026 --write
head -1 reports/jmlc_2026/aggregates/matrix_table.csv
```
Expected: тесты зелёные, в заголовке CSV есть `train_time_s,train_time_s_sd`.

- [ ] **Step 5: Коммит**

```bash
bash scripts/verify.sh quick
git add src/rc_bench/reporting/evidence.py reports/jmlc_2026/aggregates tests/test_evidence.py
git commit -m "feat(evidence): surface training time in the published matrix table (PROF-004)"
```

---

### Task 9: SELECT-001 — выбор Pareto-оптимальной архитектуры под ограничения устройства

**Files:**
- Create: `src/rc_bench/reporting/selection.py`
- Create: `tests/test_selection.py`
- Modify: `src/rc_bench/cli/app.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `pareto.load_points`, `pareto.pareto_front`, `pareto.ParetoPoint`, `profiles/summary.json`, `aggregates/matrix_table.json`.
- Produces:
  - `selection.DeviceConstraints(max_p50_us: Optional[float], max_state_bytes: Optional[int], max_model_bytes: Optional[int], max_energy_mj: Optional[float])`
  - `selection.Candidate(model, family, horizon, nrmse_std, p50_us, state_bytes, model_bytes, energy_mj, feasible, violations: List[str])`
  - `selection.select(bundle_dir, horizon, constraints) -> SelectionReport`
  - `selection.render_markdown(report) -> str`
  - CLI: `rcbench select --bundle <dir> --horizon 1 [--max-latency-us F] [--max-state-bytes N] [--max-model-bytes N] [--max-energy-mj F] [--output selection.md]`

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_selection.py
"""§9 описания проекта обещает отчёт, по которому можно найти
Pareto-оптимальную архитектуру под ограничения устройства. Тесты фиксируют
три свойства: отбрасываются только те кандидаты, что реально нарушают
ограничение; причина отказа называется; при пустом допустимом множестве
отчёт говорит об этом, а не подсовывает ближайшую модель."""
import json
from pathlib import Path

import pytest

from rc_bench.reporting.selection import DeviceConstraints, render_markdown, select


def test_selects_the_best_quality_model_within_the_latency_budget(fake_bundle):
    report = select(fake_bundle, horizon=1,
                    constraints=DeviceConstraints(max_p50_us=5.0))
    assert report.winner.model == "ridge_ar"      # esn быстрее порога не проходит
    assert report.winner.nrmse_std == pytest.approx(0.6760, abs=1e-4)
    assert [c.model for c in report.candidates if not c.feasible]
    esn = next(c for c in report.candidates if c.model == "esn")
    assert esn.feasible is False
    assert any("p50" in v for v in esn.violations)


def test_unconstrained_selection_returns_the_best_model_overall(fake_bundle):
    report = select(fake_bundle, horizon=1, constraints=DeviceConstraints())
    assert report.winner.model == "esn"
    assert all(c.feasible for c in report.candidates)


def test_impossible_constraints_report_no_winner(fake_bundle):
    report = select(fake_bundle, horizon=1,
                    constraints=DeviceConstraints(max_state_bytes=1))
    assert report.winner is None
    assert "нет допустимых" in render_markdown(report).lower()


def test_energy_constraint_is_ignored_when_energy_is_unavailable(fake_bundle_no_energy):
    report = select(fake_bundle_no_energy, horizon=1,
                    constraints=DeviceConstraints(max_energy_mj=0.01))
    assert report.energy_status == "unavailable"
    assert report.winner is not None
    assert "энергия недоступна" in render_markdown(report).lower()


def test_markdown_names_the_frontier_and_the_evidence(fake_bundle):
    md = render_markdown(select(fake_bundle, horizon=1, constraints=DeviceConstraints()))
    assert "aggregates/matrix_table.json" in md
    assert "profiles/summary.json" in md
    assert "| esn |" in md
```

Фикстуры `fake_bundle` / `fake_bundle_no_energy` собирают минимальный бандл в `tmp_path`
(`aggregates/matrix_table.json` + `profiles/summary.json` с согласованными `config_hash`),
используя те же ключи, что пишет `_summary_entry`.

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `poetry run pytest tests/test_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: rc_bench.reporting.selection`.

- [ ] **Step 3: Реализовать `selection.py`**

Ключевые решения, которые должны быть видны в коде:

```python
"""Выбор архитектуры под ограничения устройства (SELECT-001).

Отбор идёт по опубликованным артефактам бандла, а не по свежему прогону:
пользователь должен получать тот же ответ, что и проверяющий, читающий
таблицу руками.

Ограничение, для которого нет измерения (энергия без счётчика), не
превращается в отказ: оно объявляется неприменимым и называется в отчёте.
Молча отбросить все модели по ограничению, которое нечем проверить, — это
неверный ответ, выданный с уверенным видом.
"""
```

- `select()` читает `matrix_table.json` и `profiles/summary.json`, соединяет по
  `(family, model, horizon)` и `config_hash` (как `pareto.load_points`), строит кандидатов;
- проверка ограничений: каждое нарушенное даёт строку в `violations`;
- `winner` — допустимый кандидат с минимальным `nrmse_std`;
- в отчёт входит и Pareto-фронт (`pareto_front` по осям качество-задержка и
  качество-память), чтобы было видно, что победитель — не единственная разумная точка;
- `render_markdown` печатает: ограничения, таблицу кандидатов с отметками
  допустим/нет и причинами, победителя, фронт и пути к артефактам-источникам.

- [ ] **Step 4: Добавить CLI-команду**

```python
@app.command("select")
def select_command(
    bundle: str = typer.Option(..., "--bundle", help="каталог evidence-бандла"),
    horizon: int = typer.Option(1, "--horizon"),
    max_latency_us: Optional[float] = typer.Option(None, "--max-latency-us"),
    max_state_bytes: Optional[int] = typer.Option(None, "--max-state-bytes"),
    max_model_bytes: Optional[int] = typer.Option(None, "--max-model-bytes"),
    max_energy_mj: Optional[float] = typer.Option(None, "--max-energy-mj"),
    output: Optional[str] = typer.Option(None, "--output", help="куда записать Markdown"),
) -> None:
    """Найти Pareto-оптимальную модель под ограничения устройства."""
```

Печатает Markdown в stdout; с `--output` дополнительно пишет файл.

- [ ] **Step 5: Написать тест CLI**

```python
# tests/test_cli.py
def test_select_command_prints_a_report(tmp_path, fake_bundle):
    result = runner.invoke(app, ["select", "--bundle", str(fake_bundle),
                                 "--horizon", "1", "--max-latency-us", "5"])
    assert result.exit_code == 0
    assert "ridge_ar" in result.stdout
```

- [ ] **Step 6: Прогнать тесты**

Run: `poetry run pytest tests/test_selection.py tests/test_cli.py -v`
Expected: все зелёные.

- [ ] **Step 7: Сгенерировать отчёт выбора для бандла**

Run:
```bash
poetry run rcbench select --bundle reports/jmlc_2026 --horizon 1 \
    --max-latency-us 25 --max-state-bytes 4096 \
    --output reports/jmlc_2026/selection.md
```
Expected: файл создан, победитель и причины отказов совпадают с таблицей качества
и профилями (проверить глазами по `aggregates/matrix_table.csv`).

- [ ] **Step 8: Коммит**

```bash
bash scripts/verify.sh quick
git add src/rc_bench/reporting/selection.py src/rc_bench/cli/app.py reports/jmlc_2026/selection.md tests/
git commit -m "feat(reporting): select the Pareto-optimal model under device constraints (SELECT-001)"
```

---

### Task 10: API-001 — сервисный контур принимает JMLC-спеки и проверяет владельца

**Files:**
- Modify: `src/rc_bench/main.py`
- Create: `tests/test_api_ownership.py`
- Modify: `tests/test_flow.py` при необходимости

**Interfaces:**
- Consumes: `ExperimentSpec.model_type` / `model_family` (уже существуют в `core/schema.py`).
- Produces: `create_experiment` принимает baseline-спеки; `list_experiments`,
  `get_experiment`, `compare_experiments`, `plot_experiment` возвращают только
  эксперименты текущего пользователя.

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_api_ownership.py
"""Сервисный контур заявлен в §4 описания проекта. Два дефекта делают
заявление ложным: baseline-спеки роняют создание эксперимента, а список
экспериментов не фильтруется по владельцу — любой аутентифицированный
пользователь видит чужие запуски."""
import pytest


@pytest.mark.integration
def test_baseline_spec_can_be_created_through_the_api(client, auth_headers):
    payload = {
        "dataset": {"name": "uci_household_power", "length": 12000},
        "baseline": {"type": "persistence"},
        "protocol": {"forecasting_mode": "fixed_horizon", "horizon": 1,
                     "n_seeds": 0, "use_hpo": False, "seasonal_period": 24},
        "readout": {"alpha_grid": [1.0]},
        "seed": None,
    }
    response = client.post("/experiments/", json=payload, headers=auth_headers)
    assert response.status_code == 200, response.text
    assert response.json()["reservoir_type"] == "persistence"


@pytest.mark.integration
def test_list_experiments_returns_only_the_callers_experiments(client, two_users):
    alice, bob = two_users
    client.post("/experiments/", json=_esn_payload(), headers=alice.headers)
    listed = client.get("/experiments/", headers=bob.headers).json()
    assert listed == []


@pytest.mark.integration
def test_reading_another_users_experiment_is_refused(client, two_users):
    alice, bob = two_users
    created = client.post("/experiments/", json=_esn_payload(), headers=alice.headers).json()
    response = client.get(f"/experiments/{created['id']}", headers=bob.headers)
    assert response.status_code == 404
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `poetry run pytest tests/test_api_ownership.py -m integration -v`
Expected: FAIL — `AttributeError: 'NoneType' object has no attribute 'type'` на первом
тесте, пустой фильтр на остальных.
Если integration-тесты требуют PostgreSQL/Redis, поднять их
(`docker compose up -d db redis`) — это единственная задача плана, которой они нужны.

- [ ] **Step 3: Починить `create_experiment`**

```python
    new_experiment = Experiment(
        # ExperimentSpec.model_type уже нормализует обе ветки — reservoir и
        # baseline, — поэтому JMLC-конфиг проходит через API без изменений.
        reservoir_type=experiment_data.model_type,
        dataset_name=experiment_data.dataset.name,
        config=experiment_data.model_dump(),
        status=ExperimentStatus.QUEUED,
        owner_id=current_user.id,
    )
```

- [ ] **Step 4: Добавить фильтрацию по владельцу**

Во все четыре читающих эндпоинта добавить `.where(Experiment.owner_id == current_user.id)`;
для `get_experiment`/`plot_experiment` — 404 (не 403), чтобы не раскрывать существование
чужого идентификатора.

- [ ] **Step 5: Прогнать тесты**

Run: `poetry run pytest tests/test_api_ownership.py -m integration -v && poetry run pytest -q -m "not integration"`
Expected: новые тесты зелёные, unit-гейт не сломан.

- [ ] **Step 6: Проверить контур целиком**

Run:
```bash
docker compose up -d --build
curl -s localhost/  # через nginx
docker compose logs --tail=20 worker
docker compose down
```
Expected: корневой эндпоинт отвечает, worker поднялся без ошибок импорта.
Результат (включая версии, если что-то не поднялось) записать в handoff.

- [ ] **Step 7: Коммит**

```bash
bash scripts/verify.sh quick
git add src/rc_bench/main.py tests/test_api_ownership.py
git commit -m "fix(api): accept baseline specs and scope experiment reads to their owner (API-001)"
```

---

### Task 11: EXP-004 — полный перепрогон evidence с энергией, прокси и двумя режимами

**Files:**
- Modify: `reports/jmlc_2026/**` (артефакты)
- Modify: `scripts/run_profiling.py` (при необходимости — прокинуть новые параметры)
- Modify: `scripts/verify.sh` (новые обязательные файлы бандла)

**Interfaces:**
- Consumes: всё, что сделано в задачах 2–9.
- Produces: `reports/jmlc_2026/{fair,best_effort}/runs/*.json`, `profiles/*.json` с
  energy/activity, `aggregates/matrix_table*.{json,csv}`, `plots/pareto_quality_*.png`
  (включая energy, если счётчик доступен), `selection.md`, `hardware_profile.json`.

- [ ] **Step 1: Убедиться, что данные на месте и хеш сходится**

Run: `poetry run python scripts/download_jmlc_data.py`
Expected: файл на месте, SHA-256 совпадает с `configs/jmlc/dataset_manifest.json`.

- [ ] **Step 2: Перепрогнать fair-матрицу**

Run:
```bash
poetry run python scripts/run_jmlc_matrix.py \
    --config configs/jmlc/fair.yaml --output reports/jmlc_2026/fair
```
Expected: 14/14 completed, ~2 минуты. `config_hash` каждой ячейки **совпадает с
предыдущим** — иначе Task 6 Step 3 сделан неверно, и это блокер, а не мелочь:

```bash
git diff --stat reports/jmlc_2026/fair/runs/
git diff reports/jmlc_2026/fair/runs/reservoir_esn_h1.json | grep -c config_hash
```
Expected: `config_hash` в diff не участвует.

- [ ] **Step 3: Прогнать best-effort матрицу**

Run:
```bash
poetry run python scripts/run_jmlc_matrix.py \
    --config configs/jmlc/best_effort.yaml --output reports/jmlc_2026/best_effort
```
Expected: 14/14 completed. Записать фактическое время прогона.

- [ ] **Step 4: Профилирование обоих режимов**

Run:
```bash
poetry run python scripts/run_profiling.py \
    --config configs/jmlc/fair.yaml --runs reports/jmlc_2026/fair/runs \
    --output reports/jmlc_2026/profiles \
    --hardware-output reports/jmlc_2026/hardware_profile.json
```
Expected: 14 ячеек, в каждой `energy.status` и блок `activity`.
Проверка здравого смысла перед публикацией:

```bash
poetry run python -c "
import json
s=json.load(open('reports/jmlc_2026/profiles/summary.json'))
for e in sorted(s,key=lambda x:(x['horizon'],x['model'])):
    print(e['model'], e['horizon'], e.get('energy_status'),
          e.get('net_energy_per_inference_mj'), e.get('total_macs'),
          e.get('spikes_per_step'))
"
```
Ожидаемые соотношения (иначе — искать дефект измерения, а не публиковать):
persistence дешевле ridge_ar; ridge_ar дешевле esn; lsm дороже esn по `total_macs`;
энергия монотонна по задержке в пределах порядка величины; `spikes_per_step` не `None`
только у LSM.

- [ ] **Step 5: Валидация и агрегаты**

Run:
```bash
poetry run python scripts/validate_evidence.py reports/jmlc_2026 \
    --modes fair,best_effort --write
```
Expected: OK для обоих режимов, 14 cells каждый.

- [ ] **Step 6: Графики и отчёт выбора**

Run:
```bash
poetry run python scripts/plot_pareto.py --bundle reports/jmlc_2026 --output reports/jmlc_2026/plots
poetry run rcbench select --bundle reports/jmlc_2026 --horizon 1 \
    --max-latency-us 25 --max-state-bytes 4096 --output reports/jmlc_2026/selection.md
```
Expected: три графика (`pareto_quality_latency/memory/energy.png`), либо явный
`skipped: ...` для energy, если счётчик недоступен.

- [ ] **Step 7: Добавить новые артефакты в release-гейт**

В `run_release()` дописать:

```bash
  require_file "reports/jmlc_2026/selection.md"
  require_file "reports/jmlc_2026/aggregates/matrix_table_best_effort.json"
```

- [ ] **Step 8: Полный гейт**

Run: `bash scripts/verify.sh release`
Expected: зелёный целиком, включая санитизацию.

- [ ] **Step 9: Коммит**

```bash
git add reports/jmlc_2026 scripts/verify.sh
git commit -m "chore(evidence): rerun the bundle with energy, activity proxies and both modes (EXP-004)"
```

---

### Task 12: DOC-003 — документация, разметка planned/implemented и обложка репозитория

**Files:**
- Modify: `README.md`
- Modify: `reports/jmlc_2026/README.md`
- Modify: `docs/PROJECT_CONTRACT.md`
- Modify: `docs/DECISIONS.md` (**основной агент**)
- Modify: `AI_USAGE.md`
- Modify: `docs/agent/BACKLOG.md` (**основной агент**)
- Modify: `tests/test_repo_metadata.py`

**Interfaces:**
- Consumes: все артефакты Task 11.
- Produces: README со строгим разделением implemented/planned/unavailable; DEC-021…DEC-024.

- [ ] **Step 1: Написать падающий тест на разметку README**

```python
# tests/test_repo_metadata.py
def test_readme_separates_implemented_from_planned():
    """§2 описания проекта обещает читателю, что планируемое отделено от
    сделанного. Проверяем наличие самой разметки, а не формулировок."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Статус реализации" in readme
    for marker in ("Реализовано", "Запланировано", "Недоступно"):
        assert marker in readme


def test_readme_numbers_point_at_evidence_artifacts():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for artifact in (
        "reports/jmlc_2026/aggregates/matrix_table.csv",
        "reports/jmlc_2026/profiles/summary.json",
        "reports/jmlc_2026/selection.md",
    ):
        assert artifact in readme, f"{artifact} нигде не назван как источник чисел"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `poetry run pytest tests/test_repo_metadata.py -v`
Expected: FAIL — секции «Статус реализации» нет.

- [ ] **Step 3: Переписать шапку README**

Добавить сразу после описания секцию `## Статус реализации` с тремя списками:

- **Реализовано и подтверждено артефактами** — семь архитектур, реальный ряд UCI,
  fair и best-effort матрицы, baselines, MASE/MAE skill, ресурсный профиль
  (время обучения, p50/p95, throughput, peak RSS, размеры), прокси активности,
  измеренная энергия (если счётчик доступен), Pareto-выбор под ограничения,
  evidence-бандл и release-гейт, CI, demo за одну команду.
- **Запланировано** — то, что ещё не сделано, с честной формулировкой без будущего
  времени в описании возможностей (например, Jetson-профиль, изоляция peak RSS через
  spawn-процесс, PROXY для deep_esn/fhn/qrc).
- **Недоступно** — с причиной (`energy` при отсутствии счётчика; Jetson за отсутствием
  устройства).

Каждый пункт «реализовано» ссылается на артефакт или модуль.

- [ ] **Step 4: Обновить README бандла**

В `reports/jmlc_2026/README.md`: раздел про энергию (протокол, вычитание простоя,
ограничение «это энергия пакета, а не модели»), раздел про прокси активности с явным
предупреждением, что они не конвертируются в джоули, раздел сравнения fair и best-effort
(насколько индивидуальная настройка меняет выводы), ссылка на `selection.md`.

- [ ] **Step 5: Описать расхождение раскладки бандла в контракте**

В `docs/PROJECT_CONTRACT.md` — абзац о том, что `specs/frozen/`, `specs/resolved/`, `hpo/`
физически лежат внутри каждого RunRecord (`fair/runs/*.json`), а не отдельными каталогами:
информация не потеряна, но проверяющему нужно знать, где смотреть. Плюс новые каталоги
`best_effort/` и `profiles/`, и файл `selection.md`.

- [ ] **Step 6: Записать решения (основной агент)**

В `docs/DECISIONS.md`:

- **DEC-021** — энергия измеряется по package-домену RAPL с вычитанием простоя равной
  длительности; публикуются обе величины; окно измерения не короче 2 с; это не
  изолированное измерение модели, и так и сказано.
- **DEC-022** — прокси активности (MAC, разреженность, спайки) публикуются в отдельном
  блоке `activity` и никогда не пересчитываются в энергию.
- **DEC-023** — режим протокола входит в спецификацию и в `config_hash`, но значение
  по умолчанию `fair` выбрасывается из payload, чтобы опубликованные хеши остались
  прежними; fair и best_effort никогда не сводятся в одну таблицу.
- **DEC-024** — выбор под ограничения делается по опубликованному бандлу; ограничение
  без измерения объявляется неприменимым, а не отбрасывает всех кандидатов.

- [ ] **Step 7: Обновить `AI_USAGE.md`**

Добавить строки за эту сессию: какие задачи делал основной агент, какие — сабагенты,
что проверено руками (энергия на реальном железе, docker compose, соответствие
config_hash), и какие числа получены машиной, а не переписаны.

- [ ] **Step 8: Закрыть задачи в BACKLOG (основной агент)**

Проставить `DONE` для REPO-001, PROXY-001, PROXY-002, ENERGY-002, API-001; добавить
строки MODE-001, MODE-002, PROF-004, SELECT-001, ENERGY-003, EXP-004, DOC-003, REL-002
с их статусами.

- [ ] **Step 9: Обновить описание репозитория на GitHub**

Run:
```bash
gh repo edit ReFlex-cmd/rc-bench \
  --description "rc-bench — воспроизводимая платформа сравнения резервуарных моделей прогнозирования временных рядов: 7 архитектур, честный протокол, реальные данные UCI, ресурсный и энергетический профиль." \
  --add-topic reservoir-computing --add-topic echo-state-network \
  --add-topic time-series-forecasting --add-topic benchmark --add-topic edge-ai
```
Expected: описание и темы обновлены. Это изменение публичной страницы репозитория —
выполнять только после того, как содержимое `main` ему соответствует (то есть после
Task 13), либо согласовать порядок с человеком.

- [ ] **Step 10: Прогнать тесты и закоммитить**

```bash
poetry run pytest tests/test_repo_metadata.py -v
bash scripts/verify.sh release
git add README.md reports/jmlc_2026/README.md docs/ AI_USAGE.md tests/test_repo_metadata.py
git commit -m "docs: separate implemented, planned and unavailable; document energy and mode protocols (DOC-003)"
```

---

### Task 13: REL-002 — релиз: контур разработки отдельно от обложки

**Files:**
- Create: ветка `release/jmlc-2026-conformance`
- Modify: ничего в `main` напрямую

**Interfaces:**
- Consumes: всё дерево `dev`.
- Produces: PR `release/... → main`, повторяющий разделение из `1c912a5`.

- [ ] **Step 1: Убедиться, что гейт зелёный и дерево чистое**

Run: `bash scripts/verify.sh release && git status --short --branch`
Expected: гейт зелёный, дерево чистое, ветка `dev`.

- [ ] **Step 2: Проверить, что именно уходит в публичную ветку**

Run: `git diff --name-status origin/main..dev | grep -v '^A\treports/'`
Expected: список изменений; процессные файлы агентов видны и подлежат исключению.

- [ ] **Step 3: Создать релизную ветку и убрать из неё контур разработки**

```bash
git switch -c release/jmlc-2026-conformance dev
git rm -r --cached AGENTS.md CODEX_START_PROMPT.md HARNESS_README.md SESSION_HANDOFF.md docs/agent
git commit -m "chore(release): drop development-only material from the published branch"
```

Ровно то же разделение, что применил `1c912a5`: в `main` уходит научный и
пользовательский контур (код, конфиги, тесты, `docs/PROJECT_CONTRACT.md`,
`docs/DECISIONS.md`, evidence-бандл, README, LICENSE, AI_USAGE), а процессные документы
агентов остаются только в `dev`.

- [ ] **Step 4: Проверить, что гейт зелёный и на релизной ветке**

Run: `bash scripts/verify.sh release`
Expected: зелёный — удалённые файлы не были нужны тестам.

- [ ] **Step 5: Push и PR (только после подтверждения человека)**

```bash
git push -u origin release/jmlc-2026-conformance
gh pr create --base main --head release/jmlc-2026-conformance \
  --title "JMLC 2026: energy contour, activity proxies, best-effort mode and constrained selection" \
  --body "$(cat <<'EOF'
Приводит репозиторий в полное соответствие описанию проекта (03_Project_Description_rc-bench.pdf).

## Что закрыто
- ENERGY-002/003 — измеренная энергия через Intel RAPL: mJ/inference, samples/J, energy-delay product, Pareto качество-энергия
- PROXY-001/002 — число операций, разреженность состояния, спайки и синаптические события LSM
- MODE-001/002 — режим best-effort рядом с fair, раздельная валидация и агрегаты
- SELECT-001 — выбор Pareto-оптимальной архитектуры под ограничения устройства
- PROF-004 — время обучения в опубликованной таблице
- API-001 — сервисный контур принимает JMLC-спеки и проверяет владельца
- REPO-001 — файл лицензии MIT
- DOC-003 — разделение implemented / planned / unavailable

## Проверено
- `bash scripts/verify.sh release` — зелёный
- `scripts/validate_evidence.py` — OK для fair и best_effort
- энергия измерена на реальном счётчике RAPL, не оценена

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Обновить handoff**

Переписать `SESSION_HANDOFF.md` под текущее состояние: что сделано, какие числа
получены, что осталось (PRES-001, ENERGY-002 для Jetson, изоляция peak RSS),
известные отклонения.

```bash
git switch dev
git add SESSION_HANDOFF.md docs/agent/BACKLOG.md
git commit -m "docs: session-4 handoff after the PDF-conformance pass"
```

---

## Самопроверка плана

**Покрытие описания проекта.** Каждый пункт таблицы расхождений привязан к задаче:
1→Task 1, 2→Task 2, 3→Task 3, 4→Task 4+5, 5→Task 6+7, 6→Task 8, 7→Task 9, 8→Task 10,
9→Task 5 Step 7 + Task 11 Step 6, 10→Task 12, 11→Task 12 Step 5. Подтверждённые
без доработок заявления перечислены в первой таблице и не требуют задач.

**Согласованность имён между задачами.** `step_operation_counts()` (Task 2) вызывается
в `count_step_operations` (Task 2 Step 5) и в профиле (Task 5 Step 4). `spike_stats()`
(Task 3) читается через `spiking_activity` (Task 3 Step 4), используется в Task 5.
`measure_energy(step_fn, *, p50_ns, ...)` (Task 4) вызывается в Task 5 Step 4 с
`p50_ns=deployable_latency.p50_ns`. Ключи `net_energy_per_inference_mj`,
`total_macs`, `spikes_per_step` из `_summary_entry` (Task 5 Step 5) читаются
`pareto.COST_AXES["energy"]` (Task 5 Step 7) и `selection.select` (Task 9).
`ProtocolSpec.mode` (Task 6) читается `validate_records` (Task 7) и `CellRow.mode`.

**Риски, которые план обязан удержать.**
1. Новое поле `ProtocolSpec.mode` меняет `config_hash` всех опубликованных ячеек —
   снимается выбрасыванием значения по умолчанию (Task 6 Step 3) и проверяется
   тестом против реального опубликованного хеша (Task 6 Step 1) и повторным
   прогоном (Task 11 Step 2).
2. Энергия шага в десятки микросекунд короче периода обновления RAPL — снимается
   окном ≥ 2 с (Task 4).
3. Энергия пакета включает статическое потребление — снимается вычитанием простоя
   и публикацией обеих величин (Task 4, DEC-021).
4. Прокси могут быть прочитаны как энергия — снимается отдельным блоком `activity`,
   явными `note`, DEC-022 и формулировками README (Task 12).
