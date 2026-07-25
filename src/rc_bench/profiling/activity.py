"""Прокси активности вычислений (PROXY-001/PROXY-002).

Это АНАЛИТИЧЕСКИЕ оценки: они выводятся из формы весов и наблюдаемого
состояния, а не измеряются счётчиком. Они не являются энергией и не
пересчитываются в джоули — ни через TDP, ни как-либо ещё (DEC-007, §5 PDF).
Публикуются рядом с энергией, но в отдельном блоке ``activity``.

Конвенция MAC-счёта (её придерживается каждая реализация
``step_operation_counts()`` в ``core/reservoirs/*_service.py``):

- MAC — одна операция умножения-с-накоплением. Скалярно-векторное
  умножение (например, ``alpha * x``, где ``alpha`` — скаляр, а ``x`` —
  вектор из ``units`` элементов) стоит один MAC на юнит; произведение
  матрицы на вектор стоит один MAC на ненулевой вес.
- Простое сложение, не сопряжённое с умножением (например, добавление
  несмасштабированного bias-члена, или прибавление спайков напрямую к
  бегущей сумме), отдельно не считается — счёт отслеживает
  масштабирующую/смешивающую арифметику, а не каждый ``+``.
- Применение нелинейной функции (tanh, пороговое сравнение, ...)
  считается отдельно в ``reservoir_nonlinearities``, один раз на юнит на
  вызов — и никогда не сворачивается в ``reservoir_macs``.

Docstring/комментарии каждой модели должны перечислять слагаемые её
``step()`` и показывать, как они ложатся на эту конвенцию, — так формула
остаётся проверяемой по коду, который она описывает.
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


def count_step_operations(
    reservoir: BaseReservoir, *, readout_input_dim: int
) -> Dict[str, Any]:
    """Аналитическая стоимость одного шага вывода: резервуар + Ridge-выход.

    Args:
        reservoir: резервуар, чей ``step_operation_counts()`` описывает
            стоимость обновления состояния (см. конвенцию MAC-счёта в
            docstring модуля).
        readout_input_dim: размерность вектора признаков, подаваемого на
            вход Ridge-регрессии — то есть число коэффициентов на один
            выход (обычно совпадает с числом юнитов резервуара, а НЕ с
            числом выходов модели). Ridge-выход в этом проекте скалярный,
            поэтому ``readout_macs`` равен ``readout_input_dim`` напрямую
            (один MAC на коэффициент); для гипотетического многовыходного
            readout это равенство перестанет быть верным и потребует
            отдельного учёта — здесь оно не подразумевается автоматически.

    Raises:
        ValueError: если ``readout_input_dim`` отрицательный.
    """
    if readout_input_dim < 0:
        raise ValueError(f"readout_input_dim must be >= 0, got {readout_input_dim}")
    counts = reservoir.step_operation_counts()
    if counts is None:
        return {
            "backend": "unavailable",
            "reason": (
                f"{type(reservoir).__name__}.step_operation_counts() is not "
                "implemented; an operation count is not guessed"
            ),
        }
    readout_macs = int(readout_input_dim)
    return {
        "backend": "analytic",
        **{k: int(v) for k, v in counts.items()},
        "readout_macs": readout_macs,
        "total_macs": int(counts["reservoir_macs"]) + readout_macs,
        "note": "analytic MAC estimate from weight shapes; not an energy measurement",
    }
