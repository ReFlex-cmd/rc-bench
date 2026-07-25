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
