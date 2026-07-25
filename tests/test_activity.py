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
