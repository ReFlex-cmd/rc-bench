"""Прокси активности — аналитические оценки, а не измерения. Тесты фиксируют
именно это: числа должны выводиться из формы весов и состояния, быть
воспроизводимыми и никогда не превращаться в джоули.

Ассерты на счёт операций вычисляют ожидание из тех же слагаемых, что
перечислены в комментариях соответствующего ``step_operation_counts()`` —
см. конвенцию MAC-счёта в docstring ``rc_bench.profiling.activity`` (MAC
считается на каждое скалярно-векторное умножение и на каждый ненулевой вес
матрицы; несмасштабированные сложения не считаются отдельно)."""
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
    counts = count_step_operations(reservoir, readout_input_dim=50)
    nnz = int(np.count_nonzero(reservoir._W_rec))
    # W_rec @ x (nnz MAC) + W_in * u (units MAC) + leak-смешение (2*units MAC)
    assert counts["backend"] == "analytic"
    assert counts["reservoir_macs"] == nnz + 3 * 50
    assert counts["reservoir_nonlinearities"] == 50
    assert counts["readout_macs"] == 50
    assert counts["total_macs"] == nnz + 3 * 50 + 50


def test_lsm_operation_count_matches_dense_arithmetic():
    reservoir = get_reservoir("lsm", {"units": 60, "density": 0.1, "seed": 7})
    counts = count_step_operations(reservoir, readout_input_dim=60)
    nnz = int(np.count_nonzero(reservoir._W_rec))
    # W_rec @ s (nnz MAC) + W_in * u (units MAC)
    # + мембранный распад (2*units MAC) + синаптический распад (units MAC)
    assert counts["backend"] == "analytic"
    assert counts["reservoir_macs"] == nnz + 4 * 60
    assert counts["reservoir_nonlinearities"] == 60
    assert counts["readout_macs"] == 60
    assert counts["total_macs"] == nnz + 4 * 60 + 60


def test_logistic_operation_count_has_no_recurrent_matrix():
    reservoir = get_reservoir("logistic", {"units": 32, "seed": 7})
    counts = count_step_operations(reservoir, readout_input_dim=32)
    # Узлы не связаны между собой: рекуррентной матрицы нет вовсе.
    # r*x*(1-x) (2*units MAC) + coupling*(w_in*u) (2*units MAC)
    assert counts["reservoir_nonzero_recurrent_weights"] == 0
    assert counts["reservoir_macs"] == 4 * 32
    assert counts["total_macs"] == 4 * 32 + 32


def test_esn_operation_count_matches_reservoirpy_update_rule():
    reservoir = get_reservoir("esn", {"n_units": 40, "rc_connectivity": 0.2, "seed": 7})
    reservoir.reset_state()  # reservoirpy инициализирует веса лениво
    counts = count_step_operations(reservoir, readout_input_dim=40)
    w_nnz = reservoir._res.W.nnz
    win_nnz = int(np.count_nonzero(reservoir._res.Win))
    # reservoirpy: next = f(W@s + Win@x + bias); x = (1-lr)*s + lr*next.
    # W @ s (w_nnz MAC) + Win @ x (win_nnz MAC) + leak-смешение (2*units MAC);
    # "+ bias" не считается — этот проект никогда не задаёт bias-вектор, он
    # остаётся несмасштабированным нулём по умолчанию reservoirpy.
    assert counts["reservoir_nonzero_recurrent_weights"] == w_nnz
    assert counts["reservoir_macs"] == w_nnz + win_nnz + 2 * 40
    assert counts["reservoir_nonlinearities"] == 40


def test_unknown_reservoir_reports_unavailable_instead_of_guessing():
    reservoir = get_reservoir("fhn", {"units": 20, "seed": 7})
    counts = count_step_operations(reservoir, readout_input_dim=20)
    assert counts["backend"] == "unavailable"
    assert "not implemented" in counts["reason"]


def test_count_step_operations_rejects_negative_readout_dim():
    reservoir = get_reservoir("logistic", {"units": 8, "seed": 7})
    with pytest.raises(ValueError):
        count_step_operations(reservoir, readout_input_dim=-1)
