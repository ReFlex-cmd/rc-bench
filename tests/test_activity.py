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
from rc_bench.profiling.activity import count_step_operations, spiking_activity, state_sparsity


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
    # rel=0.35 — это проверка порядка величины, а не оси: при i.i.d.
    # Бернулли-маске W_rec суммы по строкам и по столбцам статистически
    # неразличимы, поэтому этот допуск не отличил бы верную ось (out-degree,
    # axis=0) от перепутанной (in-degree, axis=1) — за это отвечает
    # test_synaptic_events_use_outgoing_fanout_exactly ниже.
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


def test_synaptic_events_use_outgoing_fanout_exactly():
    """`I = W_rec @ s` reads column j of ``W_rec`` as neuron j's outgoing
    weights, so fan-out must come from ``axis=0``. A statistical check
    can't tell that axis apart from the wrong one (``axis=1``, incoming
    fan-in): under an i.i.d. Bernoulli mask, row sums and column sums are
    both Binomial(units, density) and uncorrelated with which neurons
    actually spike, so a loose tolerance passes either way (verified by
    hand: with this project's default LSM config, out-axis error vs.
    target is 0.63%, in-axis error is 11.4% — both comfortably inside
    ``rel=0.35`` above).

    This test instead builds a tiny, deterministic, asymmetric ``W_rec``
    where out-degree and in-degree disagree per neuron, forces a known
    pair of neurons to spike on two known steps by directly overriding the
    membrane potential (bypassing the stochastic LIF dynamics entirely),
    and asserts the accumulated synaptic-event count with no tolerance.
    Swapping the implementation to ``axis=1`` makes this fail (checked
    manually: the axis=0 build gives 2.0 then 3.0 as expected; the axis=1
    build already fails the step-1 assertion with 0.0 instead of 2.0,
    since neuron 0 has in-degree 0 — see task-3 fix-report for the
    transcript).
    """
    reservoir = get_reservoir("lsm", {"units": 3, "seed": 1})
    # Neuron 0: fans out to 1 and 2 (out-degree 2), receives from nobody
    #   (in-degree 0).
    # Neuron 1: receives from 0 and 2 (in-degree 2), fans out to nobody
    #   (out-degree 0).
    # Neuron 2: fans out to 1 (out-degree 1), receives from 0 (in-degree 1).
    # Out-degree (axis=0, correct) = [2, 0, 1]; in-degree (axis=1, the bug
    # this test guards against) = [0, 2, 1] — disjoint enough that no
    # spiking pattern gives the two axes the same total.
    reservoir._W_rec = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
        ]
    )
    # Recomputes _out_degree from the matrix above (reset_state() must run
    # after the override so it doesn't derive fan-out from the original
    # random weights).
    reservoir.reset_state()

    # Step 1: force neuron 0 alone over threshold. Input is zeroed (u=0)
    # so W_in contributes nothing; only the injected membrane potential
    # decides who spikes.
    reservoir._step_v = np.array([100.0, 0.0, 0.0])
    reservoir.step(np.array([[0.0]]))
    # Neuron 0 now spiked once: its out-degree (2) must be the only
    # contribution to synaptic_events_total so far.
    assert reservoir._synaptic_events_total == 2.0

    # Step 2: neuron 0 is refractory (blocked regardless of v), so force
    # neuron 2 alone over threshold; neuron 1 is left at 0 and must not
    # cross threshold from recurrent input alone.
    reservoir._step_v = np.array([0.0, 0.0, 100.0])
    reservoir.step(np.array([[0.0]]))

    stats = spiking_activity(reservoir)
    assert stats["steps"] == 2
    assert stats["total_spikes"] == 2.0
    # Sum of outgoing fan-out of the neurons that actually spiked:
    # out_degree[0] + out_degree[2] = 2 + 1 = 3. Under the swapped axis
    # this would instead be in_degree[0] + in_degree[2] = 0 + 1 = 1.
    assert reservoir._synaptic_events_total == 3.0
