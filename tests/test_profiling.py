"""Tests for rc_bench.profiling (PROF-001: latency, PROF-002: memory/hardware).

Design notes:
- Latency tests monkeypatch ``time.perf_counter_ns`` to get fully
  deterministic, instant timing math (no real sleeping, no flakiness).
- The subprocess RSS test uses small (~20 MiB) allocations so it stays fast
  while still being a meaningful "isolated process, real RSS" measurement.
- Hardware profile tests assert the sanitization contract from DEC-008:
  no hostname, no username, no absolute paths anywhere in the serialized
  output.
"""
from __future__ import annotations

import getpass
import json
import pickle
import socket
import sys

import numpy as np
import pytest

from rc_bench.profiling.latency import LatencyProfile, measure_latency
from rc_bench.profiling.memory import (
    PeakRSSProfile,
    measure_peak_rss_subprocess,
    serialized_model_bytes,
    working_state_bytes,
)
from rc_bench.profiling.hardware import HardwareProfile, get_hardware_profile


# ---------------------------------------------------------------------------
# PROF-001: latency
# ---------------------------------------------------------------------------


class _CountingStep:
    """Trivial step callable that counts invocations; stands in for a real
    model's single-step inference call without depending on any reservoir."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls


def test_measure_latency_runs_warmup_then_measured_steps():
    step = _CountingStep()
    profile = measure_latency(step, warmup=5, n_steps=20, single_threaded=False)

    assert isinstance(profile, LatencyProfile)
    # 5 warmup + 20 measured = 25 total invocations of step_fn.
    assert step.calls == 25
    assert profile.warmup_steps == 5
    assert profile.n_steps == 20
    assert profile.raw_ns is not None
    assert len(profile.raw_ns) == 20


def test_measure_latency_keep_raw_false_drops_raw_samples():
    step = _CountingStep()
    profile = measure_latency(
        step, warmup=2, n_steps=10, keep_raw=False, single_threaded=False
    )
    assert profile.raw_ns is None
    # Aggregates must still be populated.
    assert profile.p50_ns >= 0
    assert profile.p95_ns >= profile.p50_ns


def test_measure_latency_rejects_invalid_arguments():
    step = _CountingStep()
    with pytest.raises(ValueError):
        measure_latency(step, warmup=0, n_steps=0)
    with pytest.raises(ValueError):
        measure_latency(step, warmup=-1, n_steps=10)


def test_measure_latency_never_calls_anything_but_step_fn():
    """Guards against accidentally mixing train/HPO time into inference
    latency: the only callable measure_latency may invoke is step_fn."""
    other = _CountingStep()
    step = _CountingStep()
    measure_latency(step, warmup=3, n_steps=7, single_threaded=False)
    assert other.calls == 0
    assert step.calls == 10


def test_measure_latency_deterministic_percentiles(monkeypatch):
    """Fully deterministic clock: perf_counter_ns() yields a fixed sequence,
    so raw deltas and percentiles are exactly predictable."""
    import rc_bench.profiling.latency as latency_mod

    # Each step consumes two clock reads (start, end). Deltas (ns): 10..100.
    deltas = list(range(10, 101, 10))  # 10 values: 10,20,...,100
    clock_values = []
    t = 0
    for d in deltas:
        clock_values.append(t)
        t += d
        clock_values.append(t)
    it = iter(clock_values)
    monkeypatch.setattr(latency_mod.time, "perf_counter_ns", lambda: next(it))

    profile = measure_latency(
        lambda: None, warmup=0, n_steps=len(deltas), single_threaded=False
    )

    assert profile.raw_ns == tuple(deltas)
    assert profile.min_ns == 10.0
    assert profile.max_ns == 100.0
    # Linear-interpolated median of 10..100 step 10 (n=10) at rank 4.5 -> 55.0
    assert profile.p50_ns == pytest.approx(55.0)
    # p95 at rank 0.95*9=8.55 -> data[8]=90 + 0.55*(100-90) = 95.5
    assert profile.p95_ns == pytest.approx(95.5)
    total_s = sum(deltas) / 1e9
    assert profile.throughput_samples_per_s == pytest.approx(len(deltas) / total_s)


def test_measure_latency_stabilize_runs_extra_batches(monkeypatch):
    """With stabilize=True, more batches are pulled until p50 is stable or
    max_steps is hit. This test keeps everything deterministic and tiny."""
    import rc_bench.profiling.latency as latency_mod

    # First batch of 4 has wildly varying deltas -> not stable.
    # Second batch of 4 repeats the first batch's deltas -> stable (0 change).
    batch = [10, 20, 30, 40]
    deltas = batch + batch + batch  # allow up to 3 batches before hitting cap
    clock_values = []
    t = 0
    for d in deltas:
        clock_values.append(t)
        t += d
        clock_values.append(t)
    it = iter(clock_values)
    monkeypatch.setattr(latency_mod.time, "perf_counter_ns", lambda: next(it))

    profile = measure_latency(
        lambda: None,
        warmup=0,
        n_steps=4,
        single_threaded=False,
        stabilize=True,
        max_steps=12,
        stability_rtol=0.01,
    )
    # Should stop once two consecutive batches agree (after batch 2), i.e. 8
    # measured steps, well before the 12-step cap.
    assert profile.n_steps == 8
    assert profile.n_steps <= 12


# ---------------------------------------------------------------------------
# PROF-002: serialized model / working-state sizes
# ---------------------------------------------------------------------------


def test_serialized_model_bytes_matches_pickle_length():
    model = {"alpha": 1.0, "coef": [1.0, 2.0, 3.0]}
    size = serialized_model_bytes(model)
    assert size == len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    assert size > 0


def test_working_state_bytes_uses_numpy_nbytes():
    state = np.zeros((100,), dtype=np.float64)
    assert working_state_bytes(state) == 800


def test_working_state_bytes_falls_back_to_pickle_for_non_array():
    state = {"h": [1, 2, 3]}
    assert working_state_bytes(state) == len(pickle.dumps(state))


# ---------------------------------------------------------------------------
# PROF-002: isolated peak RSS (real subprocess)
# ---------------------------------------------------------------------------

_ALLOC_BYTES = 20 * 1024 * 1024  # 20 MiB: fast to allocate, easy to detect


def _allocate_and_touch():
    """Allocate a real (not tracemalloc-visible) native buffer and touch every
    page so it is actually resident, then keep it alive until return."""
    buf = bytearray(_ALLOC_BYTES)
    for i in range(0, len(buf), 4096):
        buf[i] = 1
    return len(buf)


def test_measure_peak_rss_subprocess_reflects_real_allocation():
    profile = measure_peak_rss_subprocess(_allocate_and_touch)

    assert isinstance(profile, PeakRSSProfile)
    assert profile.peak_rss_bytes > 0
    # Generous lower margin below the 20 MiB actually allocated to absorb
    # measurement granularity/interpreter overhead differences.
    assert profile.peak_rss_bytes >= int(_ALLOC_BYTES * 0.5)


def _raise_value_error():
    raise ValueError("boom-from-child")


def test_measure_peak_rss_subprocess_propagates_child_errors():
    with pytest.raises(RuntimeError, match="boom-from-child"):
        measure_peak_rss_subprocess(_raise_value_error)


# ---------------------------------------------------------------------------
# Hardware/software profile + DEC-008 sanitization
# ---------------------------------------------------------------------------


def test_hardware_profile_has_expected_fields():
    profile = get_hardware_profile()
    assert isinstance(profile, HardwareProfile)
    assert profile.logical_cores >= 1
    assert profile.os_name  # non-empty
    assert profile.python_version
    assert "numpy" in profile.library_versions
    assert profile.library_versions["numpy"]
    assert "scikit-learn" in profile.library_versions
    if profile.physical_cores is not None:
        assert 1 <= profile.physical_cores <= profile.logical_cores
    if profile.total_ram_bytes is not None:
        assert profile.total_ram_bytes > 0


def test_hardware_profile_contains_no_sensitive_identifiers():
    profile = get_hardware_profile()
    dumped = json.dumps(profile.to_dict())

    hostname = socket.gethostname()
    username = getpass.getuser()

    if hostname:
        assert hostname not in dumped
    if username:
        assert username not in dumped
    # No absolute POSIX-style paths anywhere in the payload.
    assert "/home/" not in dumped
    assert sys.prefix not in dumped
    assert sys.executable not in dumped


def test_hardware_profile_is_json_serializable():
    profile = get_hardware_profile()
    # Must not raise; also must be a plain dict of JSON-safe values.
    text = json.dumps(profile.to_dict())
    assert isinstance(text, str)
