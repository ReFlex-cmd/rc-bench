"""Single-step inference latency profiling (PROF-001).

Protocol (see docs/PROJECT_CONTRACT.md, "Ресурсный профиль" /
"Протокол latency"):

- single thread if the numeric backend allows it to be controlled
  (best-effort via ``threadpoolctl``, when installed);
- at least 100 warmup steps before measuring;
- at least 1000 measured steps, more if timing has not stabilized;
- timestamps taken with ``time.perf_counter_ns()``;
- raw per-step measurements (or sufficient aggregated statistics) are kept;
- train/HPO time is never mixed into the measured loop — the only callable
  ``measure_latency`` ever invokes is the caller-supplied ``step_fn``.

This module is intentionally generic: ``step_fn`` is any zero-argument
callable that performs one unit of inference work (e.g. a closure around a
reservoir's ``step()`` + readout ``predict()``). Nothing here imports or
depends on any concrete reservoir/readout class.
"""
from __future__ import annotations

import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LatencyProfile:
    """Aggregated result of a :func:`measure_latency` run.

    All ``*_ns`` fields are nanoseconds (matching ``time.perf_counter_ns``).
    ``raw_ns`` holds the per-step deltas in call order, or ``None`` when the
    caller opted out via ``keep_raw=False``.
    """

    warmup_steps: int
    n_steps: int
    p50_ns: float
    p95_ns: float
    mean_ns: float
    min_ns: float
    max_ns: float
    std_ns: float
    throughput_samples_per_s: float
    raw_ns: Optional[Tuple[int, ...]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "n_steps": self.n_steps,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "mean_ns": self.mean_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "std_ns": self.std_ns,
            "throughput_samples_per_s": self.throughput_samples_per_s,
            "raw_ns": list(self.raw_ns) if self.raw_ns is not None else None,
        }


def _percentile(values: Sequence[int], q: float) -> float:
    """Linear-interpolated percentile (matches numpy's default 'linear')."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample set")
    data = sorted(values)
    n = len(data)
    if n == 1:
        return float(data[0])
    rank = (q / 100.0) * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - lower
    return data[lower] + (data[upper] - data[lower]) * frac


def _thread_limiter(single_threaded: bool):
    """Best-effort single-thread pin for BLAS/OMP-backed numeric libraries.

    Uses ``threadpoolctl`` when available (it can retarget already-loaded
    OpenBLAS/MKL/OMP thread pools at runtime, unlike env vars set after
    import). If it is not installed, this is a silent no-op — the protocol
    only requires single-threading "if the library allows it to be
    controlled".
    """
    if not single_threaded:
        return nullcontext()
    try:
        import threadpoolctl
    except ImportError:
        return nullcontext()
    return threadpoolctl.threadpool_limits(limits=1)


def _run_batch(step_fn: Callable[[], Any], n: int, samples: list) -> None:
    for _ in range(n):
        t0 = time.perf_counter_ns()
        step_fn()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)


def measure_latency(
    step_fn: Callable[[], Any],
    *,
    warmup: int = 100,
    n_steps: int = 1000,
    keep_raw: bool = True,
    single_threaded: bool = True,
    stabilize: bool = False,
    max_steps: Optional[int] = None,
    stability_rtol: float = 0.05,
) -> LatencyProfile:
    """Measure single-step latency of ``step_fn`` and return a profile.

    Parameters
    ----------
    step_fn:
        Zero-argument callable performing exactly one inference step. Its
        return value is ignored. It must not itself perform training/HPO —
        this function has no way to detect that, so it is the caller's
        responsibility to pass a pure inference step.
    warmup:
        Number of untimed calls to ``step_fn`` before measurement starts.
        Contract minimum is 100; smaller values are allowed for testing.
    n_steps:
        Number of timed calls in the base measurement window. Contract
        minimum is 1000; smaller values are allowed for testing.
    keep_raw:
        When True (default), the per-step nanosecond deltas are kept on
        ``LatencyProfile.raw_ns``. Set False to drop them and keep only
        aggregated statistics (e.g. for very large ``n_steps``).
    single_threaded:
        Best-effort pin of numeric-library thread pools to 1 (see
        :func:`_thread_limiter`).
    stabilize:
        When True, keep pulling additional batches of ``n_steps`` measured
        calls until the p50 latency changes by no more than
        ``stability_rtol`` between successive batches, or until ``max_steps``
        total measured calls have been made.
    max_steps:
        Upper bound on total measured calls when ``stabilize=True``.
        Defaults to ``20 * n_steps``.
    stability_rtol:
        Relative tolerance for the stabilization check above.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    with _thread_limiter(single_threaded):
        for _ in range(warmup):
            step_fn()

        samples: list = []
        _run_batch(step_fn, n_steps, samples)

        if stabilize:
            cap = max_steps if max_steps is not None else n_steps * 20
            prev_p50 = _percentile(samples, 50.0)
            while len(samples) < cap:
                _run_batch(step_fn, n_steps, samples)
                new_p50 = _percentile(samples, 50.0)
                stable = (
                    new_p50 == prev_p50 == 0
                    or (prev_p50 != 0 and abs(new_p50 - prev_p50) / prev_p50 <= stability_rtol)
                )
                prev_p50 = new_p50
                if stable:
                    break

    p50 = _percentile(samples, 50.0)
    p95 = _percentile(samples, 95.0)
    total_s = sum(samples) / 1e9
    throughput = (len(samples) / total_s) if total_s > 0 else float("inf")

    return LatencyProfile(
        warmup_steps=warmup,
        n_steps=len(samples),
        p50_ns=p50,
        p95_ns=p95,
        mean_ns=statistics.fmean(samples),
        min_ns=float(min(samples)),
        max_ns=float(max(samples)),
        std_ns=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        throughput_samples_per_s=throughput,
        raw_ns=tuple(samples) if keep_raw else None,
    )
