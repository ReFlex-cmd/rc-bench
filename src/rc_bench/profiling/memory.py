"""Model/state size and isolated peak-RSS profiling (PROF-002).

Covers three measurements from docs/agent/PROJECT_CONTRACT.md
("Ресурсный профиль"):

- peak RSS in an isolated process;
- serialized model size;
- working-state size.

Peak RSS is measured as the real resident set size (kernel-tracked
high-water mark), not via ``tracemalloc`` — that only sees Python-heap
allocations and misses native/BLAS/reservoirpy buffers, which is exactly the
memory this profile cares about for edge deployment.

Platform assumption: the isolated-process measurement uses a ``fork``-start
multiprocessing context, so it requires ``os.fork()`` (Linux/macOS). It will
raise a clear ``RuntimeError`` on platforms without a "fork" start method
(e.g. Windows) rather than silently falling back to "spawn", which would
require ``build_and_run`` to be a picklable top-level function instead of an
arbitrary callable/closure.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import multiprocessing


@dataclass(frozen=True)
class PeakRSSProfile:
    """Result of :func:`measure_peak_rss_subprocess`."""

    peak_rss_bytes: int
    method: str  # "vmhwm" | "getrusage_kib" | "getrusage_bytes"
    platform: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_rss_bytes": self.peak_rss_bytes,
            "method": self.method,
            "platform": self.platform,
        }


def serialized_model_bytes(model: Any, *, protocol: int = pickle.HIGHEST_PROTOCOL) -> int:
    """Return the pickled byte size of ``model``.

    Generic by design: works for any picklable object (readout weights,
    reservoir config, a whole pipeline object, ...). Callers decide what
    "the model" means for their component.
    """
    return len(pickle.dumps(model, protocol=protocol))


def working_state_bytes(state: Any) -> int:
    """Return the byte size of a reservoir/readout working-state object.

    Uses ``state.nbytes`` when available (numpy arrays and array-likes that
    expose the same attribute, e.g. many reservoirpy internal buffers).
    Falls back to the pickled size for anything else, so arbitrary state
    containers (dicts of arrays, dataclasses, ...) still get a sensible byte
    estimate.
    """
    nbytes = getattr(state, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return len(pickle.dumps(state))


def _read_peak_rss_bytes() -> "tuple[int, str]":
    """Read this process's own peak (high-water mark) RSS, in bytes.

    Prefers ``/proc/self/status`` ``VmHWM`` (Linux, kibibytes, and the most
    direct high-water-mark accounting the kernel exposes). Falls back to
    ``resource.getrusage(RUSAGE_SELF).ru_maxrss``, whose unit differs by
    platform: kibibytes on Linux, bytes on macOS.
    """
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    kib = int(line.split()[1])
                    return kib * 1024, "vmhwm"
    except (OSError, ValueError, IndexError):
        pass

    import resource

    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru_maxrss), "getrusage_bytes"
    return int(ru_maxrss) * 1024, "getrusage_kib"


def _child_entry(build_and_run: Callable[[], Any], conn) -> None:
    try:
        build_and_run()
        peak_bytes, method = _read_peak_rss_bytes()
        conn.send(("ok", peak_bytes, method))
    except BaseException as exc:  # noqa: BLE001 - forward any child failure
        conn.send(("error", f"{type(exc).__name__}: {exc}", None))
    finally:
        conn.close()


def measure_peak_rss_subprocess(
    build_and_run: Callable[[], Any],
    *,
    timeout: Optional[float] = None,
) -> PeakRSSProfile:
    """Run ``build_and_run()`` to completion in a fresh child process and
    return that child's peak RSS in bytes.

    ``build_and_run`` should perform both model construction and the
    workload to profile (e.g. build a reservoir + run warmup + a batch of
    steps), so the measured peak reflects real end-to-end memory use rather
    than just the parent interpreter's baseline. Because the child is
    produced via ``os.fork()``, arbitrary closures/objects are supported —
    nothing needs to be picklable to reach the child.

    Raises
    ------
    RuntimeError
        If the platform has no "fork" start method, if ``build_and_run``
        raised inside the child, or if the child died without reporting a
        result (e.g. killed by the OOM killer).
    TimeoutError
        If ``timeout`` is given and exceeded.
    """
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError(
            "measure_peak_rss_subprocess requires a 'fork' multiprocessing "
            f"start method, unavailable on this platform ({sys.platform})."
        )
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child_entry, args=(build_and_run, child_conn))
    proc.start()
    child_conn.close()  # parent doesn't write to it

    try:
        if not parent_conn.poll(timeout):
            proc.terminate()
            proc.join(timeout=5)
            raise TimeoutError(
                f"build_and_run did not complete within timeout={timeout}s"
            )
        try:
            result = parent_conn.recv()
        except EOFError as exc:
            proc.join(timeout=5)
            raise RuntimeError(
                "child process died without producing a result "
                f"(exitcode={proc.exitcode})"
            ) from exc
    finally:
        parent_conn.close()

    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)

    status = result[0]
    if status == "error":
        raise RuntimeError(f"build_and_run failed in child process: {result[1]}")
    if status != "ok":
        raise RuntimeError(f"unexpected child response: {result!r}")

    _, peak_bytes, method = result
    return PeakRSSProfile(peak_rss_bytes=peak_bytes, method=method, platform=sys.platform)
