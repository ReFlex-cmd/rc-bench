"""Edge resource-profiling utilities for the rc-bench JMLC benchmark.

Covers backlog tasks PROF-001 (single-step inference latency) and PROF-002
(isolated peak RSS, serialized model size, working-state size), plus a
sanitized hardware/software profile capture (DEC-008).

This package is intentionally generic and has no dependency on any concrete
reservoir/readout implementation: every measurement function accepts plain
callables/objects supplied by the caller.

Energy measurement is explicitly out of scope here (DEC-007): it is
represented elsewhere in the pipeline/schema as ``{"status": "unavailable"}``
and nothing in this package tries to estimate it.
"""
from .latency import LatencyProfile, measure_latency
from .memory import (
    PeakRSSProfile,
    measure_peak_rss_subprocess,
    serialized_model_bytes,
    working_state_bytes,
)
from .hardware import HardwareProfile, get_hardware_profile

__all__ = [
    "LatencyProfile",
    "measure_latency",
    "PeakRSSProfile",
    "measure_peak_rss_subprocess",
    "serialized_model_bytes",
    "working_state_bytes",
    "HardwareProfile",
    "get_hardware_profile",
]
