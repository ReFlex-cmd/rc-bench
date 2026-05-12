"""Run record: ExperimentSpec + ResultSpec + reproducibility metadata.

Per ТЗ §3 / Wringe et al. 2024 criterion 6 ("полное логирование"):
- ``git_hash`` — short HEAD commit (subprocess git rev-parse).
- ``hostname``, ``python_version``, ``rc_bench_version`` — identification.
- ``lib_versions`` — versions of all major scientific dependencies.
- ``hardware_profile`` — CPU model, core counts, total RAM.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib import metadata as _md
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from rc_bench.core.schema import ExperimentSpec, ResultSpec


def _git_hash() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


# Packages whose versions we want frozen in every RunRecord. Missing packages
# are recorded as "not installed" rather than raising.
_TRACKED_PACKAGES = (
    "numpy", "scipy", "scikit-learn", "optuna",
    "reservoirpy", "pydantic", "matplotlib",
)


def _lib_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _rc_bench_version() -> str:
    try:
        return _md.version("rc-bench")
    except _md.PackageNotFoundError:
        return "0.1.0"


def _read_cpu_model() -> str:
    """Best-effort CPU model name. Linux: /proc/cpuinfo; fallback platform.processor."""
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_total_gb() -> Optional[float]:
    """Total RAM in GB. Tries psutil first, then /proc/meminfo, then None."""
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 2)
    except Exception:
        pass
    return None


def _hardware_profile() -> Dict[str, Any]:
    return {
        "cpu_model":    _read_cpu_model(),
        "cpu_count":    os.cpu_count(),
        "platform":     platform.platform(),
        "machine":      platform.machine(),
        "ram_total_gb": _ram_total_gb(),
    }


class RunRecord(BaseModel):
    spec: ExperimentSpec
    result: ResultSpec
    # All metadata fields default to "" / None for backward-compat with legacy JSON files
    # that only contain {"spec": ..., "result": ...}.
    timestamp: str = Field(default="")
    git_hash: Optional[str] = None
    hostname: str = Field(default="")
    python_version: str = Field(default="")
    rc_bench_version: str = Field(default="0.1.0")
    lib_versions: Dict[str, str] = Field(default_factory=dict)
    hardware_profile: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def make(cls, spec: ExperimentSpec, result: ResultSpec) -> "RunRecord":
        return cls(
            spec=spec,
            result=result,
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_hash=_git_hash(),
            hostname=socket.gethostname(),
            python_version=platform.python_version(),
            rc_bench_version=_rc_bench_version(),
            lib_versions=_lib_versions(),
            hardware_profile=_hardware_profile(),
        )


def save_run_record(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2))


def load_run_record(path: Path) -> RunRecord:
    data = json.loads(path.read_text())
    return RunRecord.model_validate(data)
