"""Hardware/software profile capture, sanitized per DEC-008.

Captures CPU model, logical/physical core count, total RAM, OS/platform,
Python version, and key library versions (numpy, scikit-learn, reservoirpy
if importable) — the fields required by docs/agent/PROJECT_CONTRACT.md
("Ресурсный профиль").

DEC-008 requires that published artifacts never contain hostname, username,
or absolute paths. This module therefore deliberately never reads
``platform.node()``, ``socket.gethostname()``, ``getpass.getuser()``,
``sys.executable``, ``sys.path``/``sys.prefix``, or any other
filesystem-location-bearing value. Only coarse hardware/software identifiers
are collected.

Platform assumption: CPU-model and physical-core detection parse
``/proc/cpuinfo`` and are Linux-specific; on other platforms those two
fields degrade gracefully to ``None``/a generic ``platform.processor()``
value rather than raising.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class HardwareProfile:
    cpu_model: str
    logical_cores: int
    physical_cores: Optional[int]
    total_ram_bytes: Optional[int]
    os_name: str
    os_release: str
    machine: str
    python_version: str
    python_implementation: str
    library_versions: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_model": self.cpu_model,
            "logical_cores": self.logical_cores,
            "physical_cores": self.physical_cores,
            "total_ram_bytes": self.total_ram_bytes,
            "os_name": self.os_name,
            "os_release": self.os_release,
            "machine": self.machine,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "library_versions": dict(self.library_versions),
        }


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _physical_cores() -> Optional[int]:
    """Count unique (physical_id, core_id) pairs from /proc/cpuinfo.

    Linux-only; returns None (unknown) when the file is unavailable or does
    not expose the expected fields (e.g. some containers/VMs/non-Linux
    platforms).
    """
    try:
        pairs = set()
        physical_id = None
        core_id = None
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                elif not line:
                    if physical_id is not None and core_id is not None:
                        pairs.add((physical_id, core_id))
                    physical_id = None
                    core_id = None
        if physical_id is not None and core_id is not None:
            pairs.add((physical_id, core_id))
        return len(pairs) if pairs else None
    except OSError:
        return None


def _total_ram_bytes() -> Optional[int]:
    """POSIX-portable total RAM via os.sysconf (Linux and macOS)."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        n_pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size) * int(n_pages)
    except (ValueError, OSError, AttributeError):
        return None


def _library_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}

    import numpy
    versions["numpy"] = getattr(numpy, "__version__", None)

    import sklearn
    versions["scikit-learn"] = getattr(sklearn, "__version__", None)

    try:
        import reservoirpy
        versions["reservoirpy"] = getattr(reservoirpy, "__version__", None)
    except ImportError:
        versions["reservoirpy"] = None

    return versions


def get_hardware_profile() -> HardwareProfile:
    """Capture a sanitized hardware/software profile of the current machine.

    Safe to serialize and publish as-is (no hostname/username/absolute
    paths) per DEC-008.
    """
    return HardwareProfile(
        cpu_model=_cpu_model(),
        logical_cores=os.cpu_count() or 1,
        physical_cores=_physical_cores(),
        total_ram_bytes=_total_ram_bytes(),
        os_name=platform.system(),
        os_release=platform.release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        library_versions=_library_versions(),
    )
