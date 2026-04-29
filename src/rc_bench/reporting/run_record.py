"""Run record: ExperimentSpec + ResultSpec + reproducibility metadata."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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

    @classmethod
    def make(cls, spec: ExperimentSpec, result: ResultSpec) -> "RunRecord":
        return cls(
            spec=spec,
            result=result,
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_hash=_git_hash(),
            hostname=socket.gethostname(),
            python_version=platform.python_version(),
        )


def save_run_record(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2))


def load_run_record(path: Path) -> RunRecord:
    data = json.loads(path.read_text())
    return RunRecord.model_validate(data)
