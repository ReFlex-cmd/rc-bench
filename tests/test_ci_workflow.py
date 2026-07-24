"""Static contract checks for the repository's unit CI workflow."""

from __future__ import annotations

from pathlib import Path

import yaml


WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def _load_unit_job() -> dict:
    workflow = yaml.load(WORKFLOW_PATH.read_text(), Loader=yaml.BaseLoader)
    return workflow["jobs"]["unit"]


def test_unit_workflow_uses_python_312_and_no_services() -> None:
    unit_job = _load_unit_job()
    assert "services" not in unit_job

    setup_python = next(
        step
        for step in unit_job["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    assert setup_python["with"]["python-version"] == "3.12"


def test_unit_workflow_installs_lock_and_runs_quick_gate() -> None:
    commands = [
        step["run"]
        for step in _load_unit_job()["steps"]
        if "run" in step
    ]
    assert any(
        "poetry install --no-interaction" in command
        for command in commands
    )
    assert "bash scripts/verify.sh quick" in commands
