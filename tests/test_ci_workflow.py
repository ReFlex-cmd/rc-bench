"""Static contract checks for the repository's unit CI workflow."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def _load_unit_job() -> dict:
    workflow = yaml.load(WORKFLOW_PATH.read_text(), Loader=yaml.BaseLoader)
    return workflow["jobs"]["unit"]


def test_unit_workflow_uses_supported_actions_and_python_312() -> None:
    unit_job = _load_unit_job()
    assert "services" not in unit_job

    action_refs = {
        step["uses"]
        for step in unit_job["steps"]
        if "uses" in step
    }
    assert action_refs == {
        "actions/checkout@v6",
        "actions/setup-python@v6",
    }

    setup_python = next(
        step
        for step in unit_job["steps"]
        if step.get("uses") == "actions/setup-python@v6"
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


VERIFY_SCRIPT = Path(__file__).parents[1] / "scripts" / "verify.sh"


def test_release_gate_checks_do_not_depend_on_an_optional_tool() -> None:
    """A missing search tool must not turn a gate check into a silent pass.

    `if rg ...; then fail; fi` reports "no match" when rg is absent — and rg is
    absent on a plain Fedora/Ubuntu box and inside CI containers (on this
    machine it existed only as a shell function, invisible to the script). The
    sanitization and raw-dataset checks are the two things standing between the
    bundle and a published hostname or a 133 MB dataset in Git, so they use the
    tool every POSIX system has.
    """
    script = VERIFY_SCRIPT.read_text()

    assert re.search(r"(?m)^\s*(if\s+)?rg\b", script) is None, (
        "verify.sh invokes rg; use grep so a missing tool cannot pass the gate"
    )
    assert "grep -rEn" in script
