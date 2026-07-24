"""Regression tests for the self-contained unit-test environment."""

from __future__ import annotations

import os

import pytest


@pytest.mark.parametrize(
    "setting_name",
    (
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "DB_HOST",
        "DB_PORT",
        "REDIS_URL",
        "SECRET_KEY",
    ),
)
def test_unit_mode_provides_required_application_settings(setting_name: str) -> None:
    """Unit collection must not depend on a developer's shell or local .env."""
    assert os.environ.get(setting_name), (
        f"{setting_name} must be supplied by the pytest test environment"
    )


def test_integration_marker_documents_external_services(
    pytestconfig: pytest.Config,
) -> None:
    """The opt-in integration boundary must name every required service."""
    marker = next(
        line
        for line in pytestconfig.getini("markers")
        if line.startswith("integration:")
    )
    assert "PostgreSQL" in marker
    assert "Redis" in marker
