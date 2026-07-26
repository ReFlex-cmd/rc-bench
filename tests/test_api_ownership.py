"""API-001: сервисный контур принимает JMLC-спеки и проверяет владельца.

Сервисный контур заявлен в §4 описания проекта. Два дефекта делали заявление
ложным: baseline-спеки роняли создание эксперимента (``experiment_data
.reservoir.type`` на спеке без резервуара — AttributeError), а читающие
эндпоинты не фильтровались по владельцу, так что любой аутентифицированный
пользователь видел чужие запуски. ``GET /experiments/{id}`` вдобавок вовсе не
требовал токена.

Структурные тесты в этом файле не требуют БД и держат самый дешёвый инвариант:
каждый эндпоинт про эксперименты обязан знать, кто его вызвал, и ограничивать
выборку владельцем. Поведенческие тесты помечены ``integration`` и требуют
поднятых PostgreSQL и Redis (``docker compose up -d --wait db redis`` —
без ``--wait`` они стартуют раньше, чем БД примет соединения).
"""

from __future__ import annotations

import inspect

import pytest

from rc_bench import main as api


def _code_of(endpoint) -> str:
    """Исходник без комментариев: тест по тексту не должен ни срабатывать, ни
    молчать из-за прозы, которая обсуждает старое выражение."""
    lines = []
    for line in inspect.getsource(endpoint).splitlines():
        code = line.split("#", 1)[0]
        if code.strip():
            lines.append(code)
    return "\n".join(lines)


EXPERIMENT_ENDPOINTS = (
    api.create_experiment,
    api.list_experiments,
    api.compare_experiments,
    api.get_experiment,
    api.plot_experiment,
)

READING_ENDPOINTS = (
    api.list_experiments,
    api.compare_experiments,
    api.get_experiment,
    api.plot_experiment,
)


class TestOwnershipIsStructural:
    """Проверки по сигнатуре и исходнику: они не доказывают поведение, но
    ловят самый вероятный регресс — снятую зависимость или потерянный
    фильтр, — и делают это без PostgreSQL, то есть в обычном гейте."""

    @pytest.mark.parametrize(
        "endpoint", EXPERIMENT_ENDPOINTS, ids=lambda f: f.__name__
    )
    def test_every_experiment_endpoint_knows_its_caller(self, endpoint):
        parameters = inspect.signature(endpoint).parameters

        assert "current_user" in parameters, (
            f"{endpoint.__name__} does not depend on the authenticated user; "
            "an experiment endpoint without a caller cannot scope anything"
        )

    @pytest.mark.parametrize(
        "endpoint", READING_ENDPOINTS, ids=lambda f: f.__name__
    )
    def test_every_reading_endpoint_filters_by_owner(self, endpoint):
        source = _code_of(endpoint)

        assert "Experiment.owner_id == current_user.id" in source, (
            f"{endpoint.__name__} reads experiments without scoping them to "
            "their owner"
        )

    def test_a_foreign_experiment_is_reported_as_missing_not_forbidden(self):
        """403 подтвердил бы существование чужого идентификатора."""
        source = _code_of(api.get_experiment)

        assert "status_code=404" in source
        assert "status_code=403" not in source


class TestCreateAcceptsBothFamilies:
    def test_the_created_row_takes_its_type_from_the_spec_not_the_reservoir(self):
        """``model_type`` нормализует обе ветки; ``reservoir.type`` существует
        только у половины матрицы."""
        source = _code_of(api.create_experiment)

        assert "experiment_data.model_type" in source
        assert "experiment_data.reservoir.type" not in source

    @pytest.mark.parametrize(
        ("payload_key", "model"),
        [("baseline", "persistence"), ("reservoir", "esn")],
    )
    def test_the_schema_resolves_model_type_for_both_families(self, payload_key, model):
        from rc_bench.schemas import ExperimentCreate

        payload = {
            "dataset": {"name": "uci_household_power", "length": 12_000},
            payload_key: {"type": model},
            "protocol": {
                "forecasting_mode": "fixed_horizon",
                "horizon": 1,
                "n_seeds": 0 if payload_key == "baseline" else 1,
                "use_hpo": False,
                "seasonal_period": 24,
            },
            "readout": {"alpha_grid": [1.0]},
            "seed": None if payload_key == "baseline" else 42,
        }

        spec = ExperimentCreate.model_validate(payload)

        assert spec.model_type == model
        assert spec.model_family == payload_key


# ---------------------------------------------------------------------------
# Поведенческие тесты: требуют PostgreSQL и Redis.
# ---------------------------------------------------------------------------


def _esn_payload() -> dict:
    return {
        "dataset": {"name": "narma10", "length": 300},
        "reservoir": {"type": "esn", "params": {"units": 20}},
        "protocol": {"washout": 20, "n_seeds": 1, "use_hpo": False},
        "readout": {"alpha_grid": [1.0]},
        "seed": 42,
    }


def _baseline_payload() -> dict:
    return {
        "dataset": {"name": "uci_household_power", "length": 12_000},
        "baseline": {"type": "persistence"},
        "protocol": {
            "forecasting_mode": "fixed_horizon",
            "horizon": 1,
            "n_seeds": 0,
            "use_hpo": False,
            "seasonal_period": 24,
        },
        "readout": {"alpha_grid": [1.0]},
        "seed": None,
    }


async def _register_and_login(client, email: str) -> dict:
    await client.post("/register", json={"email": email, "password": "strongpassword123"})
    token = (
        await client.post(
            "/token", data={"username": email, "password": "strongpassword123"}
        )
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_baseline_spec_can_be_created_through_the_api(client):
    headers = await _register_and_login(client, "baseline_owner@example.com")

    response = await client.post("/experiments/", json=_baseline_payload(), headers=headers)

    assert response.status_code == 200, response.text
    assert response.json()["reservoir_type"] == "persistence"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_experiments_returns_only_the_callers_experiments(client):
    alice = await _register_and_login(client, "alice@example.com")
    bob = await _register_and_login(client, "bob@example.com")
    await client.post("/experiments/", json=_esn_payload(), headers=alice)

    listed = (await client.get("/experiments/", headers=bob)).json()

    assert listed == []
    assert len((await client.get("/experiments/", headers=alice)).json()) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reading_another_users_experiment_is_refused(client):
    alice = await _register_and_login(client, "alice2@example.com")
    bob = await _register_and_login(client, "bob2@example.com")
    created = (
        await client.post("/experiments/", json=_esn_payload(), headers=alice)
    ).json()

    response = await client.get(f"/experiments/{created['id']}", headers=bob)

    assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_comparing_another_users_experiment_reveals_nothing(client):
    alice = await _register_and_login(client, "alice3@example.com")
    bob = await _register_and_login(client, "bob3@example.com")
    created = (
        await client.post("/experiments/", json=_esn_payload(), headers=alice)
    ).json()

    response = await client.get(
        "/experiments/compare", params={"ids": str(created["id"])}, headers=bob
    )

    assert response.status_code == 404
