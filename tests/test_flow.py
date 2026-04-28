import pytest
from httpx import AsyncClient

# Данные для теста
TEST_EMAIL = "pytest_user@example.com"
TEST_PASSWORD = "strongpassword123"

# Декоратор, говорящий pytest'у, что тест асинхронный
@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_flow(client: AsyncClient):
    # ---------------------------------------------------------
    # 1. РЕГИСТРАЦИЯ
    # ---------------------------------------------------------
    resp = await client.post("/register", json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    })
    # БД чистая перед каждым тестом (conftest), регистрация всегда 200
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == TEST_EMAIL
    assert "id" in data

    # ---------------------------------------------------------
    # 2. ЛОГИН (Получение токена)
    # ---------------------------------------------------------
    # OAuth2 форма отправляется как form-data, а не json
    resp = await client.post("/token", data={
        "username": TEST_EMAIL,
        "password": TEST_PASSWORD
    })
    assert resp.status_code == 200
    token_data = resp.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # ---------------------------------------------------------
    # 3. СОЗДАНИЕ ЭКСПЕРИМЕНТА (С токеном)
    # ---------------------------------------------------------
    headers = {"Authorization": f"Bearer {token}"}
    experiment_payload = {
        "reservoir_type": "esn",
        "dataset_name": "narma10",
        "config": {
            "n_units": 100,
            "seed": 777
        }
    }
    
    resp = await client.post("/experiments/", json=experiment_payload, headers=headers)
    assert resp.status_code == 200
    exp_data = resp.json()
    assert exp_data["status"] == "QUEUED"
    assert exp_data["config"]["seed"] == 777
    exp_id = exp_data["id"]

    # ---------------------------------------------------------
    # 4. ПОЛУЧЕНИЕ ЭКСПЕРИМЕНТА
    # ---------------------------------------------------------
    resp = await client.get(f"/experiments/{exp_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == exp_id