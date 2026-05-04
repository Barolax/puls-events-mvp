import pytest
import os
import sys
import httpx
from dotenv import load_dotenv
import time

load_dotenv(override=True)

API_URL = "http://localhost:8000"
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "puls-events-internal-2026")


def get_token() -> str:
    """Helper — récupère un JWT valide via la clé interne."""
    response = httpx.post(
        f"{API_URL}/internal/token",
        headers={"x-api-key": INTERNAL_API_KEY}
    )
    assert response.status_code == 200, "Impossible de récupérer un token"
    return response.json()["access_token"]


class TestAuth:
    """Tests d'authentification."""

    def test_internal_token_returns_200(self):
        """Vérifie que /internal/token répond avec un token."""
        response = httpx.post(
            f"{API_URL}/internal/token",
            headers={"x-api-key": INTERNAL_API_KEY}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_internal_token_wrong_key_returns_401(self):
        """Vérifie qu'une mauvaise clé est rejetée."""
        response = httpx.post(
            f"{API_URL}/internal/token",
            headers={"x-api-key": "mauvaise-cle"}
        )
        assert response.status_code == 401

    def test_chat_without_token_returns_401(self):
        """Vérifie que /chat sans token est rejeté."""
        response = httpx.post(
            f"{API_URL}/chat",
            json={"query": "test", "session_id": "test", "city": None, "radius_km": 50}
        )
        assert response.status_code == 401

    def test_chat_with_invalid_token_returns_401(self):
        """Vérifie qu'un token invalide est rejeté."""
        response = httpx.post(
            f"{API_URL}/chat",
            headers={"Authorization": "Bearer token_bidon"},
            json={"query": "test", "session_id": "test", "city": None, "radius_km": 50}
        )
        assert response.status_code == 401

    def test_register_new_user(self):
        """Vérifie qu'on peut enregistrer un nouvel utilisateur."""
        response = httpx.post(
            f"{API_URL}/register",
            json={"email": "test_pytest@puls-events.app"}
        )
        assert response.status_code in [200, 201, 400]  # 400 si déjà existant


class TestChat:
    """Tests de l'endpoint /chat."""

    def test_chat_returns_200(self):
        """Vérifie que /chat répond correctement."""
        time.sleep(5)
        token = get_token()
        response = httpx.post(
            f"{API_URL}/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "query": "événements à Lille",
                "session_id": "pytest_001",
                "city": "Lille",
                "radius_km": 50
            },
            timeout=120.0
        )
        assert response.status_code == 200

    def test_chat_returns_response_field(self):
        """Vérifie que la réponse contient un champ 'response'."""
        time.sleep(5)
        token = get_token()
        response = httpx.post(
            f"{API_URL}/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "query": "concerts à Paris",
                "session_id": "pytest_002",
                "city": "Paris",
                "radius_km": 50
            },
            timeout=120.0
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_chat_returns_session_id(self):
        """Vérifie que la réponse contient le session_id."""
        time.sleep(5)
        token = get_token()
        response = httpx.post(
            f"{API_URL}/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "query": "événements à Lille", 
                "session_id": "pytest_003",
                "city": "Lille",              
                "radius_km": 50
            },
            timeout=120.0
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "pytest_003"
class TestMetrics:
    """Tests du endpoint Prometheus."""

    def test_metrics_endpoint_returns_200(self):
        """Vérifie que /metrics est accessible."""
        response = httpx.get(f"{API_URL}/metrics")
        assert response.status_code == 200

    def test_metrics_contains_http_requests(self):
        """Vérifie que les métriques HTTP sont présentes."""
        response = httpx.get(f"{API_URL}/metrics")
        assert "http_requests_total" in response.text


class TestHealth:
    """Tests de santé de l'API."""

    def test_api_is_reachable(self):
        """Vérifie que l'API répond."""
        response = httpx.get(f"{API_URL}/docs")
        assert response.status_code == 200

    def test_openapi_schema_accessible(self):
        """Vérifie que le schéma OpenAPI est accessible."""
        response = httpx.get(f"{API_URL}/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data