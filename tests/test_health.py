"""Tests for GET /health.

These need no network and no API key — the endpoint is intentionally
dependency-free, so this file runs green on a fresh clone.

TestClient drives the app in-process: it never binds a port, so there is
no server to start and no port conflicts.
"""

from fastapi.testclient import TestClient

from backend import __version__
from backend.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_payload_shape() -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "ai-study-assistant"
    assert body["version"] == __version__


def test_health_response_has_no_extra_fields() -> None:
    """The response contract is exactly these three keys."""
    body = client.get("/health").json()
    assert set(body) == {"status", "service", "version"}


def test_health_rejects_post() -> None:
    """Only GET is registered; 405 proves the method routing works."""
    assert client.post("/health").status_code == 405


def test_docs_is_available() -> None:
    """Swagger UI is served."""
    assert client.get("/docs").status_code == 200
