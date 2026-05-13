"""Tests for /healthz + the shared-secret middleware."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch):
    """Clear the module-level cached Settings between tests so env changes take effect."""
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", "test-secret-do-not-use-in-prod")
    # Force-reset the cached settings object so the env var picks up.
    import auxima_ai.config

    auxima_ai.config._settings = None
    yield
    auxima_ai.config._settings = None


@pytest.fixture
def client():
    from auxima_ai.main import app

    return TestClient(app)


def test_healthz_is_unauthenticated_and_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "ts" in body


def test_v1_endpoint_rejects_missing_token(client):
    r = client.get("/v1/whoami")
    assert r.status_code == 401
    assert "invalid" in r.json()["detail"].lower() or "missing" in r.json()["detail"].lower()


def test_v1_endpoint_rejects_wrong_token(client):
    r = client.get(
        "/v1/whoami", headers={"X-Auxima-Sidecar-Token": "wrong-secret"}
    )
    assert r.status_code == 401


def test_v1_endpoint_accepts_correct_token(client):
    r = client.get(
        "/v1/whoami",
        headers={"X-Auxima-Sidecar-Token": "test-secret-do-not-use-in-prod"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["shared_secret_configured"] is True
    assert body["default_model"]


def test_sidecar_without_configured_secret_fails_closed(monkeypatch, client):
    """A sidecar started with an empty shared_secret must refuse every /v1/* request."""
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", "")
    import auxima_ai.config

    auxima_ai.config._settings = None  # force re-read

    r = client.get(
        "/v1/whoami", headers={"X-Auxima-Sidecar-Token": "anything"}
    )
    # Misconfiguration → 503 (not 401 — distinguishes our problem from client's).
    assert r.status_code == 503


def test_healthz_remains_accessible_even_without_secret(monkeypatch, client):
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", "")
    import auxima_ai.config

    auxima_ai.config._settings = None
    r = client.get("/healthz")
    assert r.status_code == 200
