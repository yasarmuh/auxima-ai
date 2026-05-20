"""/v1/auth/whoami is wired into the live app (R10-1, S-54 R12 / AC-8).

Uses the create_app(settings) factory to build the app in each auth mode
WITHOUT touching global env (and without entering the TestClient context, so
the startup bootstrap does not run — matching the existing main tests).

Documented behavior:
  - shared_secret mode (default): the route is reachable but request.state
    .auth_key_id is never set (the v1 middleware isn't active), so whoami's
    own fail-closed guard returns 401 reason=unauthenticated. whoami is only
    meaningful in auxima_v1 mode.
  - auxima_v1 mode: a signed request returns 200 + the active key_id.
"""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from auxima_ai.auth_v1 import sign_request
from auxima_ai.config import Settings
from auxima_ai.main import create_app

_KEY_B64 = base64.b64encode(bytes(range(32))).decode("ascii")
_TOKEN = "test-secret-do-not-use-in-prod"
_PATH = "/v1/auth/whoami"


@pytest.fixture
def shared_secret_env(monkeypatch):
    """Configure the global shared secret — shared_secret_middleware reads the
    cached get_settings(), not create_app's injected settings, for the secret
    VALUE (it reads the value per-request, globally)."""
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", _TOKEN)
    import auxima_ai.config

    auxima_ai.config._settings = None
    yield
    auxima_ai.config._settings = None


def test_whoami_route_registered_in_shared_secret_mode(shared_secret_env) -> None:
    app = create_app(Settings(shared_secret=_TOKEN))
    client = TestClient(app)
    # authenticated to the shared-secret middleware, but no v1 key → the
    # route's own guard fails closed.
    r = client.get(_PATH, headers={"X-Auxima-Sidecar-Token": _TOKEN})
    assert r.status_code == 401
    assert r.json()["reason"] == "unauthenticated"


def test_whoami_unauthenticated_blocked_by_shared_secret_middleware(shared_secret_env) -> None:
    app = create_app(Settings(shared_secret=_TOKEN))
    client = TestClient(app)
    r = client.get(_PATH)  # no token → middleware rejects before the route
    assert r.status_code == 401


def test_whoami_returns_key_id_in_auxima_v1_mode() -> None:
    app = create_app(Settings(
        sidecar_auth_mode="auxima_v1",
        primary_key_id="p2026q2",
        primary_key_b64=_KEY_B64,
    ))
    client = TestClient(app)
    auth = sign_request("p2026q2", _KEY_B64, "GET", _PATH, b"", nonce="bm9uY2Ux")
    r = client.get(_PATH, headers={"Authorization": auth})
    assert r.status_code == 200, r.text
    assert r.json()["key_id"] == "p2026q2"


def test_whoami_unsigned_401_in_auxima_v1_mode() -> None:
    app = create_app(Settings(
        sidecar_auth_mode="auxima_v1",
        primary_key_id="p2026q2",
        primary_key_b64=_KEY_B64,
    ))
    client = TestClient(app)
    assert client.get(_PATH).status_code == 401


def test_module_level_app_still_exposed() -> None:
    # the uvicorn entrypoint `auxima_ai.main:app` must remain importable.
    from auxima_ai.main import app

    assert app is not None
    assert "/v1/auth/whoami" in {r.path for r in app.routes}
