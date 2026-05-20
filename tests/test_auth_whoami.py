"""Tests for GET /v1/auth/whoami (S-54 R12 / AC-8).

Builds a minimal app wiring the Auxima-v1 middleware + the whoami router,
then drives it with TestClient using REAL signed requests. AC-8: returns the
active key_id to an authenticated caller; 401 to an unauthenticated one.

Also asserts the route's own defense-in-depth guard: mounted WITHOUT the v1
middleware (so request.state.auth_key_id is never set), an unauthenticated
hit still 401s rather than 500-ing on the missing attribute.
"""
from __future__ import annotations

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auxima_ai.auth_nonce import InMemoryNonceStore
from auxima_ai.auth_v1 import Keyring, sign_request
from auxima_ai.auth_v1_middleware import make_auth_v1_middleware
from auxima_ai.auth_whoami import router as whoami_router

_KEY_B64 = base64.b64encode(bytes(range(32))).decode("ascii")
_FIXED_NOW = 1_747_567_200
_PATH = "/v1/auth/whoami"


def _clock(now: int = _FIXED_NOW):
    return lambda: float(now)


def _build_app():
    store = InMemoryNonceStore(clock=_clock())
    keyring = Keyring(keys={"p2026q2": _KEY_B64})
    app = FastAPI()
    app.middleware("http")(make_auth_v1_middleware(keyring, store, clock=_clock()))
    app.include_router(whoami_router)
    return app


def _signed_headers(method, path, body=b"", *, key_id="p2026q2",
                    secret=_KEY_B64, nonce="bm9uY2UtMTIz", ts=_FIXED_NOW):
    auth = sign_request(key_id, secret, method, path, body, nonce=nonce, timestamp=ts)
    return {"Authorization": auth}


# ---------------------------------------------------------------------------
# AC-8: authenticated → 200 + key_id
# ---------------------------------------------------------------------------


def test_whoami_returns_active_key_id_to_authenticated_caller() -> None:
    client = TestClient(_build_app())
    r = client.get(_PATH, headers=_signed_headers("GET", _PATH))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key_id"] == "p2026q2"
    assert body["scheme"] == "Auxima-v1"


def test_whoami_reflects_the_signing_key_not_a_static_value() -> None:
    """A second key in the ring → whoami echoes whichever key signed."""
    store = InMemoryNonceStore(clock=_clock())
    second_b64 = base64.b64encode(bytes(range(31, -1, -1))).decode("ascii")
    keyring = Keyring(keys={"p2026q2": _KEY_B64, "s2026q3": second_b64})
    app = FastAPI()
    app.middleware("http")(make_auth_v1_middleware(keyring, store, clock=_clock()))
    app.include_router(whoami_router)
    client = TestClient(app)
    r = client.get(
        _PATH,
        headers=_signed_headers("GET", _PATH, key_id="s2026q3", secret=second_b64),
    )
    assert r.status_code == 200, r.text
    assert r.json()["key_id"] == "s2026q3"


# ---------------------------------------------------------------------------
# AC-8: unauthenticated → 401
# ---------------------------------------------------------------------------


def test_whoami_unauthenticated_401() -> None:
    client = TestClient(_build_app())
    r = client.get(_PATH)
    assert r.status_code == 401
    # rejected by the middleware before the route runs
    assert r.json()["reason"] == "bad_scheme"


def test_whoami_bad_hmac_401() -> None:
    client = TestClient(_build_app())
    headers = _signed_headers("GET", _PATH)
    headers["Authorization"] += "tamper"
    r = client.get(_PATH, headers=headers)
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Defense in depth: route mounted WITHOUT the v1 middleware
# ---------------------------------------------------------------------------


def test_whoami_without_middleware_401_not_500() -> None:
    """No middleware → request.state.auth_key_id never set. The route must
    fail closed with 401, never 500 on a missing attribute."""
    app = FastAPI()
    app.include_router(whoami_router)
    client = TestClient(app)
    r = client.get(_PATH)
    assert r.status_code == 401
    assert r.json()["reason"] == "unauthenticated"
