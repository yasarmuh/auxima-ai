"""Config-selected auth middleware — the dual-mode cutover (S-54 / GAP-16 R9-7).

The live app must keep working with the Phase-0 shared-secret scheme by
DEFAULT (the Frappe-side Auxima-v1 signer does not exist yet), while allowing
an opt-in switch to the Auxima-v1 middleware via config. ``sidecar_auth_mode``
defaults to ``"shared_secret"`` so nothing breaks for existing callers.
"""
from __future__ import annotations

import base64

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from auxima_ai.auth import shared_secret_middleware
from auxima_ai.auth_nonce import InMemoryNonceStore
from auxima_ai.auth_select import AuthConfigError, select_auth_middleware
from auxima_ai.auth_v1 import sign_request
from auxima_ai.config import Settings

_KEY_B64 = base64.b64encode(bytes(range(32))).decode("ascii")
_PATH = "/v1/echo"


def _app_with(middleware):
    app = FastAPI()
    app.middleware("http")(middleware)

    @app.post(_PATH)
    async def echo(request: Request):
        await request.body()
        return {"ok": True}

    return TestClient(app)


# ---------------------------------------------------------------------------
# Default = shared_secret (unchanged behavior)
# ---------------------------------------------------------------------------


def test_default_mode_is_shared_secret() -> None:
    settings = Settings()
    assert settings.sidecar_auth_mode == "shared_secret"
    assert select_auth_middleware(settings) is shared_secret_middleware


def test_unknown_mode_rejected_at_config_construction() -> None:
    with pytest.raises(ValidationError):
        Settings(sidecar_auth_mode="bogus")


# ---------------------------------------------------------------------------
# auxima_v1 mode
# ---------------------------------------------------------------------------


def test_auxima_v1_mode_accepts_signed_rejects_unsigned() -> None:
    settings = Settings(
        sidecar_auth_mode="auxima_v1",
        primary_key_id="p2026q2",
        primary_key_b64=_KEY_B64,
    )
    mw = select_auth_middleware(settings, nonce_store=InMemoryNonceStore())
    client = _app_with(mw)

    body = b'{"x":1}'
    auth = sign_request("p2026q2", _KEY_B64, "POST", _PATH, body, nonce="bm9uY2Ux")
    assert client.post(_PATH, content=body, headers={"Authorization": auth}).status_code == 200
    assert client.post(_PATH, content=body).status_code == 401  # unsigned


def test_auxima_v1_mode_loads_both_keys() -> None:
    second = base64.b64encode(bytes(range(31, -1, -1))).decode("ascii")
    settings = Settings(
        sidecar_auth_mode="auxima_v1",
        primary_key_id="p2026q2", primary_key_b64=_KEY_B64,
        secondary_key_id="s2026q3", secondary_key_b64=second,
    )
    mw = select_auth_middleware(settings, nonce_store=InMemoryNonceStore())
    client = _app_with(mw)
    body = b"{}"
    auth = sign_request("s2026q3", second, "POST", _PATH, body, nonce="bm9uY2Uy")
    assert client.post(_PATH, content=body, headers={"Authorization": auth}).status_code == 200


def test_auxima_v1_mode_without_keys_fails_fast() -> None:
    settings = Settings(sidecar_auth_mode="auxima_v1")
    with pytest.raises(AuthConfigError):
        select_auth_middleware(settings, nonce_store=InMemoryNonceStore())


def test_auxima_v1_key_value_without_key_id_fails_fast() -> None:
    settings = Settings(sidecar_auth_mode="auxima_v1", primary_key_b64=_KEY_B64)
    with pytest.raises(AuthConfigError):
        select_auth_middleware(settings, nonce_store=InMemoryNonceStore())
