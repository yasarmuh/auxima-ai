"""Log-hygiene guard for the Auxima-v1 auth path (S-54 R10 / AC-9).

AC-9: "zero hits for raw HMAC, timestamp, or nonce in production-ready log
lines at INFO level." Production runs at INFO, so EVERY record at INFO or
above (INFO accepts, WARNING rejects) must carry the ``key_id`` and a fixed
``reason`` code ONLY — never the secret-adjacent fields an attacker could
use (the HMAC defeats this preimage's signature; the nonce + timestamp let a
log-reader reconstruct or replay a request).

This is a regression guard: the current middleware already complies (it logs
``key_id`` and ``reason`` only). The test drives the accept path + every
reject path through real signed requests and asserts the property holds, so
it fails the build if a future change leaks a secret-adjacent field into the
INFO+ stream.
"""
from __future__ import annotations

import base64
import logging

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auxima_ai.auth_nonce import InMemoryNonceStore
from auxima_ai.auth_v1 import Keyring, sign_request
from auxima_ai.auth_v1_middleware import make_auth_v1_middleware

_KEY_B64 = base64.b64encode(bytes(range(32))).decode("ascii")
_FIXED_NOW = 1_747_567_200
_NONCE = "bm9uY2UtMTIz"


def _clock(now: int = _FIXED_NOW):
    return lambda: float(now)


def _build_app(store):
    keyring = Keyring(keys={"p2026q2": _KEY_B64})
    app = FastAPI()
    app.middleware("http")(make_auth_v1_middleware(keyring, store, clock=_clock()))

    @app.post("/v1/echo")
    async def echo(request: Request):
        await request.body()
        return {"ok": True}

    return app


def _sign(body, *, ts=_FIXED_NOW, nonce=_NONCE):
    return sign_request("p2026q2", _KEY_B64, "POST", "/v1/echo", body,
                        nonce=nonce, timestamp=ts)


def _secret_fields(header: str) -> dict[str, str]:
    """Pull the wire-secret-adjacent fields out of a signed header so the
    test can assert their VALUES never appear in the log stream."""
    _scheme, token = header.split(" ", 1)
    key_id, ts, nonce, hmac_b64 = token.split(":")
    return {"timestamp": ts, "nonce": nonce, "hmac": hmac_b64}


def _assert_no_secrets_at_info(caplog, secrets: dict[str, str]) -> None:
    for record in caplog.records:
        if record.levelno < logging.INFO:
            continue
        msg = record.getMessage()
        for name, value in secrets.items():
            assert value not in msg, (
                f"{name} value leaked into a level>=INFO log line: {msg!r}"
            )


def test_accept_path_logs_no_secret_fields(caplog) -> None:
    store = InMemoryNonceStore(clock=_clock())
    client = TestClient(_build_app(store))
    body = b'{"hello":"world"}'
    header = _sign(body)
    with caplog.at_level(logging.INFO):
        r = client.post("/v1/echo", content=body, headers={"Authorization": header})
    assert r.status_code == 200, r.text
    # the accept line exists and carries key_id...
    assert any("accept key_id=p2026q2" in rec.getMessage() for rec in caplog.records)
    # ...but no nonce/timestamp/hmac value anywhere at INFO+
    _assert_no_secrets_at_info(caplog, _secret_fields(header))


def test_bad_hmac_reject_logs_no_secret_fields(caplog) -> None:
    store = InMemoryNonceStore(clock=_clock())
    client = TestClient(_build_app(store))
    body = b'{"hello":"world"}'
    header = _sign(body) + "tamper"
    with caplog.at_level(logging.INFO):
        r = client.post("/v1/echo", content=body, headers={"Authorization": header})
    assert r.status_code == 401
    # guard is non-vacuous: the reject record was actually captured
    assert any("reason=bad_hmac" in rec.getMessage() for rec in caplog.records)
    _assert_no_secrets_at_info(caplog, _secret_fields(_sign(body)))


def test_replay_reject_logs_no_secret_fields(caplog) -> None:
    store = InMemoryNonceStore(clock=_clock())
    client = TestClient(_build_app(store))
    body = b'{"hello":"world"}'
    header = _sign(body)
    client.post("/v1/echo", content=body, headers={"Authorization": header})
    with caplog.at_level(logging.INFO):
        r = client.post("/v1/echo", content=body, headers={"Authorization": header})
    assert r.status_code == 401  # replay
    assert any("reason=replay" in rec.getMessage() for rec in caplog.records)
    _assert_no_secrets_at_info(caplog, _secret_fields(header))


def test_stale_timestamp_reject_logs_no_secret_fields(caplog) -> None:
    store = InMemoryNonceStore(clock=_clock())
    client = TestClient(_build_app(store))
    body = b"{}"
    stale_ts = _FIXED_NOW - 301
    header = _sign(body, ts=stale_ts)
    with caplog.at_level(logging.INFO):
        r = client.post("/v1/echo", content=body, headers={"Authorization": header})
    assert r.status_code == 401
    assert any("reason=stale_timestamp" in rec.getMessage() for rec in caplog.records)
    _assert_no_secrets_at_info(caplog, _secret_fields(header))
