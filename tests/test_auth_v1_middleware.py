"""Integration tests for the Auxima-v1 middleware (S-54 / GAP-16).

Builds a minimal FastAPI app with the middleware wired to an injected
keyring + in-memory nonce store, then drives it with TestClient using REAL
signed requests (auth_v1.sign_request). Covers the S-54 §3.5 HTTP contract
end-to-end + AC-1 (replay) + AC-2 (skew) + AC-3 (tamper).
"""
from __future__ import annotations

import base64

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auxima_ai.auth_nonce import (
    InMemoryNonceStore,
    NonceStoreUnavailable,
)
from auxima_ai.auth_v1 import Keyring, sign_request
from auxima_ai.auth_v1_middleware import make_auth_v1_middleware

_KEY_B64 = base64.b64encode(bytes(range(32))).decode("ascii")
_FIXED_NOW = 1_747_567_200


def _clock(now: int = _FIXED_NOW):
    return lambda: float(now)


def _build_app(nonce_store=None, *, clock=None):
    store = nonce_store or InMemoryNonceStore(clock=_clock())
    keyring = Keyring(keys={"p2026q2": _KEY_B64})
    app = FastAPI()
    app.middleware("http")(
        make_auth_v1_middleware(keyring, store, clock=clock or _clock())
    )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/v1/echo")
    async def echo(request: Request):
        raw = await request.body()
        return {
            "len": len(raw),
            "key_id": getattr(request.state, "auth_key_id", None),
            "body_hex": raw.hex(),  # for byte-equality assertions (H1)
        }

    return app, store


def _signed_headers(method, path, body, *, key_id="p2026q2", secret=_KEY_B64,
                    nonce="bm9uY2UtMTIz", ts=_FIXED_NOW):
    auth = sign_request(key_id, secret, method, path, body, nonce=nonce, timestamp=ts)
    return {"Authorization": auth}


# ---------------------------------------------------------------------------
# Happy path + exemptions
# ---------------------------------------------------------------------------


def test_healthz_is_exempt_no_auth_needed() -> None:
    app, _ = _build_app()
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200


def test_signed_request_passes_and_stashes_key_id() -> None:
    app, _ = _build_app()
    client = TestClient(app)
    body = b'{"hello":"world"}'
    headers = _signed_headers("POST", "/v1/echo", body)
    r = client.post("/v1/echo", content=body, headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["key_id"] == "p2026q2"
    assert r.json()["len"] == len(body)  # body re-readable downstream


# ---------------------------------------------------------------------------
# 401 paths (S-54 §3.5)
# ---------------------------------------------------------------------------


def test_missing_authorization_header_401() -> None:
    app, _ = _build_app()
    client = TestClient(app)
    r = client.post("/v1/echo", content=b"{}")
    assert r.status_code == 401
    assert r.json()["reason"] == "bad_scheme"


def test_wrong_scheme_401() -> None:
    app, _ = _build_app()
    client = TestClient(app)
    r = client.post("/v1/echo", content=b"{}", headers={"Authorization": "Bearer x.y.z"})
    assert r.status_code == 401
    assert r.json()["reason"] == "bad_scheme"


def test_tampered_body_401_bad_hmac() -> None:
    # AC-3: sign one body, send a different one.
    app, _ = _build_app()
    client = TestClient(app)
    headers = _signed_headers("POST", "/v1/echo", b'{"x":1}')
    r = client.post("/v1/echo", content=b'{"x":2}', headers=headers)
    assert r.status_code == 401
    assert r.json()["reason"] == "bad_hmac"


def test_stale_timestamp_401() -> None:
    # AC-2: timestamp 301s in the past (skew window 300s).
    app, _ = _build_app()
    client = TestClient(app)
    headers = _signed_headers("POST", "/v1/echo", b"{}", ts=_FIXED_NOW - 301)
    r = client.post("/v1/echo", content=b"{}", headers=headers)
    assert r.status_code == 401
    assert r.json()["reason"] == "stale_timestamp"


def test_unknown_key_id_401() -> None:
    app, _ = _build_app()
    client = TestClient(app)
    headers = _signed_headers("POST", "/v1/echo", b"{}", key_id="x9999")
    r = client.post("/v1/echo", content=b"{}", headers=headers)
    assert r.status_code == 401
    assert r.json()["reason"] == "unknown_key"


def test_query_string_is_part_of_signed_path() -> None:
    # The signer signs path WITH query; a request to the same path but
    # different query must 401 (signature won't match).
    app, _ = _build_app()
    client = TestClient(app)
    headers = _signed_headers("POST", "/v1/echo?a=1", b"{}")
    # Send to ?a=2 instead — signed path mismatch.
    r = client.post("/v1/echo?a=2", content=b"{}", headers=headers)
    assert r.status_code == 401
    assert r.json()["reason"] == "bad_hmac"
    # Sanity: same query as signed → passes.
    r2 = client.post("/v1/echo?a=1", content=b"{}", headers=headers)
    assert r2.status_code == 200, r2.text


# ---------------------------------------------------------------------------
# AC-1 — replay
# ---------------------------------------------------------------------------


def test_replay_same_header_twice_401_second_time() -> None:
    app, _ = _build_app()
    client = TestClient(app)
    body = b'{"x":1}'
    headers = _signed_headers("POST", "/v1/echo", body)
    r1 = client.post("/v1/echo", content=body, headers=headers)
    assert r1.status_code == 200, r1.text
    r2 = client.post("/v1/echo", content=body, headers=headers)
    assert r2.status_code == 401
    assert r2.json()["reason"] == "replay"


# ---------------------------------------------------------------------------
# AC-6 — nonce store unavailable → 503 (fail closed)
# ---------------------------------------------------------------------------


class _UnavailableStore:
    def claim(self, key_id, nonce, ttl_seconds):
        raise NonceStoreUnavailable("redis down")


def test_nonce_store_unavailable_returns_503_retry_after() -> None:
    app, _ = _build_app(nonce_store=_UnavailableStore())
    client = TestClient(app)
    headers = _signed_headers("POST", "/v1/echo", b"{}")
    r = client.post("/v1/echo", content=b"{}", headers=headers)
    assert r.status_code == 503
    assert r.json()["reason"] == "redis_unreachable"
    assert r.headers.get("Retry-After") == "5"


# ---------------------------------------------------------------------------
# H1 — body integrity: the HMAC'd bytes == the bytes the handler reads
# ---------------------------------------------------------------------------


def test_handler_reads_exact_signed_bytes() -> None:
    # The body contains bytes that would diverge if the middleware's read and
    # the handler's read came from different streams (binary, multi-byte UTF-8,
    # embedded newlines). Assert byte-equality via the echoed hex.
    app, _ = _build_app()
    client = TestClient(app)
    body = b'{"unicode":"\xe2\x82\xac\xf0\x9f\x94\x90","nl":"a\nb"}'
    headers = _signed_headers("POST", "/v1/echo", body)
    r = client.post("/v1/echo", content=body, headers=headers)
    assert r.status_code == 200, r.text
    assert bytes.fromhex(r.json()["body_hex"]) == body


# ---------------------------------------------------------------------------
# iter-279 review — newline-injected nonce 401s (not 500)
# ---------------------------------------------------------------------------


def test_newline_injected_nonce_401_bad_format() -> None:
    app, _ = _build_app()
    client = TestClient(app)
    # Hand-craft an Authorization header with a \n in the nonce.
    bad = f"Auxima-v1 p2026q2:{_FIXED_NOW}:no\nnce:aGVsbG8="
    r = client.post("/v1/echo", content=b"{}", headers={"Authorization": bad})
    assert r.status_code == 401
    assert r.json()["reason"] == "bad_format"
