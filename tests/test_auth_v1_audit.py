"""key_id audit emission from the Auxima-v1 middleware (S-54 R7 / AC-7).

AC-7 (sidecar half): every accepted request makes its validated ``key_id``
available to the audit trail; rejected requests record ``key_id=unknown`` +
the rejection reason. The Frappe-side AI Run Log doctype + the actual DB row
are cross-repo (private ``auxima``) and remain pending — this covers the
sidecar producing the audit signal.

The middleware takes an injectable ``audit_sink`` so a test can capture the
exact (outcome, key_id, reason) tuples without parsing log lines. The default
sink routes to the S-19 structured-event emitter. An audit-sink failure must
NEVER change the request outcome (observability is best-effort).
"""
from __future__ import annotations

import base64

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


def _build_app(audit_sink=None, *, sink_raises=False):
    store = InMemoryNonceStore(clock=_clock())
    keyring = Keyring(keys={"p2026q2": _KEY_B64})
    captured: list[tuple] = []

    def default_sink(outcome, key_id, reason):
        if sink_raises:
            raise RuntimeError("audit backend down")
        captured.append((outcome, key_id, reason))

    app = FastAPI()
    app.middleware("http")(
        make_auth_v1_middleware(
            keyring, store, clock=_clock(),
            audit_sink=audit_sink or default_sink,
        )
    )

    @app.post("/v1/echo")
    async def echo(request: Request):
        await request.body()
        return {"ok": True}

    return app, captured


def _signed(body, *, ts=_FIXED_NOW, nonce=_NONCE):
    return sign_request("p2026q2", _KEY_B64, "POST", "/v1/echo", body,
                        nonce=nonce, timestamp=ts)


def test_accepted_request_audits_real_key_id() -> None:
    app, captured = _build_app()
    client = TestClient(app)
    body = b'{"x":1}'
    r = client.post("/v1/echo", content=body, headers={"Authorization": _signed(body)})
    assert r.status_code == 200, r.text
    assert ("accepted", "p2026q2", None) in captured


def test_rejected_request_audits_unknown_key_and_reason() -> None:
    app, captured = _build_app()
    client = TestClient(app)
    r = client.post("/v1/echo", content=b"{}")  # no Authorization header
    assert r.status_code == 401
    assert ("rejected", "unknown", "bad_scheme") in captured


def test_replay_audits_known_key_id_and_replay_reason() -> None:
    app, captured = _build_app()
    client = TestClient(app)
    body = b'{"x":1}'
    headers = {"Authorization": _signed(body)}
    client.post("/v1/echo", content=body, headers=headers)
    captured.clear()
    r = client.post("/v1/echo", content=body, headers=headers)
    assert r.status_code == 401
    assert ("rejected", "p2026q2", "replay") in captured


def test_audit_sink_failure_does_not_break_a_valid_request() -> None:
    app, _ = _build_app(sink_raises=True)
    client = TestClient(app)
    body = b'{"x":1}'
    r = client.post("/v1/echo", content=body, headers={"Authorization": _signed(body)})
    assert r.status_code == 200, "observability failure must not 500 a valid request"


def test_default_audit_sink_emits_structured_event_with_key_id(caplog) -> None:
    """With no sink injected, the default routes to the S-19 emitter and the
    key_id reaches a structured ``auth.v1.request.*`` event (key_id only —
    never nonce/hmac/timestamp)."""
    import json
    import logging

    store = InMemoryNonceStore(clock=_clock())
    keyring = Keyring(keys={"p2026q2": _KEY_B64})
    app = FastAPI()
    app.middleware("http")(make_auth_v1_middleware(keyring, store, clock=_clock()))

    @app.post("/v1/echo")
    async def echo(request: Request):
        await request.body()
        return {"ok": True}

    body = b'{"x":1}'
    header = _signed(body)
    _scheme, token = header.split(" ", 1)
    _kid, ts, nonce, hmac_b64 = token.split(":")

    with caplog.at_level(logging.INFO, logger="auxima_ai.events"):
        TestClient(app).post("/v1/echo", content=body, headers={"Authorization": header})

    events = [
        json.loads(rec.getMessage())
        for rec in caplog.records
        if rec.name == "auxima_ai.events"
    ]
    accepted = [e for e in events if e["event"] == "auth.v1.request.accepted"]
    assert accepted, "expected an auth.v1.request.accepted structured event"
    assert accepted[0]["fields"]["key_id"] == "p2026q2"
    # no secret-adjacent material in the structured audit event either
    blob = json.dumps(events)
    assert nonce not in blob and hmac_b64 not in blob and ts not in blob
