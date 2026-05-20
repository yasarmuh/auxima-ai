"""The app uses a lifespan handler, not deprecated on_event (S-19 hardening).

FastAPI deprecated ``@app.on_event``; building the app emitted a
DeprecationWarning per handler. Migrating to a single ``lifespan`` async
context manager removes the deprecation and keeps startup/shutdown behavior
identical (startup still only runs on TestClient context-enter).
"""
from __future__ import annotations

import warnings

from fastapi.testclient import TestClient

from auxima_ai.main import create_app


def test_create_app_emits_no_on_event_deprecation() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_app()
    msgs = [str(w.message) for w in caught]
    assert not any("on_event is deprecated" in m for m in msgs), msgs


def test_healthz_still_works_after_lifespan_migration() -> None:
    # behavior preserved: /healthz is unauthenticated + 200 without entering
    # the lifespan context (so bootstrap does not run).
    client = TestClient(create_app())
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
