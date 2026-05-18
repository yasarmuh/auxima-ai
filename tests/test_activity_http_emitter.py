"""Tests for ``auxima_ai.activity.http_emitter``.

Uses ``httpx.MockTransport`` so the suite has zero network dependence.

Coverage:
  - Successful 2xx posts attach the token header + serialise the row
    to the expected wire dict shape.
  - retention enum serialised to its string value.
  - ts datetime serialised to ISO.
  - Plain dict (not a Mapping subclass) for payload on the wire.
  - emit() NEVER raises on any failure mode:
      * timeout
      * connection error
      * 4xx response
      * 5xx response
      * non-ActivityRow input
  - All failures are logged at the documented level.
  - Construction validation rejects empty URL / token / non-positive timeout.
  - Trailing slash on base_url normalised away.
  - Context manager closes the client cleanly.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable

import httpx
import pytest

from auxima_ai.activity.http_emitter import (
    DEFAULT_PATH,
    HTTPActivityEmitter,
    TOKEN_HEADER,
)
from auxima_ai.activity.row import (
    ActivityRow,
    RetentionClass,
    build_activity_row,
)

UTC = timezone.utc
TS = datetime(2026, 5, 18, 8, 0, tzinfo=UTC)
TOKEN = "frappe-callback-secret-32-chars-x"


def _row(**overrides) -> ActivityRow:
    defaults = dict(
        tenant_id="tenant-acme",
        kind="intake.extract.completed",
        payload={"model_id": "ollama/qwen2.5:32b", "tokens": 412},
        retention=RetentionClass.OPERATIONAL,
        source="sidecar.intake.extract",
        idempotency_key="k-1",
        ts=TS,
    )
    defaults.update(overrides)
    return build_activity_row(**defaults)


def _emitter(handler: Callable[[httpx.Request], httpx.Response]) -> HTTPActivityEmitter:
    return HTTPActivityEmitter(
        base_url="http://localhost:8000",
        token=TOKEN,
        transport=httpx.MockTransport(handler),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_successful_emit_posts_to_expected_path_with_token() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    with _emitter(handler) as emitter:
        emitter.emit(_row())

    assert captured["url"].endswith(DEFAULT_PATH)
    assert captured["headers"].get(TOKEN_HEADER.lower()) == TOKEN
    assert captured["body"]["kind"] == "intake.extract.completed"
    assert captured["body"]["retention"] == "operational"  # enum serialised
    assert captured["body"]["ts"] == TS.isoformat()


def test_wire_shape_has_all_expected_keys() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200)

    with _emitter(handler) as emitter:
        emitter.emit(_row(customer_id="cust-123"))

    for key in (
        "id", "tenant_id", "customer_id", "kind", "payload",
        "retention", "source", "idempotency_key", "redaction_applied", "ts",
    ):
        assert key in captured, f"missing wire key {key}"
    assert captured["customer_id"] == "cust-123"


def test_payload_serialised_as_plain_dict_not_mapping_subclass() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200)

    with _emitter(handler) as emitter:
        emitter.emit(_row())

    assert isinstance(captured["payload"], dict)


# ---------------------------------------------------------------------------
# Failure modes — emit() must never raise
# ---------------------------------------------------------------------------


def test_timeout_is_swallowed_and_logged_warning(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout", request=request)

    with _emitter(handler) as emitter:
        with caplog.at_level(logging.WARNING, logger="auxima_ai.activity.http_emitter"):
            emitter.emit(_row())
    # emit() returned without raising — that's the contract.
    assert any("timed out" in r.getMessage() for r in caplog.records)


def test_connection_error_is_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _emitter(handler) as emitter:
        with caplog.at_level(logging.WARNING, logger="auxima_ai.activity.http_emitter"):
            emitter.emit(_row())
    assert any("network error" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 429])
def test_4xx_swallowed_logged_warning(
    code: int, caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, text=f"client error {code}")

    with _emitter(handler) as emitter:
        with caplog.at_level(logging.WARNING, logger="auxima_ai.activity.http_emitter"):
            emitter.emit(_row())
    assert any("rejected" in r.getMessage() and str(code) in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_5xx_swallowed_logged_error(
    code: int, caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, text=f"upstream {code}")

    with _emitter(handler) as emitter:
        with caplog.at_level(logging.ERROR, logger="auxima_ai.activity.http_emitter"):
            emitter.emit(_row())
    matching = [r for r in caplog.records if "upstream" in r.getMessage()]
    assert matching
    assert matching[0].levelno == logging.ERROR


def test_non_activity_row_input_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    """A bad caller passing the wrong type doesn't crash the pipeline."""
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(True)
        return httpx.Response(200)

    with _emitter(handler) as emitter:
        with caplog.at_level(logging.ERROR, logger="auxima_ai.activity.http_emitter"):
            emitter.emit("not-a-row")  # type: ignore[arg-type]
    assert posted == [], "should not have posted on bad input"
    assert any("non-ActivityRow" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_base_url_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        HTTPActivityEmitter(base_url=bad, token=TOKEN)


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_token_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="token"):
        HTTPActivityEmitter(base_url="http://x", token=bad)


@pytest.mark.parametrize("bad", [0, -1, -0.001])
def test_non_positive_timeout_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        HTTPActivityEmitter(base_url="http://x", token=TOKEN, timeout_seconds=bad)


def test_trailing_slash_on_base_url_stripped() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200)

    e = HTTPActivityEmitter(
        base_url="http://localhost:8000/",
        token=TOKEN,
        transport=httpx.MockTransport(handler),
    )
    try:
        e.emit(_row())
    finally:
        e.close()
    # Path must not double the slash.
    assert "//api/method" not in captured["url"]


def test_close_via_context_manager_does_not_raise() -> None:
    e = _emitter(lambda r: httpx.Response(200))
    with e:
        pass  # __exit__ closes; should not raise
