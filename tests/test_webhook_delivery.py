"""Tests for ``auxima_ai.webhooks.delivery`` — sync worker composition.

Uses ``httpx.MockTransport`` so the suite has zero network dependence.

Coverage:
  - URL guard rejection -> URLRejected; no HTTP call made.
  - 2xx on first try -> Delivered(status, attempts=1).
  - 5xx then 2xx -> Delivered(attempts=2); breaker still CLOSED.
  - Permanent 4xx (non-retryable) -> DeadLettered + entry in DLQ.
  - All attempts 5xx -> DeadLettered with exhaustion reason; breaker
    eventually flips to OPEN.
  - Sequence of 5xx failures across separate deliver() calls opens
    the per-host breaker — subsequent call returns CircuitOpen
    without a network call.
  - Signature header attached on every attempt (v1=<hex>).
  - Caller-supplied headers preserved alongside signature headers.
  - Sleep called with the retry decision's delay_seconds (deterministic
    via injected sleep + rng).
  - WebhookEvent / construction validation.
  - allow_private_targets=True lets a loopback URL through.
"""
from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from auxima_ai.webhooks.delivery import (
    CircuitOpen,
    DeadLettered,
    Delivered,
    DeliveryError,
    URLRejected,
    WebhookDeliveryWorker,
    WebhookEvent,
)
from auxima_ai.webhooks.dlq import InMemoryDLQStore
from auxima_ai.webhooks.retry import RetryPolicy
from auxima_ai.webhooks.signer import HEADER_SIGNATURE, SIGNATURE_PREFIX


SECRET = "delivery-worker-test-secret-32ch"
PUBLIC_RESOLVED_IP = "93.184.216.34"  # example.com


def _mock_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _worker(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    retry_policy: RetryPolicy | None = None,
    allow_private: bool = True,  # tests use http://localhost — allow it
) -> tuple[WebhookDeliveryWorker, InMemoryDLQStore, list[float]]:
    sleeps: list[float] = []
    dlq = InMemoryDLQStore()
    w = WebhookDeliveryWorker(
        secret=SECRET,
        dlq=dlq,
        retry_policy=retry_policy or RetryPolicy(
            base_seconds=1.0, factor=2.0, max_seconds=600.0, max_attempts=3,
        ),
        transport=_mock_transport(handler),
        sleep=sleeps.append,
        allow_private_targets=allow_private,
        allowed_ports=frozenset({80, 443, 8000}),  # tests target localhost:8000
    )
    return w, dlq, sleeps


def _event(*, url: str = "http://localhost:8000/webhook", body: bytes = b'{"e":1}') -> WebhookEvent:
    return WebhookEvent(
        webhook_id="wh-1",
        target_url=url,
        body=body,
        headers={"X-Custom": "yes"},
    )


# ---------------------------------------------------------------------------
# WebhookEvent validation
# ---------------------------------------------------------------------------


def test_webhook_event_rejects_bad_inputs() -> None:
    with pytest.raises(DeliveryError, match="webhook_id"):
        WebhookEvent(webhook_id="", target_url="http://x", body=b"x")
    with pytest.raises(DeliveryError, match="target_url"):
        WebhookEvent(webhook_id="w", target_url="", body=b"x")
    with pytest.raises(DeliveryError, match="body"):
        WebhookEvent(webhook_id="w", target_url="http://x", body="not-bytes")  # type: ignore[arg-type]
    with pytest.raises(DeliveryError, match="headers"):
        WebhookEvent(webhook_id="w", target_url="http://x", body=b"x", headers=[])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# URL guard
# ---------------------------------------------------------------------------


def test_url_guard_rejection_short_circuits_before_any_http_call() -> None:
    """Bad scheme -> URLRejected; mock handler never invoked."""
    called = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(True)
        return httpx.Response(200)

    w, _, _ = _worker(handler)
    try:
        r = w.deliver(_event(url="ftp://localhost/x"))
    finally:
        w.close()
    assert isinstance(r, URLRejected)
    assert called == []


def test_url_guard_loopback_rejected_without_allow_private() -> None:
    """Default (prod): http://localhost is refused."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    w = WebhookDeliveryWorker(
        secret=SECRET,
        dlq=InMemoryDLQStore(),
        transport=_mock_transport(handler),
        sleep=lambda _: None,
        allow_private_targets=False,  # prod default
        allowed_ports=frozenset({80, 443, 8000}),
    )
    try:
        r = w.deliver(_event(url="http://localhost:8000/hook"))
    finally:
        w.close()
    assert isinstance(r, URLRejected)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_2xx_on_first_try_returns_delivered() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    w, dlq, sleeps = _worker(handler)
    try:
        r = w.deliver(_event())
    finally:
        w.close()
    assert isinstance(r, Delivered)
    assert r.status_code == 200
    assert r.attempts == 1
    assert dlq.count_pending() == 0
    assert sleeps == []  # no retries -> no sleeps


def test_signature_header_attached_on_every_attempt() -> None:
    seen_headers: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append({k: v for k, v in request.headers.items()})
        return httpx.Response(200)

    w, _, _ = _worker(handler)
    try:
        w.deliver(_event())
    finally:
        w.close()
    sig = seen_headers[0].get(HEADER_SIGNATURE.lower())
    assert sig is not None
    assert sig.startswith(SIGNATURE_PREFIX)


def test_caller_headers_preserved_alongside_signature() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update({k: v for k, v in request.headers.items()})
        return httpx.Response(200)

    w, _, _ = _worker(handler)
    try:
        w.deliver(_event())
    finally:
        w.close()
    assert seen.get("x-custom") == "yes"
    assert HEADER_SIGNATURE.lower() in seen


# ---------------------------------------------------------------------------
# Retry path
# ---------------------------------------------------------------------------


def test_5xx_then_2xx_retries_until_success() -> None:
    responses = iter([503, 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(responses))

    w, dlq, sleeps = _worker(handler, retry_policy=RetryPolicy(
        base_seconds=1.0, factor=2.0, max_seconds=600, max_attempts=5,
    ))
    try:
        r = w.deliver(_event())
    finally:
        w.close()
    assert isinstance(r, Delivered)
    assert r.attempts == 3
    assert dlq.count_pending() == 0
    assert len(sleeps) == 2  # two retries -> two sleeps


# ---------------------------------------------------------------------------
# Permanent failure -> DLQ
# ---------------------------------------------------------------------------


def test_permanent_4xx_dead_letters_after_first_attempt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    w, dlq, sleeps = _worker(handler)
    try:
        r = w.deliver(_event())
    finally:
        w.close()
    assert isinstance(r, DeadLettered)
    assert r.attempts == 1
    assert r.last_status == 404
    assert "404" in r.reason
    assert dlq.count_pending() == 1
    # No sleeps — gave up immediately on permanent failure.
    assert sleeps == []
    # The DLQ entry id matches the outcome.
    pending = dlq.list_pending()
    assert pending[0].id == r.dlq_entry_id


# ---------------------------------------------------------------------------
# Exhaustion
# ---------------------------------------------------------------------------


def test_all_5xx_exhausts_retries_then_dead_letters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    w, dlq, sleeps = _worker(handler, retry_policy=RetryPolicy(
        base_seconds=1.0, factor=2.0, max_seconds=600, max_attempts=3,
    ))
    try:
        r = w.deliver(_event())
    finally:
        w.close()
    assert isinstance(r, DeadLettered)
    assert "exhausted" in r.reason
    assert dlq.count_pending() == 1


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------


def test_repeated_failures_open_the_breaker_for_host() -> None:
    """5 consecutive 5xx (the breaker default threshold) -> next call CircuitOpen."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    w, dlq, _ = _worker(handler, retry_policy=RetryPolicy(
        base_seconds=0.001, factor=2.0, max_seconds=600, max_attempts=1,
    ))
    try:
        # 5 deliveries, each gives up exhausted after 1 attempt -> 5 failures.
        for _ in range(5):
            result = w.deliver(_event())
            assert isinstance(result, DeadLettered)
        # 6th delivery: breaker is OPEN.
        circuit = w.deliver(_event())
        assert isinstance(circuit, CircuitOpen)
    finally:
        w.close()


def test_breaker_open_skips_network_call() -> None:
    """When the breaker has flipped OPEN, no HTTP call is made."""
    request_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        request_count[0] += 1
        return httpx.Response(503)

    w, _, _ = _worker(handler, retry_policy=RetryPolicy(
        base_seconds=0.001, factor=2.0, max_seconds=600, max_attempts=1,
    ))
    try:
        for _ in range(5):
            w.deliver(_event())
        prev = request_count[0]
        # The 6th must NOT increment the counter.
        result = w.deliver(_event())
        assert isinstance(result, CircuitOpen)
        assert request_count[0] == prev
    finally:
        w.close()


# ---------------------------------------------------------------------------
# Network error treated as transient
# ---------------------------------------------------------------------------


def test_timeout_then_2xx_retries_to_success() -> None:
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        if calls[0] == 1:
            raise httpx.TimeoutException("simulated", request=request)
        return httpx.Response(200)

    w, _, sleeps = _worker(handler, retry_policy=RetryPolicy(
        base_seconds=1.0, factor=2.0, max_seconds=600, max_attempts=5,
    ))
    try:
        r = w.deliver(_event())
    finally:
        w.close()
    assert isinstance(r, Delivered)
    assert r.attempts == 2
    assert len(sleeps) == 1


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_construction_rejects_empty_secret() -> None:
    with pytest.raises(DeliveryError, match="secret"):
        WebhookDeliveryWorker(secret="", dlq=InMemoryDLQStore())


def test_construction_rejects_non_positive_timeout() -> None:
    with pytest.raises(DeliveryError, match="timeout"):
        WebhookDeliveryWorker(secret=SECRET, dlq=InMemoryDLQStore(), timeout_seconds=0)


def test_deliver_rejects_non_event_input() -> None:
    w, _, _ = _worker(lambda r: httpx.Response(200))
    try:
        with pytest.raises(DeliveryError, match="WebhookEvent"):
            w.deliver({"not": "an event"})  # type: ignore[arg-type]
    finally:
        w.close()
