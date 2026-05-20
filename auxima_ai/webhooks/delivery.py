"""Synchronous webhook delivery worker (S-34 §3 — composes the primitives).

Brings together every webhook primitive shipped in prior iters:

  - :mod:`auxima_ai.webhooks.url_guard`  — SSRF allow-list (iter 64)
  - :mod:`auxima_ai.webhooks.signer`     — HMAC-SHA256 v1 (iter 53)
  - :mod:`auxima_ai.webhooks.retry`      — full-jitter backoff (iter 57)
  - :mod:`auxima_ai.resilience.circuit`  — 3-state breaker (iter 63)
  - :mod:`auxima_ai.webhooks.dlq`        — dead-letter queue (iter 78)

One :meth:`WebhookDeliveryWorker.deliver` call walks the full happy /
sad path: validate URL, sign body, check breaker, POST, evaluate
result via retry policy, sleep + retry on transient failures, DLQ on
permanent / exhausted failures, and record breaker outcomes so a
sequence of failures eventually opens the circuit.

This is a SYNCHRONOUS worker — fine for v1 where one outbound delivery
runs per request. An async variant lands when concurrent delivery
matters; the primitive composition stays the same.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Final, Mapping

import httpx

from auxima_ai.resilience.circuit import (
    Admit,
    CircuitBreaker,
    RejectHalfOpenSaturated,
    RejectOpen,
)
from auxima_ai.webhooks.dlq import DLQStore, build_entry
from auxima_ai.webhooks.retry import (
    DeliverySuccess,
    GiveUpExhausted,
    GiveUpPermanent,
    RetryPolicy,
    RetryScheduled,
    evaluate,
)
from auxima_ai.webhooks.signer import sign
from auxima_ai.webhooks.url_guard import (
    URLValidationError,
    ValidatedURL,
    validate_webhook_url,
)

logger = logging.getLogger(__name__)


DEFAULT_HTTP_TIMEOUT: Final[float] = 30.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DeliveryError(ValueError):
    """Raised on invalid input to :meth:`WebhookDeliveryWorker.deliver`."""


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebhookEvent:
    """One outbound webhook to deliver."""

    webhook_id: str
    target_url: str
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.webhook_id, str) or not self.webhook_id:
            raise DeliveryError("webhook_id must be a non-empty string")
        if not isinstance(self.target_url, str) or not self.target_url:
            raise DeliveryError("target_url must be a non-empty string")
        if not isinstance(self.body, (bytes, bytearray)):
            raise DeliveryError(
                f"body must be bytes/bytearray; got {type(self.body).__name__}"
            )
        if not isinstance(self.headers, Mapping):
            raise DeliveryError(
                f"headers must be a Mapping; got {type(self.headers).__name__}"
            )


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Delivered:
    """2xx received within the retry budget."""

    status_code: int
    attempts: int


@dataclass(frozen=True)
class DeadLettered:
    """Terminal failure — entry was enqueued in the DLQ."""

    reason: str
    attempts: int
    last_status: int | None
    dlq_entry_id: str


@dataclass(frozen=True)
class CircuitOpen:
    """Breaker is OPEN — caller should retry after ``retry_after_seconds``."""

    retry_after_seconds: float


@dataclass(frozen=True)
class URLRejected:
    """URL guard refused the target — never went on the wire."""

    reason: str


DeliveryOutcome = Delivered | DeadLettered | CircuitOpen | URLRejected


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@dataclass
class WebhookDeliveryWorker:
    """Composes URL-guard / signer / retry / circuit / DLQ into one call.

    Constructed once at app startup. The breaker registry is per-host —
    one degrading partner doesn't burn down well-behaved targets.
    """

    secret: str
    dlq: DLQStore
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT
    transport: httpx.BaseTransport | None = None
    sleep: Callable[[float], None] = field(default=time.sleep)
    clock: Callable[[], float] = field(default=time.time)
    allow_private_targets: bool = False
    allowed_ports: frozenset[int] | None = None
    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    _client: httpx.Client = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.secret, str) or not self.secret.strip():
            raise DeliveryError("secret must be a non-empty string")
        if self.timeout_seconds <= 0:
            raise DeliveryError(
                f"timeout_seconds must be > 0; got {self.timeout_seconds}"
            )
        self._client = httpx.Client(
            timeout=self.timeout_seconds,
            transport=self.transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WebhookDeliveryWorker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- main entry -------------------------------------------------------

    def deliver(self, event: WebhookEvent) -> DeliveryOutcome:
        """Deliver one webhook event with the full primitive composition."""
        if not isinstance(event, WebhookEvent):
            raise DeliveryError(
                f"event must be WebhookEvent; got {type(event).__name__}"
            )

        # 1. URL guard — refuse SSRF / private / non-https targets.
        try:
            validated = validate_webhook_url(
                event.target_url,
                allow_private=self.allow_private_targets,
                allowed_ports=self.allowed_ports,
            )
        except URLValidationError as e:
            return URLRejected(reason=str(e))

        # 2. Circuit-breaker check (per host).
        breaker = self._breaker_for(validated.host)
        admission = breaker.try_call()
        if isinstance(admission, (RejectOpen, RejectHalfOpenSaturated)):
            return CircuitOpen(retry_after_seconds=admission.retry_after_seconds)
        assert isinstance(admission, Admit)

        # 3. The actual delivery loop. The retry policy decides each step.
        attempt = 0
        last_status: int | None = None
        while True:
            attempt += 1
            try:
                signed = sign(event.body, self.secret, clock=self.clock)
                headers = {
                    **dict(event.headers),
                    **signed.as_dict(),
                    "Content-Type": "application/json",
                }
                resp = self._client.post(validated.url, content=bytes(event.body), headers=headers)
                last_status = resp.status_code
            except httpx.TimeoutException:
                last_status = None
                logger.info(
                    "delivery timeout: webhook=%s host=%s attempt=%d",
                    event.webhook_id, validated.host, attempt,
                )
            except httpx.HTTPError as e:
                last_status = None
                logger.info(
                    "delivery network error: webhook=%s host=%s attempt=%d err=%s",
                    event.webhook_id, validated.host, attempt, e,
                )

            decision = evaluate(
                attempt=attempt,
                status_code=last_status,
                retry_after_header=None,  # retry-after threaded by future iter
                policy=self.retry_policy,
            )

            if isinstance(decision, DeliverySuccess):
                breaker.record_success()
                return Delivered(status_code=last_status or 0, attempts=attempt)

            if isinstance(decision, GiveUpPermanent):
                breaker.record_success()  # 4xx isn't a "server is broken" signal
                return self._dead_letter(
                    event, validated, attempts=attempt,
                    last_status=last_status, reason=decision.reason,
                )

            if isinstance(decision, GiveUpExhausted):
                breaker.record_failure()
                return self._dead_letter(
                    event, validated, attempts=attempt,
                    last_status=last_status,
                    reason=(
                        f"exhausted after {attempt} attempts "
                        f"(last_status={last_status})"
                    ),
                )

            assert isinstance(decision, RetryScheduled)
            # Transient failure mid-loop counts as a breaker failure so a
            # broken target opens the circuit after enough deliveries.
            breaker.record_failure()
            # If the breaker just flipped OPEN, surface that to the caller
            # rather than burning more attempts against a dead target.
            if breaker.state.value == "open":
                return CircuitOpen(retry_after_seconds=self.retry_policy.max_seconds)
            self.sleep(decision.delay_seconds)
            # Loop continues; attempt counter advances.

    # -- internal ---------------------------------------------------------

    def _breaker_for(self, host: str) -> CircuitBreaker:
        breaker = self._breakers.get(host)
        if breaker is None:
            breaker = CircuitBreaker(name=host)
            self._breakers[host] = breaker
        return breaker

    def _dead_letter(
        self,
        event: WebhookEvent,
        validated: ValidatedURL,
        *,
        attempts: int,
        last_status: int | None,
        reason: str,
    ) -> DeadLettered:
        entry = build_entry(
            webhook_id=event.webhook_id,
            target_url=validated.url,
            body=event.body,
            headers=dict(event.headers),
            attempts=attempts,
            last_status=last_status,
            reason=reason,
        )
        self.dlq.enqueue(entry)
        logger.warning(
            "webhook DLQ: webhook=%s host=%s attempts=%d reason=%s",
            event.webhook_id, validated.host, attempts, reason,
        )
        return DeadLettered(
            reason=reason,
            attempts=attempts,
            last_status=last_status,
            dlq_entry_id=entry.id,
        )


__all__ = (
    "CircuitOpen",
    "DeadLettered",
    "DeliveryError",
    "DeliveryOutcome",
    "Delivered",
    "URLRejected",
    "WebhookDeliveryWorker",
    "WebhookEvent",
)
