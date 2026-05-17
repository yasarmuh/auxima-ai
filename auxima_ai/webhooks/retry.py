"""Webhook retry policy + backoff calculator (S-34 §3.5).

The delivery layer asks two questions after each outbound POST:

  1. **Should we retry?**  — based on the HTTP status code + the
     attempt counter + the policy's ``max_attempts``.
  2. **How long should we wait first?** — exponential backoff with
     full-jitter, capped at ``max_seconds``; if the server sent a
     ``Retry-After`` header that value wins (clamped to the same cap).

This module returns *decisions*; it does not perform HTTP itself. The
delivery worker is responsible for the actual ``await asyncio.sleep`` +
re-send. Keeping the policy pure makes it trivially unit-testable and
reusable across the sync HTTP client and the async one.

Retry rules (deliberately conservative — Stripe / Slack parity):

  - 2xx                    → :class:`DeliverySuccess`
  - 408 / 425 / 429 / 5xx  → :class:`RetryScheduled` (transient)
  - any other 4xx          → :class:`GiveUpPermanent` (client error;
                              retrying will keep failing the same way)
  - network exception      → caller passes ``status_code=None``; treated
                              as transient and scheduled for retry

The full-jitter algorithm (AWS Architecture Blog, "Exponential Backoff
And Jitter") gives uniform delay in ``[0, expo]`` rather than ``expo``
itself — this spreads the retry herd across the whole interval and
materially reduces synchronised retry storms when many workers hit the
same outage at once.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Final

logger = logging.getLogger(__name__)

# Transient status codes per Stripe + RFC 7231 + RFC 6585.
_TRANSIENT_STATUS: Final[frozenset[int]] = frozenset(
    {
        408,  # Request Timeout
        425,  # Too Early
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    },
)

DEFAULT_BASE_SECONDS: Final[float] = 1.0
DEFAULT_FACTOR: Final[float] = 2.0
DEFAULT_MAX_SECONDS: Final[float] = 600.0  # 10 minutes
DEFAULT_MAX_ATTEMPTS: Final[int] = 8


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential-backoff-with-jitter policy.

    Per S-34 §3.5 defaults: 1s base, factor 2, cap 600s, 8 attempts
    (≈ 5 minutes of retries before giving up + the final attempt).
    """

    base_seconds: float = DEFAULT_BASE_SECONDS
    factor: float = DEFAULT_FACTOR
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    def __post_init__(self) -> None:
        if self.base_seconds <= 0:
            raise ValueError(f"base_seconds must be > 0; got {self.base_seconds}")
        if self.factor < 1:
            raise ValueError(f"factor must be >= 1; got {self.factor}")
        if self.max_seconds < self.base_seconds:
            raise ValueError(
                f"max_seconds ({self.max_seconds}) must be >= base_seconds ({self.base_seconds})"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1; got {self.max_attempts}")


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliverySuccess:
    """The endpoint returned 2xx — nothing more to do."""

    attempt: int


@dataclass(frozen=True)
class RetryScheduled:
    """Transient failure — caller should sleep ``delay_seconds`` then retry."""

    next_attempt: int
    delay_seconds: float
    reason: str


@dataclass(frozen=True)
class GiveUpPermanent:
    """Non-retryable failure (4xx other than 408/425/429) — DLQ the payload."""

    attempt: int
    status_code: int | None
    reason: str


@dataclass(frozen=True)
class GiveUpExhausted:
    """Transient failure but ``max_attempts`` reached — DLQ the payload."""

    attempt: int
    status_code: int | None


Decision = DeliverySuccess | RetryScheduled | GiveUpPermanent | GiveUpExhausted


# ---------------------------------------------------------------------------
# Backoff math
# ---------------------------------------------------------------------------


def compute_delay(
    attempt: int,
    policy: RetryPolicy,
    *,
    rng: Callable[[float, float], float] = random.uniform,
) -> float:
    """Full-jitter exponential backoff for the *next* attempt.

    Parameters
    ----------
    attempt
        The number of attempts ALREADY made (>= 1). Delay for the
        first retry (after attempt #1 failed) uses ``attempt=1``.
    policy
        Policy carrying base, factor, cap.
    rng
        Injectable uniform RNG ``(low, high) -> float`` so tests are
        deterministic. Defaults to :func:`random.uniform`.

    Returns
    -------
    A non-negative float in seconds. The expected value scales like
    ``min(base * factor**(attempt - 1), max_seconds) / 2`` because we
    pick uniformly in ``[0, expo]``.
    """
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1; got {attempt}")
    expo = policy.base_seconds * (policy.factor ** (attempt - 1))
    capped = min(expo, policy.max_seconds)
    return float(rng(0.0, capped))


def _parse_retry_after(raw: str | None, max_seconds: float) -> float | None:
    """Parse a ``Retry-After`` header value (delta-seconds form only).

    The HTTP-date form (RFC 7231 §7.1.3) is intentionally NOT supported
    — its date parsing is a security-adjacent surface (timezone bugs,
    DoS via expensive parsing). Our outbound webhooks should normalise
    to integer seconds on the receiver side; if a partner endpoint
    sends an HTTP date, we ignore it and fall back to the policy.

    Returns ``None`` if the header is missing, empty, malformed, or
    negative. Clamps to ``max_seconds`` on the upper side.
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        n = float(s)
    except (TypeError, ValueError):
        return None
    if n < 0:
        return None
    return min(n, max_seconds)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def evaluate(
    attempt: int,
    *,
    status_code: int | None,
    retry_after_header: str | None = None,
    policy: RetryPolicy | None = None,
    rng: Callable[[float, float], float] = random.uniform,
) -> Decision:
    """Decide what to do after one delivery attempt.

    Parameters
    ----------
    attempt
        The 1-based attempt counter that just completed (the result of
        attempt #1 is the first call; ``attempt`` is what just happened).
    status_code
        The HTTP status returned by the receiver, or ``None`` if the
        HTTP call itself raised (timeout / DNS / TLS / connection reset).
        ``None`` is always treated as transient.
    retry_after_header
        The value of the ``Retry-After`` response header, if present.
        Only the delta-seconds form is honoured (see :func:`_parse_retry_after`
        for the rationale on the HTTP-date form).
    policy
        Retry policy; defaults to :class:`RetryPolicy()` (1s base,
        factor 2, cap 600s, 8 attempts).
    rng
        Injectable uniform RNG for tests.
    """
    policy = policy or RetryPolicy()

    if attempt < 1:
        raise ValueError(f"attempt must be >= 1; got {attempt}")

    if status_code is not None and 200 <= status_code < 300:
        return DeliverySuccess(attempt=attempt)

    is_transient = status_code is None or status_code in _TRANSIENT_STATUS
    if not is_transient:
        return GiveUpPermanent(
            attempt=attempt,
            status_code=status_code,
            reason=f"non-retryable status code {status_code}",
        )

    if attempt >= policy.max_attempts:
        return GiveUpExhausted(attempt=attempt, status_code=status_code)

    # Choose delay — server-supplied Retry-After wins; otherwise jittered backoff.
    server_delay = _parse_retry_after(retry_after_header, policy.max_seconds)
    delay = (
        server_delay
        if server_delay is not None
        else compute_delay(attempt, policy, rng=rng)
    )

    reason = (
        "network error / no response"
        if status_code is None
        else f"transient status code {status_code}"
    )
    logger.debug(
        "retry scheduled: attempt=%d -> %d, delay=%.2fs, reason=%s",
        attempt, attempt + 1, delay, reason,
    )
    return RetryScheduled(
        next_attempt=attempt + 1,
        delay_seconds=delay,
        reason=reason,
    )


__all__ = (
    "DEFAULT_BASE_SECONDS",
    "DEFAULT_FACTOR",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_SECONDS",
    "Decision",
    "DeliverySuccess",
    "GiveUpExhausted",
    "GiveUpPermanent",
    "RetryPolicy",
    "RetryScheduled",
    "compute_delay",
    "evaluate",
)
