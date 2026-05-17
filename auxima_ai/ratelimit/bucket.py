"""Token-bucket rate limiter — per-tenant backpressure (S-19).

A token bucket has a fixed ``capacity`` and refills at ``refill_per_second``
tokens/sec. Each request asks to ``try_consume(n)``; if the bucket holds
``>= n`` tokens, the request is allowed and the level drops by ``n``;
otherwise the request is denied and the caller is told how long to wait
before the bucket will hold enough tokens.

Why token bucket and not a fixed window:
  - **Burst tolerance.** Capacity > 1 lets short bursts through without
    artificially smearing them across the window; a strict window
    would reject the 2nd request even when capacity is plenty.
  - **No edge-of-window effect.** Fixed windows let a sender push
    ``2 * limit`` traffic in 2 ms by straddling the window boundary;
    token buckets refill continuously.

Tenant scoping is the caller's job: each tenant owns its own
:class:`TokenBucket`. The optional :class:`PerTenantRateLimiter` indexes
buckets by tenant id and lazily constructs them on first access.

Pure stdlib (``threading`` + ``time``); no FastAPI / Frappe deps.
Thread-safe via per-bucket lock. Injectable clock for tests.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Final, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RateLimitError(ValueError):
    """Raised on invalid limiter configuration / inputs."""


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Allowed:
    """The request was admitted. ``remaining`` is the post-consume level."""

    remaining: float


@dataclass(frozen=True)
class Denied:
    """The bucket couldn't satisfy the request right now.

    ``retry_after_seconds`` is the minimum wait until the bucket will
    hold enough tokens, given no other consumption. Callers SHOULD
    forward this in a ``Retry-After`` header on the HTTP 429 response.
    """

    retry_after_seconds: float
    remaining: float


Decision = Allowed | Denied


# ---------------------------------------------------------------------------
# TokenBucket — single bucket
# ---------------------------------------------------------------------------


_DEFAULT_CAPACITY: Final[float] = 10.0
_DEFAULT_REFILL: Final[float] = 1.0


@dataclass
class TokenBucket:
    """Single token bucket.

    Parameters
    ----------
    capacity
        Maximum tokens the bucket holds. Equivalent to the largest
        single burst the bucket can serve from a fully-refilled state.
        Must be > 0.
    refill_per_second
        Sustained rate at which tokens are added (subject to the
        capacity ceiling). Must be > 0.
    clock
        Injectable wall-clock; defaults to :func:`time.monotonic`
        (monotonic, not :func:`time.time`, so a wall-clock jump cannot
        retroactively grant or starve tokens).
    """

    capacity: float = _DEFAULT_CAPACITY
    refill_per_second: float = _DEFAULT_REFILL
    clock: Callable[[], float] = field(default=time.monotonic)
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise RateLimitError(f"capacity must be > 0; got {self.capacity}")
        if self.refill_per_second <= 0:
            raise RateLimitError(
                f"refill_per_second must be > 0; got {self.refill_per_second}"
            )
        # Start full so a fresh bucket admits its first burst.
        self._tokens = float(self.capacity)
        self._last_refill = self.clock()

    # -- public API --------------------------------------------------------

    def try_consume(self, tokens: float = 1.0) -> Decision:
        """Atomically refill and either admit (consuming ``tokens``) or deny."""
        if tokens <= 0:
            raise RateLimitError(f"tokens must be > 0; got {tokens}")
        if tokens > self.capacity:
            # No amount of waiting will satisfy a request bigger than the
            # bucket — fail fast so the caller doesn't loop on it.
            raise RateLimitError(
                f"requested tokens ({tokens}) exceeds capacity ({self.capacity}); "
                "request can never be satisfied — raise capacity or split the call"
            )

        with self._lock:
            self._refill_locked()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return Allowed(remaining=self._tokens)
            deficit = tokens - self._tokens
            wait = deficit / self.refill_per_second
            return Denied(retry_after_seconds=wait, remaining=self._tokens)

    @property
    def level(self) -> float:
        """Current token count, after lazy refill."""
        with self._lock:
            self._refill_locked()
            return self._tokens

    # -- internal ----------------------------------------------------------

    def _refill_locked(self) -> None:
        """Caller must hold ``self._lock``."""
        now = self.clock()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            # Monotonic clock can equal the previous read inside the same
            # tick; treat as no-op rather than rewinding the bucket.
            return
        gained = elapsed * self.refill_per_second
        self._tokens = min(self.capacity, self._tokens + gained)
        self._last_refill = now


# ---------------------------------------------------------------------------
# Rate-limiter Protocol + per-tenant lazy registry
# ---------------------------------------------------------------------------


@runtime_checkable
class RateLimiter(Protocol):
    """Abstract limiter — in-memory + Redis impls both satisfy this."""

    def try_consume(self, tenant_id: str, tokens: float = 1.0) -> Decision: ...


@dataclass
class PerTenantRateLimiter:
    """Lazy-instantiating per-tenant token buckets.

    First call for a tenant constructs the bucket with the policy
    defaults; subsequent calls reuse the same bucket. Buckets are never
    evicted — for a multi-thousand-tenant deployment, swap in the Redis
    backend (future module).
    """

    capacity: float = _DEFAULT_CAPACITY
    refill_per_second: float = _DEFAULT_REFILL
    clock: Callable[[], float] = field(default=time.monotonic)
    _buckets: dict[str, TokenBucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        # Validate policy bounds once at construction so the first
        # try_consume doesn't surface them at request time.
        if self.capacity <= 0:
            raise RateLimitError(f"capacity must be > 0; got {self.capacity}")
        if self.refill_per_second <= 0:
            raise RateLimitError(
                f"refill_per_second must be > 0; got {self.refill_per_second}"
            )

    def try_consume(self, tenant_id: str, tokens: float = 1.0) -> Decision:
        if not isinstance(tenant_id, str) or not tenant_id:
            raise RateLimitError("tenant_id must be a non-empty string")
        bucket = self._get_or_create(tenant_id)
        return bucket.try_consume(tokens)

    def bucket_for(self, tenant_id: str) -> TokenBucket:
        """Direct access to a tenant's bucket — useful for tests + metrics."""
        if not isinstance(tenant_id, str) or not tenant_id:
            raise RateLimitError("tenant_id must be a non-empty string")
        return self._get_or_create(tenant_id)

    def tenant_count(self) -> int:
        with self._lock:
            return len(self._buckets)

    # -- internal ----------------------------------------------------------

    def _get_or_create(self, tenant_id: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                bucket = TokenBucket(
                    capacity=self.capacity,
                    refill_per_second=self.refill_per_second,
                    clock=self.clock,
                )
                self._buckets[tenant_id] = bucket
            return bucket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seconds_until_bucket_full(bucket: TokenBucket) -> float:
    """Seconds until the bucket reaches its capacity at the current refill rate."""
    deficit = max(0.0, bucket.capacity - bucket.level)
    if deficit == 0:
        return 0.0
    return deficit / bucket.refill_per_second


__all__ = (
    "Allowed",
    "Decision",
    "Denied",
    "PerTenantRateLimiter",
    "RateLimitError",
    "RateLimiter",
    "TokenBucket",
    "seconds_until_bucket_full",
)
