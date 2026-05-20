"""Nonce replay-protection store for the Auxima-v1 sidecar auth (S-54 §3.3 / R5).

The :mod:`auxima_ai.auth_v1` core verifies a request's HMAC + timestamp skew
*statelessly*. Replay protection is the stateful second half: an attacker who
captures a valid Authorization header can re-send it byte-for-byte within the
skew window unless the sidecar remembers which nonces it has already accepted.

This module provides that memory, following the SAME pattern as
:mod:`auxima_ai.idempotency.store`:
  - a :class:`NonceStore` Protocol (the contract),
  - an :class:`InMemoryNonceStore` (thread-safe, lazy-TTL, injectable clock)
    for single-process / test use,
  - a Redis-backed implementation deferred to prod (it will satisfy the same
    Protocol — Redis ``SET key value NX EX ttl`` is the atomic primitive,
    avoiding the TOCTOU window S-54 §3.3 warns about).

Contract (S-54 R5):
  - key space: ``(key_id, nonce)`` — the nonce is scoped to its signing key.
  - TTL: ``2 × skew_window`` (default 600 s). After TTL the nonce can be seen
    again, but by then the timestamp window in auth_v1 rejects it anyway.
  - **fail-closed**: if the backing store is unreachable, the caller MUST 503
    (NOT fall back to "no replay protection"). The Protocol surfaces this as
    :class:`NonceStoreUnavailable`; the in-memory store never raises it, but
    the middleware contract + the future Redis impl rely on it.

The single ``claim`` call is the whole API: it atomically records a
first-seen nonce (→ :class:`NonceFresh`) or detects a repeat
(→ :class:`NonceReplay`). There is no separate "check then record" — that
split would re-introduce the TOCTOU race.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Final, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_NONCE_TTL_SECONDS: Final[int] = 600  # 2 × the 300 s auth skew window.
MAX_NONCE_LEN: Final[int] = 256  # generous; a base64url(16 bytes) nonce is 22.


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NonceStoreError(ValueError):
    """Base — invalid input to the store."""


class InvalidNonceError(NonceStoreError):
    """Raised when key_id / nonce is empty, too long, or not a string."""


class NonceStoreUnavailable(RuntimeError):
    """The backing store could not be reached. The caller MUST fail closed
    (HTTP 503 + Retry-After), NOT skip replay protection (S-54 R5). The
    in-memory store never raises this; the Redis impl will."""


# ---------------------------------------------------------------------------
# Result types — one per outcome of claim()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonceFresh:
    """The (key_id, nonce) pair was not seen within the TTL — request proceeds."""

    key_id: str
    nonce: str


@dataclass(frozen=True)
class NonceReplay:
    """The (key_id, nonce) pair was already accepted within the TTL — REJECT
    (S-54 §3.5: 401 reason=replay, and PAGE — replay = active attack or a
    buggy retry)."""

    key_id: str
    nonce: str


ClaimResult = NonceFresh | NonceReplay


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class NonceStore(Protocol):
    """Abstract replay store — in-memory + Redis impls both satisfy this."""

    def claim(
        self, key_id: str, nonce: str, ttl_seconds: int
    ) -> ClaimResult:
        """Atomically record a first-seen nonce or detect a replay.

        Raises :class:`NonceStoreUnavailable` if the store is unreachable
        (caller fails closed → 503)."""
        ...


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(key_id: str, nonce: str) -> None:
    for label, value in (("key_id", key_id), ("nonce", nonce)):
        if not isinstance(value, str) or not value:
            raise InvalidNonceError(f"{label} must be a non-empty string")
        if len(value) > MAX_NONCE_LEN:
            raise InvalidNonceError(
                f"{label} length {len(value)} exceeds maximum {MAX_NONCE_LEN}"
            )


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


@dataclass
class InMemoryNonceStore:
    """Thread-safe in-memory nonce store with lazy TTL eviction.

    Mirrors :class:`auxima_ai.idempotency.store.InMemoryIdempotencyStore`.
    Suitable for single-process FastAPI (uvicorn workers=1) + tests. NOT
    suitable for multi-replica prod — a captured nonce replayed against a
    DIFFERENT replica would not be caught (each replica has its own memory).
    Use the Redis impl in prod so all replicas share one nonce namespace.

    The ``clock`` is injectable so tests drive time forward without sleeping.
    """

    clock: Callable[[], float] = field(default=time.time)
    _seen: dict[str, float] = field(default_factory=dict)  # composite -> expires_at
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _composite(key_id: str, nonce: str) -> str:
        # key_id and nonce are bounded + validated; "::" cannot occur inside a
        # base64url nonce, so the separator is unambiguous.
        return f"{key_id}::{nonce}"

    def claim(
        self,
        key_id: str,
        nonce: str,
        ttl_seconds: int = DEFAULT_NONCE_TTL_SECONDS,
    ) -> ClaimResult:
        _validate(key_id, nonce)
        if ttl_seconds <= 0:
            raise NonceStoreError(f"ttl_seconds must be > 0; got {ttl_seconds}")

        now = self.clock()
        composite = self._composite(key_id, nonce)

        with self._lock:
            self._evict_expired_locked(now)
            existing = self._seen.get(composite)
            if existing is not None and existing > now:
                # Still within TTL → this is a replay.
                return NonceReplay(key_id=key_id, nonce=nonce)
            # Fresh (never seen, or its prior TTL has lapsed) → record + accept.
            self._seen[composite] = now + ttl_seconds
            return NonceFresh(key_id=key_id, nonce=nonce)

    def size(self) -> int:
        """Number of live nonces (after eviction). Diagnostics / tests."""
        with self._lock:
            self._evict_expired_locked(self.clock())
            return len(self._seen)

    def clear(self) -> None:
        """Drop everything. Test-only."""
        with self._lock:
            self._seen.clear()

    def _evict_expired_locked(self, now: float) -> None:
        """Caller must hold ``self._lock``. Drop nonces past their TTL."""
        expired = [k for k, exp in self._seen.items() if exp <= now]
        for k in expired:
            del self._seen[k]


# ---------------------------------------------------------------------------
# Redis implementation (prod — shared nonce namespace across replicas)
# ---------------------------------------------------------------------------


#: Builtin socket-level errors caught even when redis-py is not installed.
#: redis-py's own ``ConnectionError`` / ``TimeoutError`` do NOT subclass these
#: builtins, so a real deployment SHOULD pass ``unavailable_errors`` (or use
#: :meth:`RedisNonceStore.with_redis_errors`) — but a bare socket failure on a
#: duck-typed client still fails closed by default.
_DEFAULT_UNAVAILABLE_ERRORS: Final[tuple[type[BaseException], ...]] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


@dataclass
class RedisNonceStore:
    """Redis-backed replay store — the prod impl (S-54 §3.3).

    Unlike :class:`InMemoryNonceStore`, all sidecar replicas share one nonce
    namespace, so a captured header replayed against a *different* replica is
    still caught (AC-1 cross-process).

    The single ``SET key value NX EX ttl`` is the atomic claim — it both
    records the first-seen nonce AND reports whether it already existed in one
    round-trip, closing the TOCTOU window a separate GET-then-SET would open
    (S-54 §3.3). redis-py returns ``True`` when the key was set, ``None`` when
    NX prevented it (replay).

    **No hard ``redis`` dependency:** ``client`` is duck-typed — any object
    with ``set(name, value, *, nx, ex)`` works (the real redis-py client, a
    fake, or a connection-pool wrapper). This keeps ``redis`` an optional/
    deploy-time dependency so the no-frappe isolation + license-hygiene CI is
    unaffected.

    **Fail-closed (R5):** if the client raises any of ``unavailable_errors``
    (default: builtin socket errors; pass redis-py's exception classes in
    prod via :meth:`with_redis_errors`), ``claim`` raises
    :class:`NonceStoreUnavailable` so the middleware returns 503 — never
    "skip replay protection".
    """

    client: object
    key_prefix: str = "auxima_ai:nonce"
    unavailable_errors: tuple[type[BaseException], ...] = _DEFAULT_UNAVAILABLE_ERRORS

    @classmethod
    def with_redis_errors(cls, client: object, **kwargs) -> RedisNonceStore:
        """Build a store wired to redis-py's own connection-error classes.

        Resolves ``redis.exceptions.ConnectionError`` / ``TimeoutError`` at
        call time (so ``redis`` need not be importable unless this is used).
        Falls back to the builtin socket errors if redis-py is absent.
        """
        errors: tuple[type[BaseException], ...] = _DEFAULT_UNAVAILABLE_ERRORS
        try:  # pragma: no cover - exercised only where redis-py is installed
            from redis import exceptions as _redis_exc

            errors = (_redis_exc.ConnectionError, _redis_exc.TimeoutError)
        except ImportError:
            pass
        return cls(client, unavailable_errors=errors, **kwargs)

    def _key(self, key_id: str, nonce: str) -> str:
        return f"{self.key_prefix}:{key_id}:{nonce}"

    def claim(
        self,
        key_id: str,
        nonce: str,
        ttl_seconds: int = DEFAULT_NONCE_TTL_SECONDS,
    ) -> ClaimResult:
        _validate(key_id, nonce)
        if ttl_seconds <= 0:
            raise NonceStoreError(f"ttl_seconds must be > 0; got {ttl_seconds}")

        key = self._key(key_id, nonce)
        try:
            # value is diagnostic only (S-54 §3.3); the SET return is what
            # decides fresh vs replay. NX+EX is one atomic op (no TOCTOU).
            was_set = self.client.set(key, "1", nx=True, ex=ttl_seconds)
        except self.unavailable_errors as e:
            # Fail closed (R5): we cannot verify uniqueness → 503, not 401/200.
            logger.warning("nonce store unreachable; failing closed")
            raise NonceStoreUnavailable("redis unreachable") from e

        if was_set:
            return NonceFresh(key_id=key_id, nonce=nonce)
        return NonceReplay(key_id=key_id, nonce=nonce)


__all__ = (
    "DEFAULT_NONCE_TTL_SECONDS",
    "MAX_NONCE_LEN",
    "ClaimResult",
    "InMemoryNonceStore",
    "InvalidNonceError",
    "NonceFresh",
    "NonceReplay",
    "NonceStore",
    "NonceStoreError",
    "NonceStoreUnavailable",
    "RedisNonceStore",
)
