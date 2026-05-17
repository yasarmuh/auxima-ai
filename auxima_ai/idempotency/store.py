"""Idempotency store — keys + body fingerprints + cached responses.

Protocol (the contract every write endpoint follows):

    1. Client sends ``Idempotency-Key: <opaque-string>`` plus the request body.
    2. Endpoint computes ``fingerprint = sha256(canonical(body))``.
    3. Endpoint calls :meth:`IdempotencyStore.try_begin(key, fingerprint, ttl)`.
       The result is one of:
         - :class:`BeginAccepted` — first-time submission; process the
           request and call :meth:`complete` when done.
         - :class:`BeginReplay`   — same key + same body seen before AND
           completed; return the cached response verbatim.
         - :class:`BeginInFlight` — same key seen before but not yet
           completed (another worker is processing this exact request);
           return HTTP 409 + Retry-After.
         - :class:`BeginConflict` — same key seen before with a
           **different** body fingerprint; return HTTP 422 (the client
           is reusing a key for a different operation, which is a bug).
    4. After successful processing, the endpoint calls
       :meth:`IdempotencyStore.complete(key, response)` to cache the
       response for subsequent replays.

Implementation notes:
  - Keys are scoped per tenant — the wire key + the tenant id form the
    real lookup key. Cross-tenant collision is impossible by construction.
  - The in-memory store is thread-safe via a :class:`threading.Lock`;
    suitable for a single-process FastAPI deployment with uvicorn
    workers=1 OR test environments. For multi-worker / multi-replica
    prod, plug in the Redis-backed implementation (which will satisfy
    the same Protocol).
  - TTL enforcement is lazy — we evict on every access rather than
    running a background thread, so the store has no thread/cron
    surface area to manage. The trade-off is a small amount of dead
    memory between evictions; acceptable at the expected scale.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS: Final[int] = 24 * 3600  # 24h — Stripe parity.
MAX_KEY_LEN: Final[int] = 255  # enough for any UUID variant + tenant prefix.


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class IdempotencyError(ValueError):
    """Base — any invalid input to the store raises a subclass of this."""


class InvalidKeyError(IdempotencyError):
    """Raised when the idempotency key is empty / too long / not a string."""


class InvalidFingerprintError(IdempotencyError):
    """Raised when the request fingerprint is not a valid hex digest."""


@dataclass(frozen=True)
class IdempotencyKey:
    """A namespaced idempotency key — scope is per-tenant."""

    tenant_id: str
    key: str

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id:
            raise InvalidKeyError("tenant_id must be a non-empty string")
        if not isinstance(self.key, str) or not self.key:
            raise InvalidKeyError("key must be a non-empty string")
        if len(self.key) > MAX_KEY_LEN:
            raise InvalidKeyError(
                f"key length {len(self.key)} exceeds maximum {MAX_KEY_LEN}"
            )

    def composite(self) -> str:
        """The actual lookup key inside the store — tenant + sep + key."""
        return f"{self.tenant_id}::{self.key}"


@dataclass(frozen=True)
class IdempotencyRecord:
    """One row in the store."""

    fingerprint: str
    expires_at: float
    completed: bool
    response: Any | None  # populated when completed=True


# ---------------------------------------------------------------------------
# Result types — one per outcome of try_begin()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeginAccepted:
    """First-time submission — caller should process the request."""

    key: IdempotencyKey


@dataclass(frozen=True)
class BeginReplay:
    """Same key + body seen before AND completed — return cached response."""

    key: IdempotencyKey
    response: Any


@dataclass(frozen=True)
class BeginInFlight:
    """Same key seen before but not yet completed — return 409 + retry."""

    key: IdempotencyKey


@dataclass(frozen=True)
class BeginConflict:
    """Same key seen before with a DIFFERENT body — client bug; return 422."""

    key: IdempotencyKey
    seen_fingerprint: str
    new_fingerprint: str


BeginResult = BeginAccepted | BeginReplay | BeginInFlight | BeginConflict


# ---------------------------------------------------------------------------
# Fingerprinting — canonical JSON + SHA-256
# ---------------------------------------------------------------------------


def fingerprint_payload(payload: Any) -> str:
    """Deterministic fingerprint of a JSON-shaped payload.

    Uses ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` so the
    same logical payload always hashes to the same digest regardless of
    key order or insignificant whitespace. The output is a 64-char
    lowercase hex SHA-256 digest.
    """
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_fingerprint(fp: str) -> None:
    if not isinstance(fp, str) or len(fp) != 64:
        raise InvalidFingerprintError(
            f"fingerprint must be a 64-char hex digest; got len={len(fp) if isinstance(fp, str) else type(fp).__name__}"
        )
    try:
        int(fp, 16)
    except ValueError as e:
        raise InvalidFingerprintError(
            f"fingerprint must be hex; got {fp!r}"
        ) from e


# ---------------------------------------------------------------------------
# Protocol — what every store implementation must satisfy
# ---------------------------------------------------------------------------


@runtime_checkable
class IdempotencyStore(Protocol):
    """Abstract store — in-memory + Redis implementations both satisfy this."""

    def try_begin(
        self, key: IdempotencyKey, fingerprint: str, ttl_seconds: int
    ) -> BeginResult: ...

    def complete(self, key: IdempotencyKey, response: Any) -> None: ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


@dataclass
class InMemoryIdempotencyStore:
    """Thread-safe in-memory store with lazy TTL eviction.

    Suitable for: single-process FastAPI (uvicorn workers=1), tests.
    NOT suitable for: multi-replica prod (use the Redis impl instead).

    The ``clock`` is injectable so tests can drive time forward without
    sleeping.
    """

    clock: Callable[[], float] = field(default=time.time)
    _records: dict[str, IdempotencyRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- public API --------------------------------------------------------

    def try_begin(
        self,
        key: IdempotencyKey,
        fingerprint: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> BeginResult:
        """Atomically reserve the key for processing OR return the cached outcome."""
        if not isinstance(key, IdempotencyKey):
            raise InvalidKeyError(f"key must be IdempotencyKey; got {type(key).__name__}")
        _validate_fingerprint(fingerprint)
        if ttl_seconds <= 0:
            raise IdempotencyError(f"ttl_seconds must be > 0; got {ttl_seconds}")

        now = self.clock()
        composite = key.composite()

        with self._lock:
            self._evict_expired_locked(now)

            existing = self._records.get(composite)
            if existing is None:
                # First time — reserve the slot in PENDING state.
                self._records[composite] = IdempotencyRecord(
                    fingerprint=fingerprint,
                    expires_at=now + ttl_seconds,
                    completed=False,
                    response=None,
                )
                return BeginAccepted(key=key)

            # Existing record — check fingerprint match before anything else.
            if existing.fingerprint != fingerprint:
                return BeginConflict(
                    key=key,
                    seen_fingerprint=existing.fingerprint,
                    new_fingerprint=fingerprint,
                )

            if not existing.completed:
                return BeginInFlight(key=key)

            return BeginReplay(key=key, response=existing.response)

    def complete(self, key: IdempotencyKey, response: Any) -> None:
        """Mark the in-flight reservation as completed and cache the response.

        Raises :class:`InvalidKeyError` if the key was never reserved (the
        caller never received a :class:`BeginAccepted` for it) — that's a
        programming bug, not a request-time concern.
        """
        if not isinstance(key, IdempotencyKey):
            raise InvalidKeyError(f"key must be IdempotencyKey; got {type(key).__name__}")

        composite = key.composite()
        with self._lock:
            existing = self._records.get(composite)
            if existing is None:
                raise InvalidKeyError(
                    f"cannot complete unknown idempotency key {composite!r}; "
                    "did you call try_begin() first?"
                )
            if existing.completed:
                # Idempotent complete — accept identical second call but log.
                logger.debug("complete() called twice for key %s", composite)
                return
            self._records[composite] = IdempotencyRecord(
                fingerprint=existing.fingerprint,
                expires_at=existing.expires_at,
                completed=True,
                response=response,
            )

    def size(self) -> int:
        """Number of records currently stored (after eviction)."""
        with self._lock:
            self._evict_expired_locked(self.clock())
            return len(self._records)

    def clear(self) -> None:
        """Drop everything. Test-only — never call in prod."""
        with self._lock:
            self._records.clear()

    # -- internal ----------------------------------------------------------

    def _evict_expired_locked(self, now: float) -> None:
        """Caller must hold ``self._lock``. Drops any record past its TTL."""
        # Snapshot so we can mutate the dict while iterating.
        expired = [k for k, rec in self._records.items() if rec.expires_at <= now]
        for k in expired:
            del self._records[k]


__all__ = (
    "BeginAccepted",
    "BeginConflict",
    "BeginInFlight",
    "BeginReplay",
    "BeginResult",
    "DEFAULT_TTL_SECONDS",
    "IdempotencyError",
    "IdempotencyKey",
    "IdempotencyRecord",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "InvalidFingerprintError",
    "InvalidKeyError",
    "MAX_KEY_LEN",
    "fingerprint_payload",
)
