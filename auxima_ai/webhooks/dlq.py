"""Dead-letter queue for failed webhook deliveries (S-34 §3.7).

When the retry policy (:mod:`auxima_ai.webhooks.retry`) emits
``GiveUpPermanent`` (non-retryable 4xx) or ``GiveUpExhausted`` (all
retries used up), the delivery worker drops the payload into this
DLQ instead of losing it. Operators replay the queue once the
receiver is fixed; nothing ever silently disappears.

The Protocol :class:`DLQStore` lets the in-memory implementation be
swapped for a Postgres / Redis / object-store backend without
touching the calling code.

In-memory implementation properties:
  - Thread-safe via :class:`threading.Lock`.
  - Bounded capacity. When the queue is full, the oldest non-replayed
    entry is evicted and an :class:`EvictionEvent` is recorded so ops
    can spot a runaway DLQ before it silently loses data.
  - Entries carry a ULID (monotonic per process) — ``list_pending``
    returns them in insertion-order, which matches chronological
    order for replay.
  - ``mark_replayed`` is idempotent (a second call is a no-op).
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, Mapping, Protocol, runtime_checkable

from auxima_ai.ids.ulid import MonotonicGenerator, is_valid

logger = logging.getLogger(__name__)


DEFAULT_DLQ_CAPACITY: Final[int] = 10_000
MAX_REASON_LEN: Final[int] = 1024


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DLQError(ValueError):
    """Base — every invalid input raises a subclass of this."""


class InvalidDLQEntryError(DLQError):
    """Raised when a DLQ entry fails validation at construction time."""


class DLQOverflowError(DLQError):
    """Raised when capacity overflows AND eviction would lose data the
    caller has marked must-not-evict (reserved for future use)."""


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DLQEntry:
    """One row in the dead-letter queue."""

    id: str  # ULID
    webhook_id: str
    target_url: str
    body: bytes
    headers: Mapping[str, str]
    attempts: int
    last_status: int | None
    reason: str
    created_at: datetime
    replayed: bool = False

    def __post_init__(self) -> None:
        if not is_valid(self.id):
            raise InvalidDLQEntryError(f"id must be a valid ULID; got {self.id!r}")
        if not isinstance(self.webhook_id, str) or not self.webhook_id:
            raise InvalidDLQEntryError("webhook_id must be a non-empty string")
        if not isinstance(self.target_url, str) or not self.target_url:
            raise InvalidDLQEntryError("target_url must be a non-empty string")
        if not isinstance(self.body, (bytes, bytearray)):
            raise InvalidDLQEntryError(
                f"body must be bytes/bytearray; got {type(self.body).__name__}"
            )
        if not isinstance(self.headers, Mapping):
            raise InvalidDLQEntryError(
                f"headers must be a Mapping; got {type(self.headers).__name__}"
            )
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise InvalidDLQEntryError(
                f"attempts must be int; got {type(self.attempts).__name__}"
            )
        if self.attempts < 1:
            raise InvalidDLQEntryError(f"attempts must be >= 1; got {self.attempts}")
        if self.last_status is not None and (
            isinstance(self.last_status, bool)
            or not isinstance(self.last_status, int)
        ):
            raise InvalidDLQEntryError(
                f"last_status must be int or None; got {type(self.last_status).__name__}"
            )
        if not isinstance(self.reason, str):
            raise InvalidDLQEntryError(
                f"reason must be str; got {type(self.reason).__name__}"
            )
        if len(self.reason) > MAX_REASON_LEN:
            raise InvalidDLQEntryError(
                f"reason length {len(self.reason)} exceeds {MAX_REASON_LEN}"
            )
        if not isinstance(self.created_at, datetime):
            raise InvalidDLQEntryError(
                f"created_at must be datetime; got {type(self.created_at).__name__}"
            )
        if (
            self.created_at.tzinfo is None
            or self.created_at.tzinfo.utcoffset(self.created_at) is None
        ):
            raise InvalidDLQEntryError(
                "created_at must be timezone-aware (UTC strongly recommended)"
            )


@dataclass(frozen=True)
class EvictionEvent:
    """Telemetry record for an entry evicted to free capacity."""

    evicted_id: str
    queue_size_at_eviction: int
    evicted_at: datetime


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DLQStore(Protocol):
    """Abstract store — in-memory + future Postgres impl both satisfy this."""

    def enqueue(self, entry: DLQEntry) -> None: ...

    def list_pending(self, *, limit: int | None = None) -> list[DLQEntry]: ...

    def mark_replayed(self, entry_id: str) -> bool: ...

    def count_pending(self) -> int: ...

    def count_total(self) -> int: ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


@dataclass
class InMemoryDLQStore:
    """Bounded, thread-safe in-memory DLQ.

    Suitable for single-process FastAPI deployments + tests. Multi-
    replica prod needs a shared backend; the Protocol above keeps the
    swap clean.
    """

    capacity: int = DEFAULT_DLQ_CAPACITY
    _entries: "OrderedDict[str, DLQEntry]" = field(default_factory=OrderedDict)
    _evictions: list[EvictionEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not isinstance(self.capacity, int) or self.capacity < 1:
            raise DLQError(f"capacity must be int >= 1; got {self.capacity!r}")

    def enqueue(self, entry: DLQEntry) -> None:
        if not isinstance(entry, DLQEntry):
            raise InvalidDLQEntryError(
                f"entry must be DLQEntry; got {type(entry).__name__}"
            )
        with self._lock:
            if entry.id in self._entries:
                # Idempotent: re-enqueueing the same id is a no-op.
                logger.debug("DLQ enqueue: id %s already present, ignoring", entry.id)
                return
            # Evict oldest non-replayed entry if at capacity.
            while len(self._entries) >= self.capacity:
                evicted_id = self._pop_oldest_locked()
                self._evictions.append(
                    EvictionEvent(
                        evicted_id=evicted_id,
                        queue_size_at_eviction=len(self._entries) + 1,
                        evicted_at=datetime.now(timezone.utc),
                    ),
                )
                logger.warning(
                    "DLQ eviction: dropped %s to make room; queue at capacity %d",
                    evicted_id, self.capacity,
                )
            self._entries[entry.id] = entry

    def list_pending(self, *, limit: int | None = None) -> list[DLQEntry]:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise DLQError(f"limit must be a positive int or None; got {limit!r}")
        with self._lock:
            out: list[DLQEntry] = []
            for e in self._entries.values():
                if e.replayed:
                    continue
                out.append(e)
                if limit is not None and len(out) >= limit:
                    break
            return out

    def mark_replayed(self, entry_id: str) -> bool:
        """Mark ``entry_id`` as replayed. Returns ``True`` iff the entry
        existed AND was not already replayed."""
        if not isinstance(entry_id, str):
            raise DLQError(f"entry_id must be str; got {type(entry_id).__name__}")
        with self._lock:
            existing = self._entries.get(entry_id)
            if existing is None:
                return False
            if existing.replayed:
                return False
            self._entries[entry_id] = DLQEntry(
                id=existing.id,
                webhook_id=existing.webhook_id,
                target_url=existing.target_url,
                body=existing.body,
                headers=existing.headers,
                attempts=existing.attempts,
                last_status=existing.last_status,
                reason=existing.reason,
                created_at=existing.created_at,
                replayed=True,
            )
            return True

    def count_pending(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if not e.replayed)

    def count_total(self) -> int:
        with self._lock:
            return len(self._entries)

    def evictions(self) -> tuple[EvictionEvent, ...]:
        """Snapshot of every eviction since process start (for telemetry)."""
        with self._lock:
            return tuple(self._evictions)

    # -- internal -----------------------------------------------------------

    def _pop_oldest_locked(self) -> str:
        """Caller must hold ``self._lock``. Pops + returns the oldest id."""
        oldest_id = next(iter(self._entries))
        del self._entries[oldest_id]
        return oldest_id


# ---------------------------------------------------------------------------
# Helper — construct an entry from the retry-policy result
# ---------------------------------------------------------------------------


_ids: Final[MonotonicGenerator] = MonotonicGenerator()


def build_entry(
    *,
    webhook_id: str,
    target_url: str,
    body: bytes | bytearray,
    headers: Mapping[str, str] | None,
    attempts: int,
    last_status: int | None,
    reason: str,
    now: datetime | None = None,
    entry_id: str | None = None,
) -> DLQEntry:
    """Convenience constructor that fills the ULID + UTC ``now`` defaults.

    Use this from the delivery worker when GiveUpPermanent /
    GiveUpExhausted comes back — it ensures every DLQ entry carries a
    monotonic ULID without the caller needing to know about the
    generator.
    """
    return DLQEntry(
        id=entry_id if entry_id is not None else _ids.generate(),
        webhook_id=webhook_id,
        target_url=target_url,
        body=bytes(body),
        headers=dict(headers) if headers is not None else {},
        attempts=attempts,
        last_status=last_status,
        reason=reason,
        created_at=now if now is not None else datetime.now(timezone.utc),
    )


__all__ = (
    "DEFAULT_DLQ_CAPACITY",
    "DLQEntry",
    "DLQError",
    "DLQOverflowError",
    "DLQStore",
    "EvictionEvent",
    "InMemoryDLQStore",
    "InvalidDLQEntryError",
    "MAX_REASON_LEN",
    "build_entry",
)
