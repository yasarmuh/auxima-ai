"""Per-tenant AI cost ledger with monthly ceiling enforcement (CLAUDE §2).

Two responsibilities, atomically combined:

  1. **Record** each LLM call (provider, model, tokens, latency, cost)
     to the AI Run Log for audit / billing.
  2. **Enforce** the per-tenant monthly ceiling — refuse to record a
     call that would push the month-to-date total past the ceiling.
     Refusal happens BEFORE the underlying LLM call, so the tenant
     never burns cost beyond their ceiling.

Money is :class:`decimal.Decimal` everywhere (CLAUDE.md §6 — "Money is
``Decimal``, never ``float``"). Cost values are quantised to
:data:`COST_QUANTUM` (6 decimal places, i.e. micro-dollars) at record
time so floating-point-shaped accumulation drift can never accrue.

The contract is **check-then-spend in one atomic call** (``try_spend``)
to eliminate the TOCTOU window between "would this exceed?" and "I
recorded this entry" — a busy tenant making concurrent calls could
otherwise overshoot the ceiling by the in-flight cost.

The in-memory implementation is thread-safe via a single Lock; suitable
for single-process FastAPI deployments + tests. For multi-replica prod,
the Frappe-backed ledger satisfies the same :class:`CostLedger`
Protocol and uses a transactional UPDATE on the AI Run Log doctype.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Final, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Quantum for cost values — micro-dollars (six decimal places).
# Big enough for current Tier-A model costs; small enough to never round
# a single per-token charge to zero.
COST_QUANTUM: Final[Decimal] = Decimal("0.000001")

# Sentinel for tenants without a configured ceiling — record always
# admitted. Use Decimal("Infinity") so comparison stays in Decimal
# arithmetic without coercing to float.
UNLIMITED: Final[Decimal] = Decimal("Infinity")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CostLedgerError(ValueError):
    """Base — every invalid input raises a subclass of this."""


class InvalidLedgerEntryError(CostLedgerError):
    """Raised when a :class:`LedgerEntry` field is invalid."""


class InvalidCeilingError(CostLedgerError):
    """Raised when a per-tenant ceiling value is malformed."""


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    """One AI Run Log row.

    Fields mirror the doctype defined by CLAUDE.md §2 — provider, model,
    version, tokens, latency, cost. All fields are validated at
    construction; the ledger never sees an invalid entry.
    """

    tenant_id: str
    provider: str
    model: str
    model_version: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost: Decimal
    ts: datetime  # MUST be timezone-aware (UTC strongly recommended)

    def __post_init__(self) -> None:
        for name in ("tenant_id", "provider", "model", "model_version"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v:
                raise InvalidLedgerEntryError(
                    f"{name} must be a non-empty string; got {v!r}"
                )
        for name in ("prompt_tokens", "completion_tokens", "latency_ms"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise InvalidLedgerEntryError(
                    f"{name} must be int; got {type(v).__name__}"
                )
            if v < 0:
                raise InvalidLedgerEntryError(f"{name} must be >= 0; got {v}")
        if not isinstance(self.cost, Decimal):
            raise InvalidLedgerEntryError(
                f"cost must be Decimal (not {type(self.cost).__name__}) — "
                "money invariant per CLAUDE.md §6"
            )
        if self.cost.is_nan() or self.cost.is_infinite():
            raise InvalidLedgerEntryError(f"cost must be finite; got {self.cost}")
        if self.cost < 0:
            raise InvalidLedgerEntryError(f"cost must be >= 0; got {self.cost}")
        if not isinstance(self.ts, datetime):
            raise InvalidLedgerEntryError(
                f"ts must be datetime; got {type(self.ts).__name__}"
            )
        if self.ts.tzinfo is None or self.ts.tzinfo.utcoffset(self.ts) is None:
            raise InvalidLedgerEntryError(
                "ts must be timezone-aware (UTC strongly recommended); "
                "naive datetimes invite double-counting at month boundaries"
            )

    @property
    def quantised_cost(self) -> Decimal:
        """Cost rounded to :data:`COST_QUANTUM` precision."""
        return self.cost.quantize(COST_QUANTUM, rounding=ROUND_HALF_UP)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recorded:
    """Spend was within the ceiling; the entry was recorded."""

    entry: LedgerEntry
    period_total: Decimal


@dataclass(frozen=True)
class CeilingExceeded:
    """Spend would exceed the ceiling; the entry was NOT recorded.

    ``would_be_total`` is the period total IF the entry had been recorded —
    intentionally not the live total — so callers + dashboards can show
    the overage even though it never actually happened.
    """

    entry: LedgerEntry
    would_be_total: Decimal
    ceiling: Decimal
    current_total: Decimal


SpendDecision = Recorded | CeilingExceeded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def month_key(ts: datetime) -> str:
    """Return the UTC ``YYYY-MM`` bucket key for an aware ``datetime``.

    The key is always computed in UTC regardless of the input timezone
    so per-tenant monthly aggregates align with the audit log — KSA-
    resident tenants on UTC+3 still bucket by UTC month, matching the
    IA audit retention contract.
    """
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise CostLedgerError(
            "month_key requires a timezone-aware datetime; got naive"
        )
    utc = ts.astimezone(timezone.utc)
    return f"{utc.year:04d}-{utc.month:02d}"


def _validate_ceiling(value: Decimal) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, Decimal):
        raise InvalidCeilingError(
            f"ceiling must be Decimal; got {type(value).__name__}"
        )
    if value.is_nan():
        raise InvalidCeilingError("ceiling must not be NaN")
    if value < 0:
        raise InvalidCeilingError(f"ceiling must be >= 0; got {value}")
    return value


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CostLedger(Protocol):
    """Abstract ledger — in-memory + Frappe impls both satisfy this."""

    def set_ceiling(self, tenant_id: str, monthly_ceiling: Decimal) -> None: ...

    def try_spend(self, entry: LedgerEntry) -> SpendDecision: ...

    def period_total(self, tenant_id: str, ts: datetime) -> Decimal: ...

    def ceiling_for(self, tenant_id: str) -> Decimal: ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


@dataclass
class InMemoryCostLedger:
    """Thread-safe per-tenant cost ledger.

    Entries are kept in chronological insertion order (the lock makes
    that ordering safe under concurrent ``try_spend``). Per-(tenant,
    month) totals are maintained alongside the raw entries so the
    ceiling check is O(1) regardless of how many entries the tenant
    has recorded this month.
    """

    _entries: list[LedgerEntry] = field(default_factory=list)
    _ceilings: dict[str, Decimal] = field(default_factory=dict)
    _totals: dict[tuple[str, str], Decimal] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- ceiling -----------------------------------------------------------

    def set_ceiling(self, tenant_id: str, monthly_ceiling: Decimal) -> None:
        if not isinstance(tenant_id, str) or not tenant_id:
            raise InvalidCeilingError("tenant_id must be a non-empty string")
        validated = _validate_ceiling(monthly_ceiling)
        with self._lock:
            self._ceilings[tenant_id] = validated

    def ceiling_for(self, tenant_id: str) -> Decimal:
        with self._lock:
            return self._ceilings.get(tenant_id, UNLIMITED)

    # -- spend -------------------------------------------------------------

    def try_spend(self, entry: LedgerEntry) -> SpendDecision:
        """Atomically check the ceiling and record (or reject) the entry."""
        if not isinstance(entry, LedgerEntry):
            raise InvalidLedgerEntryError(
                f"entry must be LedgerEntry; got {type(entry).__name__}"
            )
        bucket = month_key(entry.ts)
        cost = entry.quantised_cost
        key = (entry.tenant_id, bucket)

        with self._lock:
            current = self._totals.get(key, Decimal("0"))
            ceiling = self._ceilings.get(entry.tenant_id, UNLIMITED)
            would_be = current + cost
            if would_be > ceiling:
                logger.info(
                    "cost ceiling exceeded: tenant=%s bucket=%s "
                    "current=%s + cost=%s > ceiling=%s",
                    entry.tenant_id, bucket, current, cost, ceiling,
                )
                return CeilingExceeded(
                    entry=entry,
                    would_be_total=would_be,
                    ceiling=ceiling,
                    current_total=current,
                )
            self._entries.append(entry)
            self._totals[key] = would_be
            return Recorded(entry=entry, period_total=would_be)

    # -- queries -----------------------------------------------------------

    def period_total(self, tenant_id: str, ts: datetime) -> Decimal:
        bucket = month_key(ts)
        with self._lock:
            return self._totals.get((tenant_id, bucket), Decimal("0"))

    def entries(self) -> tuple[LedgerEntry, ...]:
        """Snapshot of the recorded entries in chronological insertion order."""
        with self._lock:
            return tuple(self._entries)

    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)


__all__ = (
    "COST_QUANTUM",
    "CeilingExceeded",
    "CostLedger",
    "CostLedgerError",
    "InMemoryCostLedger",
    "InvalidCeilingError",
    "InvalidLedgerEntryError",
    "LedgerEntry",
    "Recorded",
    "SpendDecision",
    "UNLIMITED",
    "month_key",
)
