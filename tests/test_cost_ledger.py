"""Tests for ``auxima_ai.cost.ledger`` — per-tenant AI cost ledger.

Coverage per CLAUDE §2 (AI Run Log) + §6 (Money is Decimal):
  - LedgerEntry validation: empty strings, negative ints, non-Decimal cost,
    NaN/Inf cost, naive datetime all rejected.
  - month_key UTC bucketing including across-the-boundary edge cases.
  - try_spend records when under ceiling.
  - try_spend rejects when would exceed ceiling.
  - Spend on boundary (== ceiling) is recorded.
  - Cross-tenant: ceiling on A doesn't affect B.
  - Cross-month: November entries don't count against October ceiling.
  - Default (no ceiling configured) is unlimited.
  - Cost is quantised to micro-dollars; accumulation stays Decimal-exact.
  - Concurrent try_spend never overshoots the ceiling.
  - ceiling_for / period_total accessors.
  - Validation paths raise the right exception subclasses.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from auxima_ai.cost.ledger import (
    COST_QUANTUM,
    CeilingExceeded,
    CostLedgerError,
    InMemoryCostLedger,
    InvalidCeilingError,
    InvalidLedgerEntryError,
    LedgerEntry,
    Recorded,
    UNLIMITED,
    month_key,
)

UTC = timezone.utc
TS = datetime(2026, 5, 17, 12, 0, tzinfo=UTC)


def _entry(
    *,
    tenant: str = "tenant-acme",
    cost: Decimal = Decimal("0.10"),
    ts: datetime | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 200,
    latency_ms: int = 1234,
) -> LedgerEntry:
    return LedgerEntry(
        tenant_id=tenant,
        provider="ollama",
        model="qwen2.5:32b",
        model_version="2025-09-01",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost=cost,
        ts=ts or TS,
    )


# ---------------------------------------------------------------------------
# month_key
# ---------------------------------------------------------------------------


def test_month_key_basic_utc() -> None:
    assert month_key(datetime(2026, 5, 17, tzinfo=UTC)) == "2026-05"


def test_month_key_pads_month() -> None:
    assert month_key(datetime(2026, 1, 1, tzinfo=UTC)) == "2026-01"


def test_month_key_converts_to_utc_first() -> None:
    """A KSA-local (UTC+3) datetime at 01:00 on June 1 is still May 31 UTC."""
    ksa = timezone(timedelta(hours=3))
    ts = datetime(2026, 6, 1, 1, 0, tzinfo=ksa)
    assert month_key(ts) == "2026-05"


def test_month_key_rejects_naive_datetime() -> None:
    with pytest.raises(CostLedgerError, match="timezone-aware"):
        month_key(datetime(2026, 5, 17, 12, 0))


# ---------------------------------------------------------------------------
# LedgerEntry validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["tenant_id", "provider", "model", "model_version"])
def test_entry_rejects_empty_string_fields(field: str) -> None:
    kwargs = dict(
        tenant_id="t", provider="p", model="m", model_version="v",
        prompt_tokens=0, completion_tokens=0, latency_ms=0,
        cost=Decimal("0"), ts=TS,
    )
    kwargs[field] = ""
    with pytest.raises(InvalidLedgerEntryError, match=field):
        LedgerEntry(**kwargs)


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "latency_ms"])
def test_entry_rejects_negative_ints(field: str) -> None:
    with pytest.raises(InvalidLedgerEntryError, match=field):
        kwargs = dict(
            tenant_id="t", provider="p", model="m", model_version="v",
            prompt_tokens=0, completion_tokens=0, latency_ms=0,
            cost=Decimal("0"), ts=TS,
        )
        kwargs[field] = -1
        LedgerEntry(**kwargs)


def test_entry_rejects_float_cost() -> None:
    with pytest.raises(InvalidLedgerEntryError, match="Decimal"):
        _entry(cost=0.10)  # type: ignore[arg-type]


def test_entry_rejects_negative_cost() -> None:
    with pytest.raises(InvalidLedgerEntryError, match="cost"):
        _entry(cost=Decimal("-0.01"))


def test_entry_rejects_nan_cost() -> None:
    with pytest.raises(InvalidLedgerEntryError, match="finite"):
        _entry(cost=Decimal("NaN"))


def test_entry_rejects_inf_cost() -> None:
    with pytest.raises(InvalidLedgerEntryError, match="finite"):
        _entry(cost=Decimal("Infinity"))


def test_entry_rejects_naive_ts() -> None:
    with pytest.raises(InvalidLedgerEntryError, match="timezone-aware"):
        _entry(ts=datetime(2026, 5, 17, 12, 0))


def test_entry_quantises_cost_to_micro_dollars() -> None:
    e = _entry(cost=Decimal("0.0000004999"))  # would round to 0 at micro precision
    assert e.quantised_cost == Decimal("0.000000")
    e2 = _entry(cost=Decimal("0.0000005001"))  # rounds up
    assert e2.quantised_cost == Decimal("0.000001")


def test_entry_total_tokens() -> None:
    e = _entry(prompt_tokens=10, completion_tokens=15)
    assert e.total_tokens == 25


def test_entry_is_frozen() -> None:
    e = _entry()
    with pytest.raises((AttributeError, TypeError)):
        e.tenant_id = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# try_spend — happy path
# ---------------------------------------------------------------------------


def test_no_ceiling_means_unlimited_spend() -> None:
    ledger = InMemoryCostLedger()
    for i in range(5):
        r = ledger.try_spend(_entry(cost=Decimal("100")))
        assert isinstance(r, Recorded)
    assert ledger.period_total("tenant-acme", TS) == Decimal("500.000000")


def test_spend_under_ceiling_recorded() -> None:
    ledger = InMemoryCostLedger()
    ledger.set_ceiling("tenant-acme", Decimal("1.00"))
    r = ledger.try_spend(_entry(cost=Decimal("0.50")))
    assert isinstance(r, Recorded)
    assert r.period_total == Decimal("0.500000")


def test_spend_exactly_at_ceiling_recorded() -> None:
    ledger = InMemoryCostLedger()
    ledger.set_ceiling("tenant-acme", Decimal("1.00"))
    r1 = ledger.try_spend(_entry(cost=Decimal("0.60")))
    r2 = ledger.try_spend(_entry(cost=Decimal("0.40")))
    assert isinstance(r1, Recorded)
    assert isinstance(r2, Recorded)
    assert r2.period_total == Decimal("1.000000")


def test_spend_over_ceiling_rejected_with_diagnostic() -> None:
    ledger = InMemoryCostLedger()
    ledger.set_ceiling("tenant-acme", Decimal("1.00"))
    ledger.try_spend(_entry(cost=Decimal("0.90")))
    r = ledger.try_spend(_entry(cost=Decimal("0.20")))
    assert isinstance(r, CeilingExceeded)
    assert r.current_total == Decimal("0.900000")
    assert r.would_be_total == Decimal("1.100000")
    assert r.ceiling == Decimal("1.00")


def test_rejected_spend_is_not_recorded() -> None:
    ledger = InMemoryCostLedger()
    ledger.set_ceiling("tenant-acme", Decimal("0.50"))
    ledger.try_spend(_entry(cost=Decimal("0.50")))
    r = ledger.try_spend(_entry(cost=Decimal("0.01")))
    assert isinstance(r, CeilingExceeded)
    assert ledger.period_total("tenant-acme", TS) == Decimal("0.500000")
    assert ledger.entry_count() == 1


# ---------------------------------------------------------------------------
# Cross-tenant + cross-month isolation
# ---------------------------------------------------------------------------


def test_ceiling_is_per_tenant() -> None:
    ledger = InMemoryCostLedger()
    ledger.set_ceiling("tenant-a", Decimal("1.00"))
    # tenant-b has no ceiling -> unlimited
    a_blocked = ledger.try_spend(_entry(tenant="tenant-a", cost=Decimal("2.00")))
    b_ok = ledger.try_spend(_entry(tenant="tenant-b", cost=Decimal("2.00")))
    assert isinstance(a_blocked, CeilingExceeded)
    assert isinstance(b_ok, Recorded)


def test_ceiling_resets_at_month_boundary() -> None:
    ledger = InMemoryCostLedger()
    ledger.set_ceiling("tenant-acme", Decimal("1.00"))
    may = datetime(2026, 5, 31, 23, 59, tzinfo=UTC)
    june = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    ledger.try_spend(_entry(cost=Decimal("1.00"), ts=may))
    # June 1 — fresh bucket
    r = ledger.try_spend(_entry(cost=Decimal("0.99"), ts=june))
    assert isinstance(r, Recorded)


def test_period_total_returns_zero_for_unknown_tenant() -> None:
    ledger = InMemoryCostLedger()
    assert ledger.period_total("ghost", TS) == Decimal("0")


# ---------------------------------------------------------------------------
# ceiling accessors
# ---------------------------------------------------------------------------


def test_ceiling_for_unconfigured_tenant_is_unlimited() -> None:
    ledger = InMemoryCostLedger()
    assert ledger.ceiling_for("ghost") == UNLIMITED


def test_ceiling_for_after_set() -> None:
    ledger = InMemoryCostLedger()
    ledger.set_ceiling("t1", Decimal("5.00"))
    assert ledger.ceiling_for("t1") == Decimal("5.00")


def test_ceiling_can_be_zero() -> None:
    """A ceiling of 0 means "no AI spend allowed" — a useful kill switch."""
    ledger = InMemoryCostLedger()
    ledger.set_ceiling("t1", Decimal("0"))
    r = ledger.try_spend(_entry(tenant="t1", cost=Decimal("0.000001")))
    assert isinstance(r, CeilingExceeded)


# ---------------------------------------------------------------------------
# Ceiling validation
# ---------------------------------------------------------------------------


def test_set_ceiling_rejects_negative() -> None:
    ledger = InMemoryCostLedger()
    with pytest.raises(InvalidCeilingError, match=">= 0"):
        ledger.set_ceiling("t1", Decimal("-1"))


def test_set_ceiling_rejects_float() -> None:
    ledger = InMemoryCostLedger()
    with pytest.raises(InvalidCeilingError, match="Decimal"):
        ledger.set_ceiling("t1", 1.0)  # type: ignore[arg-type]


def test_set_ceiling_rejects_nan() -> None:
    ledger = InMemoryCostLedger()
    with pytest.raises(InvalidCeilingError, match="NaN"):
        ledger.set_ceiling("t1", Decimal("NaN"))


@pytest.mark.parametrize("bad", ["", None, 42])
def test_set_ceiling_rejects_bad_tenant(bad: object) -> None:
    ledger = InMemoryCostLedger()
    with pytest.raises(InvalidCeilingError, match="tenant_id"):
        ledger.set_ceiling(bad, Decimal("1"))  # type: ignore[arg-type]


def test_try_spend_rejects_non_entry() -> None:
    ledger = InMemoryCostLedger()
    with pytest.raises(InvalidLedgerEntryError, match="LedgerEntry"):
        ledger.try_spend({"cost": Decimal("0.01")})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Decimal accumulation stays exact
# ---------------------------------------------------------------------------


def test_accumulation_is_decimal_exact() -> None:
    """Three 0.1 charges sum to exactly 0.3 (floats would give 0.30000000000000004)."""
    ledger = InMemoryCostLedger()
    for _ in range(3):
        ledger.try_spend(_entry(cost=Decimal("0.1")))
    assert ledger.period_total("tenant-acme", TS) == Decimal("0.300000")


# ---------------------------------------------------------------------------
# Concurrency — ceiling never overshoots
# ---------------------------------------------------------------------------


def test_concurrent_spend_never_overshoots_ceiling() -> None:
    """20 threads each try to spend 0.1; ceiling is 1.0. Only 10 must succeed."""
    ledger = InMemoryCostLedger()
    ledger.set_ceiling("tenant-acme", Decimal("1.0"))
    results: list = []
    lock = threading.Lock()
    barrier = threading.Barrier(20)

    def worker() -> None:
        barrier.wait()
        r = ledger.try_spend(_entry(cost=Decimal("0.1")))
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    recorded = sum(1 for r in results if isinstance(r, Recorded))
    rejected = sum(1 for r in results if isinstance(r, CeilingExceeded))
    assert recorded == 10
    assert rejected == 10
    assert ledger.period_total("tenant-acme", TS) == Decimal("1.000000")
