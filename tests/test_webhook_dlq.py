"""Tests for ``auxima_ai.webhooks.dlq`` — dead-letter queue.

Coverage:
  - build_entry fills ULID + UTC timestamp defaults.
  - DLQEntry construction validates every field (bad id, wrong types,
    naive datetime, oversized reason, negative attempts, etc).
  - enqueue is idempotent on duplicate id.
  - list_pending returns insertion-order, excludes replayed entries.
  - mark_replayed returns True iff entry existed + was not already replayed.
  - count_pending excludes replayed; count_total includes them.
  - Capacity overflow evicts the OLDEST non-replayed entry + records the event.
  - Concurrent enqueue is safe under threads.
  - DLQEntry is frozen.
  - DLQStore Protocol satisfied by InMemoryDLQStore.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from auxima_ai.webhooks.dlq import (
    DEFAULT_DLQ_CAPACITY,
    DLQEntry,
    DLQError,
    DLQStore,
    EvictionEvent,
    InMemoryDLQStore,
    InvalidDLQEntryError,
    MAX_REASON_LEN,
    build_entry,
)

UTC = timezone.utc
TS = datetime(2026, 5, 18, 6, 0, tzinfo=UTC)


def _entry(**kwargs) -> DLQEntry:
    defaults = dict(
        webhook_id="wh-1",
        target_url="https://partner.example.com/hook",
        body=b'{"event":"x"}',
        headers={"X-Signature": "v1=abc"},
        attempts=3,
        last_status=500,
        reason="upstream 500s after exhaustion",
    )
    defaults.update(kwargs)
    return build_entry(**defaults)


# ---------------------------------------------------------------------------
# build_entry
# ---------------------------------------------------------------------------


def test_build_entry_fills_ulid_and_now_defaults() -> None:
    e = _entry()
    assert len(e.id) == 26  # ULID
    assert e.created_at.tzinfo is not None


def test_build_entry_accepts_explicit_ulid_and_ts() -> None:
    e = _entry(entry_id="01HXZ0M5K0RX6P0V7W3GHJK8MN", now=TS)
    assert e.id == "01HXZ0M5K0RX6P0V7W3GHJK8MN"
    assert e.created_at == TS


# ---------------------------------------------------------------------------
# DLQEntry validation
# ---------------------------------------------------------------------------


def test_entry_rejects_invalid_ulid() -> None:
    with pytest.raises(InvalidDLQEntryError, match="ULID"):
        DLQEntry(
            id="not-a-ulid",
            webhook_id="x", target_url="https://x", body=b"x",
            headers={}, attempts=1, last_status=500, reason="r",
            created_at=TS,
        )


def test_entry_rejects_naive_datetime() -> None:
    with pytest.raises(InvalidDLQEntryError, match="timezone-aware"):
        DLQEntry(
            id="01HXZ0M5K0RX6P0V7W3GHJK8MN",
            webhook_id="x", target_url="https://x", body=b"x",
            headers={}, attempts=1, last_status=500, reason="r",
            created_at=datetime(2026, 5, 18, 6, 0),  # naive
        )


def test_entry_rejects_non_bytes_body() -> None:
    with pytest.raises(InvalidDLQEntryError, match="bytes"):
        DLQEntry(
            id="01HXZ0M5K0RX6P0V7W3GHJK8MN",
            webhook_id="x", target_url="https://x",
            body="string-not-bytes",  # type: ignore[arg-type]
            headers={}, attempts=1, last_status=500, reason="r",
            created_at=TS,
        )


def test_entry_rejects_zero_attempts() -> None:
    with pytest.raises(InvalidDLQEntryError, match="attempts"):
        DLQEntry(
            id="01HXZ0M5K0RX6P0V7W3GHJK8MN",
            webhook_id="x", target_url="https://x", body=b"x",
            headers={}, attempts=0, last_status=500, reason="r",
            created_at=TS,
        )


def test_entry_rejects_oversized_reason() -> None:
    with pytest.raises(InvalidDLQEntryError, match="reason"):
        _entry(reason="x" * (MAX_REASON_LEN + 1))


def test_entry_accepts_none_last_status_for_network_failures() -> None:
    e = _entry(last_status=None)
    assert e.last_status is None


def test_entry_is_frozen() -> None:
    e = _entry()
    with pytest.raises((AttributeError, TypeError)):
        e.webhook_id = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# enqueue + list_pending
# ---------------------------------------------------------------------------


def test_enqueue_persists_entry_and_lists_it() -> None:
    store = InMemoryDLQStore()
    e = _entry()
    store.enqueue(e)
    assert store.count_pending() == 1
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0].id == e.id


def test_enqueue_duplicate_id_is_noop() -> None:
    store = InMemoryDLQStore()
    e = _entry()
    store.enqueue(e)
    store.enqueue(e)
    assert store.count_total() == 1


def test_list_pending_preserves_insertion_order() -> None:
    store = InMemoryDLQStore()
    entries = [_entry(webhook_id=f"wh-{i}") for i in range(5)]
    for e in entries:
        store.enqueue(e)
    pending_ids = [p.id for p in store.list_pending()]
    assert pending_ids == [e.id for e in entries]


def test_list_pending_honours_limit() -> None:
    store = InMemoryDLQStore()
    for i in range(10):
        store.enqueue(_entry(webhook_id=f"wh-{i}"))
    assert len(store.list_pending(limit=3)) == 3


def test_list_pending_rejects_bad_limit() -> None:
    store = InMemoryDLQStore()
    for bad in (0, -1, "1", True):
        with pytest.raises(DLQError):
            store.list_pending(limit=bad)  # type: ignore[arg-type]


def test_enqueue_validates_input_type() -> None:
    store = InMemoryDLQStore()
    with pytest.raises(InvalidDLQEntryError, match="DLQEntry"):
        store.enqueue({"not": "an entry"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# mark_replayed
# ---------------------------------------------------------------------------


def test_mark_replayed_returns_true_first_time_false_after() -> None:
    store = InMemoryDLQStore()
    e = _entry()
    store.enqueue(e)
    assert store.mark_replayed(e.id) is True
    assert store.mark_replayed(e.id) is False  # idempotent


def test_mark_replayed_unknown_id_returns_false() -> None:
    store = InMemoryDLQStore()
    assert store.mark_replayed("01HXZ0M5K0RX6P0V7W3GHJK8MN") is False


def test_mark_replayed_removes_from_pending_not_total() -> None:
    store = InMemoryDLQStore()
    e = _entry()
    store.enqueue(e)
    store.mark_replayed(e.id)
    assert store.count_pending() == 0
    assert store.count_total() == 1
    assert store.list_pending() == []


def test_mark_replayed_rejects_non_string_id() -> None:
    store = InMemoryDLQStore()
    with pytest.raises(DLQError):
        store.mark_replayed(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Capacity overflow
# ---------------------------------------------------------------------------


def test_overflow_evicts_oldest_and_records_event() -> None:
    store = InMemoryDLQStore(capacity=3)
    ids = []
    for i in range(5):  # 5 entries into a capacity-3 store
        e = _entry(webhook_id=f"wh-{i}")
        store.enqueue(e)
        ids.append(e.id)
    assert store.count_total() == 3
    # The first two should have been evicted; only ids[2..4] remain.
    remaining_ids = {p.id for p in store.list_pending()}
    assert remaining_ids == set(ids[2:])
    # Two eviction events recorded — for ids[0] and ids[1].
    evictions = store.evictions()
    assert len(evictions) == 2
    assert evictions[0].evicted_id == ids[0]
    assert evictions[1].evicted_id == ids[1]
    for ev in evictions:
        assert isinstance(ev, EvictionEvent)
        assert ev.evicted_at.tzinfo is not None


def test_capacity_must_be_positive_int() -> None:
    for bad in (0, -1, "1", 1.5):
        with pytest.raises(DLQError):
            InMemoryDLQStore(capacity=bad)  # type: ignore[arg-type]


def test_default_capacity_is_10000() -> None:
    assert DEFAULT_DLQ_CAPACITY == 10_000


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_enqueue_no_duplicates_no_loss() -> None:
    store = InMemoryDLQStore(capacity=10_000)
    barrier = threading.Barrier(20)

    def worker(thread_id: int) -> None:
        barrier.wait()
        for i in range(25):
            store.enqueue(_entry(webhook_id=f"wh-{thread_id}-{i}"))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.count_total() == 20 * 25
    ids = {e.id for e in store.list_pending()}
    assert len(ids) == 500  # all unique


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_inmemory_store_satisfies_dlq_store_protocol() -> None:
    store: DLQStore = InMemoryDLQStore()
    assert isinstance(store, DLQStore)
