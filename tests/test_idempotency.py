"""Tests for ``auxima_ai.idempotency.store`` — keys, fingerprints, in-memory store.

Coverage:
  - IdempotencyKey validates tenant_id + key (non-empty, length ceiling).
  - fingerprint_payload is deterministic for key-order-irrelevant payloads.
  - try_begin returns BeginAccepted on first submission.
  - try_begin returns BeginInFlight on same-key duplicate before complete.
  - try_begin returns BeginReplay (with cached response) after complete.
  - try_begin returns BeginConflict on same-key-different-body submissions.
  - complete on an unknown key raises.
  - complete is idempotent (second call is a no-op).
  - TTL eviction: expired records are gone on the next access.
  - Cross-tenant: same wire key in two tenants does NOT collide.
  - Thread-safety: concurrent try_begin sees exactly one Accepted.
  - Injectable clock lets us drive time forward deterministically.
  - Validation paths raise the correct exception subclasses.
"""
from __future__ import annotations

import threading
from typing import Any

import pytest

from auxima_ai.idempotency.store import (
    BeginAccepted,
    BeginConflict,
    BeginInFlight,
    BeginReplay,
    DEFAULT_TTL_SECONDS,
    IdempotencyError,
    IdempotencyKey,
    InMemoryIdempotencyStore,
    InvalidFingerprintError,
    InvalidKeyError,
    MAX_KEY_LEN,
    fingerprint_payload,
)


# ---------------------------------------------------------------------------
# IdempotencyKey
# ---------------------------------------------------------------------------


def test_idempotency_key_valid() -> None:
    k = IdempotencyKey("tenant-acme", "op-123")
    assert k.tenant_id == "tenant-acme"
    assert k.key == "op-123"
    assert k.composite() == "tenant-acme::op-123"


@pytest.mark.parametrize("bad_tenant", ["", None, 42])
def test_idempotency_key_rejects_bad_tenant(bad_tenant: object) -> None:
    with pytest.raises(InvalidKeyError):
        IdempotencyKey(bad_tenant, "k")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_key", ["", None, 42])
def test_idempotency_key_rejects_bad_key(bad_key: object) -> None:
    with pytest.raises(InvalidKeyError):
        IdempotencyKey("t", bad_key)  # type: ignore[arg-type]


def test_idempotency_key_rejects_overlong_key() -> None:
    with pytest.raises(InvalidKeyError, match="exceeds maximum"):
        IdempotencyKey("t", "x" * (MAX_KEY_LEN + 1))


def test_idempotency_key_is_frozen() -> None:
    k = IdempotencyKey("t", "k")
    with pytest.raises((AttributeError, TypeError)):
        k.tenant_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic() -> None:
    a = fingerprint_payload({"b": 1, "a": 2, "c": [1, 2]})
    b = fingerprint_payload({"c": [1, 2], "a": 2, "b": 1})
    assert a == b
    assert len(a) == 64
    assert int(a, 16) >= 0


def test_fingerprint_differs_when_body_differs() -> None:
    a = fingerprint_payload({"x": 1})
    b = fingerprint_payload({"x": 2})
    assert a != b


def test_fingerprint_handles_nested_structures() -> None:
    fp = fingerprint_payload({"customer": {"id": 7, "emails": ["a@b.co"]}})
    assert len(fp) == 64


# ---------------------------------------------------------------------------
# Store — happy paths
# ---------------------------------------------------------------------------


def _store(now: float = 1_000_000.0) -> InMemoryIdempotencyStore:
    clock_box = [now]

    def clk() -> float:
        return clock_box[0]

    store = InMemoryIdempotencyStore(clock=clk)
    store._clock_box = clock_box  # type: ignore[attr-defined]  # test handle
    return store


def _advance(store: InMemoryIdempotencyStore, by_seconds: float) -> None:
    store._clock_box[0] += by_seconds  # type: ignore[attr-defined]


def test_first_submission_is_accepted() -> None:
    store = _store()
    key = IdempotencyKey("t1", "op-1")
    fp = fingerprint_payload({"x": 1})
    result = store.try_begin(key, fp)
    assert isinstance(result, BeginAccepted)
    assert result.key == key
    assert store.size() == 1


def test_in_flight_duplicate_returns_in_flight() -> None:
    store = _store()
    key = IdempotencyKey("t1", "op-1")
    fp = fingerprint_payload({"x": 1})
    store.try_begin(key, fp)
    again = store.try_begin(key, fp)
    assert isinstance(again, BeginInFlight)
    assert again.key == key


def test_completed_duplicate_returns_replay_with_response() -> None:
    store = _store()
    key = IdempotencyKey("t1", "op-1")
    fp = fingerprint_payload({"x": 1})
    store.try_begin(key, fp)
    response = {"status": "ok", "id": 42}
    store.complete(key, response)
    replay = store.try_begin(key, fp)
    assert isinstance(replay, BeginReplay)
    assert replay.response == response


def test_same_key_different_body_returns_conflict() -> None:
    store = _store()
    key = IdempotencyKey("t1", "op-1")
    store.try_begin(key, fingerprint_payload({"x": 1}))
    store.complete(key, {"ok": True})
    conflict = store.try_begin(key, fingerprint_payload({"x": 999}))
    assert isinstance(conflict, BeginConflict)
    assert conflict.seen_fingerprint != conflict.new_fingerprint


def test_in_flight_with_different_body_is_conflict() -> None:
    """Conflict detection runs before in-flight check; client must fix the key."""
    store = _store()
    key = IdempotencyKey("t1", "op-1")
    store.try_begin(key, fingerprint_payload({"x": 1}))
    conflict = store.try_begin(key, fingerprint_payload({"x": 2}))
    assert isinstance(conflict, BeginConflict)


# ---------------------------------------------------------------------------
# complete() semantics
# ---------------------------------------------------------------------------


def test_complete_unknown_key_raises() -> None:
    store = _store()
    with pytest.raises(InvalidKeyError, match="unknown"):
        store.complete(IdempotencyKey("t1", "ghost"), {"ok": True})


def test_complete_is_idempotent_on_already_completed() -> None:
    store = _store()
    key = IdempotencyKey("t1", "op-1")
    fp = fingerprint_payload({"x": 1})
    store.try_begin(key, fp)
    store.complete(key, {"first": True})
    store.complete(key, {"second": True})  # second call should be a no-op
    replay = store.try_begin(key, fp)
    assert isinstance(replay, BeginReplay)
    assert replay.response == {"first": True}, "first response wins"


def test_complete_validates_key_type() -> None:
    store = _store()
    with pytest.raises(InvalidKeyError):
        store.complete("not-a-key-object", {"ok": True})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cross-tenant isolation
# ---------------------------------------------------------------------------


def test_same_wire_key_in_two_tenants_does_not_collide() -> None:
    store = _store()
    fp = fingerprint_payload({"x": 1})
    a = store.try_begin(IdempotencyKey("tenant-a", "op-1"), fp)
    b = store.try_begin(IdempotencyKey("tenant-b", "op-1"), fp)
    assert isinstance(a, BeginAccepted)
    assert isinstance(b, BeginAccepted), "second tenant must not be blocked"
    assert store.size() == 2


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


def test_expired_record_is_evicted_on_next_access() -> None:
    store = _store()
    key = IdempotencyKey("t1", "op-1")
    fp = fingerprint_payload({"x": 1})
    store.try_begin(key, fp, ttl_seconds=10)
    store.complete(key, {"ok": True})
    assert store.size() == 1

    _advance(store, 11)  # past the TTL

    # Same key + same body now appears as a brand-new submission.
    result = store.try_begin(key, fp, ttl_seconds=10)
    assert isinstance(result, BeginAccepted)


def test_ttl_zero_is_rejected() -> None:
    store = _store()
    with pytest.raises(IdempotencyError, match="ttl_seconds"):
        store.try_begin(IdempotencyKey("t", "k"), fingerprint_payload({"x": 1}), ttl_seconds=0)


def test_default_ttl_is_24h() -> None:
    assert DEFAULT_TTL_SECONDS == 24 * 3600


# ---------------------------------------------------------------------------
# Fingerprint validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_fp",
    [
        "",                  # empty
        "abc",               # too short
        "g" * 64,            # wrong charset
        "f" * 63,            # length 63
        "f" * 65,            # length 65
        42,                  # not a string
        None,
    ],
)
def test_try_begin_rejects_bad_fingerprint(bad_fp: object) -> None:
    store = _store()
    with pytest.raises(InvalidFingerprintError):
        store.try_begin(IdempotencyKey("t", "k"), bad_fp)  # type: ignore[arg-type]


def test_try_begin_rejects_non_key_object() -> None:
    store = _store()
    with pytest.raises(InvalidKeyError):
        store.try_begin("not-a-key", fingerprint_payload({}))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Thread safety — concurrent try_begin produces exactly one Accepted
# ---------------------------------------------------------------------------


def test_concurrent_try_begin_yields_exactly_one_accepted() -> None:
    store = InMemoryIdempotencyStore()
    key = IdempotencyKey("t1", "op-1")
    fp = fingerprint_payload({"x": 1})
    results: list[Any] = []
    barrier = threading.Barrier(20)

    def worker() -> None:
        barrier.wait()
        results.append(store.try_begin(key, fp))

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    accepted = [r for r in results if isinstance(r, BeginAccepted)]
    in_flight = [r for r in results if isinstance(r, BeginInFlight)]
    assert len(accepted) == 1, f"expected exactly 1 Accepted; got {len(accepted)}"
    assert len(in_flight) == 19


# ---------------------------------------------------------------------------
# Utility — clear / size
# ---------------------------------------------------------------------------


def test_clear_drops_everything() -> None:
    store = _store()
    for i in range(5):
        store.try_begin(IdempotencyKey("t", f"k{i}"), fingerprint_payload({"i": i}))
    assert store.size() == 5
    store.clear()
    assert store.size() == 0
