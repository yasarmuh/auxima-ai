"""Unit tests for RedisNonceStore (S-54 §3.3 / R5 / GAP-16 AC-1, AC-6).

There is NO real Redis in this environment, so these drive a FAKE in-memory
client that mimics the ONE redis primitive the store relies on:
``SET key value NX EX ttl`` (returns True if set, None if the key already
exists). This proves the store's logic — atomic claim, replay detection,
fail-closed on connection error, key namespacing.

NOT covered here (flagged, not faked): the real cross-process replay (AC-1)
and the kill-Redis-mid-request 503 (AC-6) need a staging Redis. See the
backlog R9-4 note.
"""
from __future__ import annotations

import pytest

from auxima_ai.auth_nonce import (
    InvalidNonceError,
    NonceFresh,
    NonceReplay,
    NonceStore,
    NonceStoreError,
    NonceStoreUnavailable,
    RedisNonceStore,
)


class _FakeConnError(Exception):
    """Stand-in for redis.exceptions.ConnectionError (injected by the test)."""


class _FakeRedis:
    """Minimal fake of the redis-py client surface the store uses.

    Implements only ``set(name, value, nx=, ex=)`` with the real return
    contract: ``True`` when the key was set, ``None`` when NX prevented it.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.ex_seen: dict[str, int] = {}
        self.nx_seen: list[bool] = []
        self.fail = fail

    def set(self, name, value, *, nx=False, ex=None):
        if self.fail:
            raise _FakeConnError("connection refused")
        self.nx_seen.append(nx)
        self.ex_seen[name] = ex
        if nx and name in self.store:
            return None  # NX: key exists → not set
        self.store[name] = value
        return True


def _store(client):
    # Inject the fake's connection-error type so the store knows what
    # "unreachable" looks like without a hard redis dependency.
    return RedisNonceStore(client, unavailable_errors=(_FakeConnError,))


def test_first_claim_is_fresh_replay_on_second() -> None:
    store = _store(_FakeRedis())
    first = store.claim("p2026q2", "nonceAAA", 600)
    assert isinstance(first, NonceFresh)
    second = store.claim("p2026q2", "nonceAAA", 600)
    assert isinstance(second, NonceReplay)


def test_uses_atomic_set_nx_ex_with_namespaced_key() -> None:
    client = _FakeRedis()
    _store(client).claim("p2026q2", "nonceAAA", 600)
    key = "auxima_ai:nonce:p2026q2:nonceAAA"
    assert key in client.store
    assert client.ex_seen[key] == 600          # EX ttl
    assert client.nx_seen == [True]            # NX (atomic, no TOCTOU)


def test_distinct_key_id_or_nonce_are_independent() -> None:
    client = _FakeRedis()
    store = _store(client)
    assert isinstance(store.claim("p2026q2", "n1", 600), NonceFresh)
    assert isinstance(store.claim("s2026q3", "n1", 600), NonceFresh)  # other key_id
    assert isinstance(store.claim("p2026q2", "n2", 600), NonceFresh)  # other nonce


def test_connection_error_fails_closed() -> None:
    store = _store(_FakeRedis(fail=True))
    with pytest.raises(NonceStoreUnavailable):
        store.claim("p2026q2", "nonceAAA", 600)


def test_invalid_input_rejected_before_touching_redis() -> None:
    client = _FakeRedis()
    with pytest.raises(InvalidNonceError):
        _store(client).claim("p2026q2", "", 600)
    assert client.store == {}  # never reached the client


def test_nonpositive_ttl_rejected() -> None:
    with pytest.raises(NonceStoreError):
        _store(_FakeRedis()).claim("p2026q2", "nonceAAA", 0)


def test_satisfies_nonce_store_protocol() -> None:
    assert isinstance(_store(_FakeRedis()), NonceStore)


def test_with_redis_errors_factory_builds_a_working_store() -> None:
    # redis-py is not installed here, so the factory falls back to builtin
    # socket errors; the store must still claim/replay correctly.
    store = RedisNonceStore.with_redis_errors(_FakeRedis())
    assert isinstance(store.claim("p2026q2", "n1", 600), NonceFresh)
    assert isinstance(store.claim("p2026q2", "n1", 600), NonceReplay)
    # OSError is a builtin socket error → in the default fail-closed set.
    assert OSError in store.unavailable_errors
