"""Real-Redis integration tests for RedisNonceStore (S-54 §3.3 / R5 / GAP-16 AC-1, AC-6).

These complement ``test_auth_nonce_redis.py`` (which drives a *fake* client) by
exercising the store against a REAL redis-py client and a REAL redis server —
the two halves that could not honestly be faked:

  - **AC-1 (cross-process replay):** a header captured and replayed against a
    *different* sidecar replica must still be caught. We model "different
    replica" as two independent :class:`RedisNonceStore` instances, each with
    its own redis-py client/connection. The shared namespace in redis is what
    makes the second claim a :class:`NonceReplay`.
  - **AC-6 (fail-closed on outage):** a store pointed at a dead redis must raise
    :class:`NonceStoreUnavailable` (→ middleware 503), never silently skip
    replay protection.

Connection comes from ``REDIS_URL`` (default ``redis://localhost:6379/0``). If
no server is reachable, every test in this module SKIPS — so contributors
without redis, and the no-frappe isolation job, are unaffected. CI provides a
redis service container (see ``.github/workflows/ci.yml``) so these actually run
there.
"""
from __future__ import annotations

import os
import time
import uuid

import pytest

from auxima_ai.auth_nonce import (
    NonceFresh,
    NonceReplay,
    NonceStoreUnavailable,
    RedisNonceStore,
)

# Skip the whole module if redis-py is not installed (it is an optional/
# deploy-time dependency — the store itself never imports it at runtime).
redis = pytest.importorskip("redis")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _reachable(url: str) -> bool:
    """True if a redis server answers PING within a short timeout."""
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=0.5)
        return bool(client.ping())
    except Exception:
        return False


# Skip (not fail) when no live redis — keeps local/dev + isolation runs green.
pytestmark = pytest.mark.skipif(
    not _reachable(REDIS_URL),
    reason=f"no reachable redis at {REDIS_URL}; integration tests skipped",
)


def _client():
    return redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.5)


def _unique_key_id() -> str:
    """A fresh key_id per test so runs never collide on a shared server."""
    return f"itest-{uuid.uuid4().hex[:12]}"


def test_real_redis_fresh_then_replay_same_instance() -> None:
    store = RedisNonceStore.with_redis_errors(_client())
    key_id, nonce = _unique_key_id(), "nonceAAA"

    first = store.claim(key_id, nonce, 600)
    assert isinstance(first, NonceFresh)

    second = store.claim(key_id, nonce, 600)
    assert isinstance(second, NonceReplay)


def test_real_redis_replay_across_two_instances() -> None:
    """AC-1: same (key_id, nonce) seen by a *second* store/connection is a replay.

    Two independent RedisNonceStore instances stand in for two sidecar replicas
    sharing one redis. The replay must be caught cross-instance — which the
    in-memory store could not do (each replica has private memory).
    """
    store_a = RedisNonceStore.with_redis_errors(_client())
    store_b = RedisNonceStore.with_redis_errors(_client())
    key_id, nonce = _unique_key_id(), "shared-nonce"

    assert isinstance(store_a.claim(key_id, nonce, 600), NonceFresh)
    # Different instance, different connection — still rejected via shared key.
    assert isinstance(store_b.claim(key_id, nonce, 600), NonceReplay)


def test_real_redis_distinct_pairs_are_independent() -> None:
    store = RedisNonceStore.with_redis_errors(_client())
    base = _unique_key_id()

    assert isinstance(store.claim(base, "n1", 600), NonceFresh)
    assert isinstance(store.claim(base + "-other", "n1", 600), NonceFresh)  # other key_id
    assert isinstance(store.claim(base, "n2", 600), NonceFresh)            # other nonce


def test_real_redis_nonce_expires_after_ttl() -> None:
    """The atomic SET uses EX, so a nonce is claimable again once its TTL lapses.

    Uses a 1 s TTL + a small margin. This proves the EX argument is honoured by
    a real server (the fake test can only assert the value passed, not effect).
    """
    store = RedisNonceStore.with_redis_errors(_client())
    key_id, nonce = _unique_key_id(), "expiring"

    assert isinstance(store.claim(key_id, nonce, 1), NonceFresh)
    assert isinstance(store.claim(key_id, nonce, 1), NonceReplay)
    time.sleep(1.3)
    # TTL lapsed → the key is gone → fresh again.
    assert isinstance(store.claim(key_id, nonce, 1), NonceFresh)


def test_real_redis_unavailable_fails_closed() -> None:
    """AC-6: a store pointed at a dead redis raises NonceStoreUnavailable (→503).

    ``with_redis_errors`` wires redis-py's own ConnectionError/TimeoutError, so
    the refused connection at a dead port is classified as "unavailable" and the
    store fails closed rather than skipping replay protection.
    """
    dead = redis.Redis(host="127.0.0.1", port=6390, socket_connect_timeout=0.25)
    store = RedisNonceStore.with_redis_errors(dead)

    with pytest.raises(NonceStoreUnavailable):
        store.claim(_unique_key_id(), "nonceAAA", 600)
