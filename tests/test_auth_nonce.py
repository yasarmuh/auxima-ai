"""Tests for the nonce replay store (auxima_ai.auth_nonce / S-54 §3.3 / AC-1)."""
from __future__ import annotations

import pytest

from auxima_ai.auth_nonce import (
    DEFAULT_NONCE_TTL_SECONDS,
    InMemoryNonceStore,
    InvalidNonceError,
    NonceFresh,
    NonceReplay,
    NonceStore,
    NonceStoreError,
)


class _Clock:
    """Manually-advanced clock for deterministic TTL tests."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_first_claim_is_fresh() -> None:
    store = InMemoryNonceStore(clock=_Clock())
    result = store.claim("p2026q2", "nonce-abc")
    assert isinstance(result, NonceFresh)
    assert result.key_id == "p2026q2"
    assert result.nonce == "nonce-abc"


def test_second_claim_same_nonce_is_replay() -> None:
    # AC-1: the same nonce within the TTL is a replay.
    store = InMemoryNonceStore(clock=_Clock())
    assert isinstance(store.claim("p2026q2", "nonce-abc"), NonceFresh)
    assert isinstance(store.claim("p2026q2", "nonce-abc"), NonceReplay)


def test_same_nonce_different_key_id_is_fresh() -> None:
    # Nonce is scoped to its signing key — the same nonce string under a
    # different key_id is a distinct claim.
    store = InMemoryNonceStore(clock=_Clock())
    assert isinstance(store.claim("p2026q2", "n"), NonceFresh)
    assert isinstance(store.claim("s2026q3", "n"), NonceFresh)


def test_replay_then_fresh_after_ttl_expiry() -> None:
    clock = _Clock()
    store = InMemoryNonceStore(clock=clock)
    assert isinstance(store.claim("k", "n", ttl_seconds=600), NonceFresh)
    assert isinstance(store.claim("k", "n", ttl_seconds=600), NonceReplay)
    # Advance just past the TTL — the nonce can be seen again (the timestamp
    # window in auth_v1 would reject it by now anyway).
    clock.advance(601)
    assert isinstance(store.claim("k", "n", ttl_seconds=600), NonceFresh)


def test_nonce_still_replay_at_ttl_edge_minus_one() -> None:
    clock = _Clock()
    store = InMemoryNonceStore(clock=clock)
    store.claim("k", "n", ttl_seconds=600)
    clock.advance(599)
    assert isinstance(store.claim("k", "n", ttl_seconds=600), NonceReplay)


def test_default_ttl_is_600() -> None:
    assert DEFAULT_NONCE_TTL_SECONDS == 600


def test_size_reflects_live_nonces_and_evicts() -> None:
    clock = _Clock()
    store = InMemoryNonceStore(clock=clock)
    store.claim("k", "n1")
    store.claim("k", "n2")
    assert store.size() == 2
    clock.advance(DEFAULT_NONCE_TTL_SECONDS + 1)
    # size() evicts expired entries lazily.
    assert store.size() == 0


def test_clear_drops_everything() -> None:
    store = InMemoryNonceStore(clock=_Clock())
    store.claim("k", "n")
    store.clear()
    assert store.size() == 0
    # After clear, the same nonce is fresh again.
    assert isinstance(store.claim("k", "n"), NonceFresh)


@pytest.mark.parametrize("bad", ["", None, 123])
def test_empty_or_nonstring_key_id_raises(bad) -> None:
    store = InMemoryNonceStore(clock=_Clock())
    with pytest.raises(InvalidNonceError):
        store.claim(bad, "nonce")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", None, 123])
def test_empty_or_nonstring_nonce_raises(bad) -> None:
    store = InMemoryNonceStore(clock=_Clock())
    with pytest.raises(InvalidNonceError):
        store.claim("k", bad)  # type: ignore[arg-type]


def test_overlong_nonce_raises() -> None:
    store = InMemoryNonceStore(clock=_Clock())
    with pytest.raises(InvalidNonceError):
        store.claim("k", "x" * 257)


def test_nonpositive_ttl_raises() -> None:
    store = InMemoryNonceStore(clock=_Clock())
    with pytest.raises(NonceStoreError):
        store.claim("k", "n", ttl_seconds=0)
    with pytest.raises(NonceStoreError):
        store.claim("k", "n", ttl_seconds=-5)


def test_in_memory_store_satisfies_protocol() -> None:
    store = InMemoryNonceStore(clock=_Clock())
    assert isinstance(store, NonceStore)
