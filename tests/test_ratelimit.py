"""Tests for ``auxima_ai.ratelimit.bucket`` — token bucket + per-tenant limiter.

Coverage:
  - Fresh bucket admits its first capacity-worth of requests.
  - Once empty, requests are denied with a retry_after that respects refill rate.
  - Refill caps at capacity.
  - try_consume(N) where N <= remaining decrements correctly.
  - try_consume(N) where N > capacity raises (request can never succeed).
  - tokens=0 / negative raises.
  - level property reflects lazy refill (does not consume).
  - Construction validation rejects bad capacity / refill.
  - Thread safety: concurrent calls add up consistently.
  - Per-tenant limiter isolates tenants.
  - Per-tenant limiter is lazy (no bucket until first request).
  - Tenant-id validation rejects empty / non-string.
  - seconds_until_bucket_full helper.
"""
from __future__ import annotations

import threading

import pytest

from auxima_ai.ratelimit.bucket import (
    Allowed,
    Denied,
    PerTenantRateLimiter,
    RateLimitError,
    TokenBucket,
    seconds_until_bucket_full,
)


# ---------------------------------------------------------------------------
# Helper — driveable clock
# ---------------------------------------------------------------------------


def _clock_box(start: float = 0.0) -> tuple[list[float], "callable[[], float]"]:
    box = [start]
    return box, lambda: box[0]


# ---------------------------------------------------------------------------
# TokenBucket — initial state + admission
# ---------------------------------------------------------------------------


def test_fresh_bucket_is_full_and_admits_capacity_requests() -> None:
    box, clk = _clock_box()
    b = TokenBucket(capacity=3, refill_per_second=1, clock=clk)
    assert isinstance(b.try_consume(), Allowed)
    assert isinstance(b.try_consume(), Allowed)
    assert isinstance(b.try_consume(), Allowed)
    # Bucket now empty
    denied = b.try_consume()
    assert isinstance(denied, Denied)
    assert denied.remaining == 0
    assert denied.retry_after_seconds == 1.0


def test_denied_retry_after_scales_with_refill_rate() -> None:
    box, clk = _clock_box()
    b = TokenBucket(capacity=1, refill_per_second=0.5, clock=clk)
    b.try_consume()  # empty it
    d = b.try_consume()
    assert isinstance(d, Denied)
    assert d.retry_after_seconds == pytest.approx(2.0)  # 1 token / 0.5 per sec


# ---------------------------------------------------------------------------
# Refill
# ---------------------------------------------------------------------------


def test_refill_caps_at_capacity() -> None:
    box, clk = _clock_box()
    b = TokenBucket(capacity=3, refill_per_second=10, clock=clk)
    b.try_consume()
    b.try_consume()
    box[0] += 1000  # huge elapsed -> tons of tokens would accrue
    assert b.level == 3  # capped


def test_partial_refill_admits_partial_burst() -> None:
    box, clk = _clock_box()
    b = TokenBucket(capacity=5, refill_per_second=1, clock=clk)
    for _ in range(5):
        b.try_consume()
    assert b.level == 0
    box[0] += 2  # 2 tokens refilled
    assert isinstance(b.try_consume(), Allowed)
    assert isinstance(b.try_consume(), Allowed)
    assert isinstance(b.try_consume(), Denied)


def test_negative_or_zero_elapsed_is_noop() -> None:
    """Monotonic clock can return the same value inside one tick — must not rewind."""
    box, clk = _clock_box(start=100.0)
    b = TokenBucket(capacity=2, refill_per_second=1, clock=clk)
    b.try_consume()
    pre = b.level
    # clock did not advance; trigger another refill via level access
    post = b.level
    assert post == pre


# ---------------------------------------------------------------------------
# try_consume(N)
# ---------------------------------------------------------------------------


def test_consume_multiple_tokens_at_once() -> None:
    box, clk = _clock_box()
    b = TokenBucket(capacity=5, refill_per_second=1, clock=clk)
    a = b.try_consume(3)
    assert isinstance(a, Allowed)
    assert a.remaining == 2


def test_consume_exact_remaining_admitted() -> None:
    box, clk = _clock_box()
    b = TokenBucket(capacity=5, refill_per_second=1, clock=clk)
    b.try_consume(3)
    assert isinstance(b.try_consume(2), Allowed)


def test_consume_more_than_remaining_denied_with_correct_wait() -> None:
    box, clk = _clock_box()
    b = TokenBucket(capacity=5, refill_per_second=1, clock=clk)
    b.try_consume(4)  # remaining = 1
    d = b.try_consume(3)
    assert isinstance(d, Denied)
    assert d.retry_after_seconds == pytest.approx(2.0)  # need 2 more / 1 per sec


def test_consume_more_than_capacity_raises() -> None:
    b = TokenBucket(capacity=5, refill_per_second=1)
    with pytest.raises(RateLimitError, match="exceeds capacity"):
        b.try_consume(6)


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_consume_non_positive_tokens_raises(bad: float) -> None:
    b = TokenBucket(capacity=5, refill_per_second=1)
    with pytest.raises(RateLimitError, match="tokens"):
        b.try_consume(bad)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [0, -1])
def test_capacity_must_be_positive(cap: float) -> None:
    with pytest.raises(RateLimitError, match="capacity"):
        TokenBucket(capacity=cap, refill_per_second=1)


@pytest.mark.parametrize("refill", [0, -1])
def test_refill_must_be_positive(refill: float) -> None:
    with pytest.raises(RateLimitError, match="refill_per_second"):
        TokenBucket(capacity=1, refill_per_second=refill)


# ---------------------------------------------------------------------------
# level
# ---------------------------------------------------------------------------


def test_level_does_not_consume() -> None:
    box, clk = _clock_box()
    b = TokenBucket(capacity=5, refill_per_second=1, clock=clk)
    pre = b.level
    post = b.level
    assert pre == post == 5


# ---------------------------------------------------------------------------
# Thread safety — concurrent consumes never over-admit
# ---------------------------------------------------------------------------


def test_concurrent_consume_does_not_over_admit() -> None:
    # 100 threads each try to consume 1; capacity is 50. We must see
    # exactly 50 Allowed and 50 Denied (or the remaining must never
    # have gone below zero).
    b = TokenBucket(capacity=50, refill_per_second=0.0001)  # negligible refill
    results: list = []
    lock = threading.Lock()
    barrier = threading.Barrier(100)

    def worker() -> None:
        barrier.wait()
        r = b.try_consume()
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    allowed = sum(1 for r in results if isinstance(r, Allowed))
    denied = sum(1 for r in results if isinstance(r, Denied))
    assert allowed == 50, f"expected 50 Allowed; got {allowed}"
    assert denied == 50
    assert all(
        isinstance(r, Denied) or r.remaining >= 0 for r in results
    )


# ---------------------------------------------------------------------------
# PerTenantRateLimiter
# ---------------------------------------------------------------------------


def test_per_tenant_isolation() -> None:
    box, clk = _clock_box()
    limiter = PerTenantRateLimiter(capacity=2, refill_per_second=1, clock=clk)
    a1 = limiter.try_consume("tenant-a")
    a2 = limiter.try_consume("tenant-a")
    a3 = limiter.try_consume("tenant-a")
    b1 = limiter.try_consume("tenant-b")
    b2 = limiter.try_consume("tenant-b")
    assert isinstance(a1, Allowed)
    assert isinstance(a2, Allowed)
    assert isinstance(a3, Denied), "tenant-a should be exhausted"
    assert isinstance(b1, Allowed), "tenant-b unaffected by tenant-a"
    assert isinstance(b2, Allowed)


def test_per_tenant_is_lazy() -> None:
    limiter = PerTenantRateLimiter()
    assert limiter.tenant_count() == 0
    limiter.try_consume("t1")
    assert limiter.tenant_count() == 1
    limiter.try_consume("t1")
    assert limiter.tenant_count() == 1
    limiter.try_consume("t2")
    assert limiter.tenant_count() == 2


def test_per_tenant_bucket_for_returns_same_instance() -> None:
    limiter = PerTenantRateLimiter()
    a = limiter.bucket_for("t1")
    b = limiter.bucket_for("t1")
    assert a is b


@pytest.mark.parametrize("bad", ["", None, 42, [], {}])
def test_per_tenant_rejects_bad_tenant_id(bad: object) -> None:
    limiter = PerTenantRateLimiter()
    with pytest.raises(RateLimitError, match="tenant_id"):
        limiter.try_consume(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("cap", [0, -1])
def test_per_tenant_validates_capacity_at_construction(cap: float) -> None:
    with pytest.raises(RateLimitError, match="capacity"):
        PerTenantRateLimiter(capacity=cap, refill_per_second=1)


# ---------------------------------------------------------------------------
# seconds_until_bucket_full
# ---------------------------------------------------------------------------


def test_seconds_until_bucket_full_for_full_bucket_is_zero() -> None:
    b = TokenBucket(capacity=5, refill_per_second=1)
    assert seconds_until_bucket_full(b) == 0.0


def test_seconds_until_bucket_full_for_empty_bucket() -> None:
    box, clk = _clock_box()
    b = TokenBucket(capacity=5, refill_per_second=2, clock=clk)
    for _ in range(5):
        b.try_consume()
    assert seconds_until_bucket_full(b) == pytest.approx(5 / 2)
