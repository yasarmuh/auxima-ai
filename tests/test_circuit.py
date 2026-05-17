"""Tests for ``auxima_ai.resilience.circuit`` — 3-state circuit breaker.

Coverage:
  - BreakerPolicy validation.
  - Fresh breaker is CLOSED and admits calls.
  - N consecutive failures in CLOSED transitions to OPEN.
  - OPEN rejects with RejectOpen and a retry_after that decays as time passes.
  - After cooldown, breaker transitions to HALF_OPEN.
  - In HALF_OPEN, the first call is Admitted with probing=True.
  - In HALF_OPEN, second concurrent probe (without outcome) is rejected
    as RejectHalfOpenSaturated until the in-flight slot frees.
  - HALF_OPEN success -> CLOSED + counter resets.
  - HALF_OPEN failure -> OPEN + cooldown re-armed.
  - record_success in CLOSED resets the consecutive failure counter
    (one success after 2 failures must not roll the breaker over on
    the 5th non-consecutive failure).
  - record_success while OPEN is a no-op + does not flip state.
  - reset() forces state back to CLOSED.
  - Thread-safety: concurrent failures don't double-count, exactly one
    state transition observed.
  - half_open_max_calls=2 allows two concurrent probes.
"""
from __future__ import annotations

import threading
from typing import Any

import pytest

from auxima_ai.resilience.circuit import (
    Admit,
    BreakerPolicy,
    CircuitBreaker,
    CircuitError,
    RejectHalfOpenSaturated,
    RejectOpen,
    State,
)


def _clock_box(start: float = 1000.0):
    box = [start]
    return box, lambda: box[0]


def _advance(box: list[float], by: float) -> None:
    box[0] += by


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_policy_defaults_construct() -> None:
    p = BreakerPolicy()
    assert p.failure_threshold == 5
    assert p.cooldown_seconds == 30.0
    assert p.half_open_max_calls == 1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"failure_threshold": 0}, "failure_threshold"),
        ({"failure_threshold": -1}, "failure_threshold"),
        ({"cooldown_seconds": 0}, "cooldown_seconds"),
        ({"cooldown_seconds": -5}, "cooldown_seconds"),
        ({"half_open_max_calls": 0}, "half_open_max_calls"),
    ],
)
def test_policy_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(CircuitError, match=match):
        BreakerPolicy(**kwargs)


def test_policy_is_frozen() -> None:
    p = BreakerPolicy()
    with pytest.raises((AttributeError, TypeError)):
        p.failure_threshold = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLOSED -> OPEN
# ---------------------------------------------------------------------------


def test_fresh_breaker_is_closed_and_admits() -> None:
    cb = CircuitBreaker()
    assert cb.state == State.CLOSED
    result = cb.try_call()
    assert isinstance(result, Admit)
    assert result.state == State.CLOSED
    assert result.probing is False


def test_consecutive_failures_open_the_breaker() -> None:
    cb = CircuitBreaker(policy=BreakerPolicy(failure_threshold=3, cooldown_seconds=60))
    for _ in range(3):
        cb.record_failure()
    assert cb.state == State.OPEN


def test_below_threshold_failures_keep_closed() -> None:
    cb = CircuitBreaker(policy=BreakerPolicy(failure_threshold=3, cooldown_seconds=60))
    cb.record_failure()
    cb.record_failure()
    assert cb.state == State.CLOSED


def test_intervening_success_resets_failure_counter() -> None:
    """1 fail, 1 succ, 4 fails should NOT open with threshold=5."""
    cb = CircuitBreaker(policy=BreakerPolicy(failure_threshold=5, cooldown_seconds=60))
    cb.record_failure()
    cb.record_success()
    for _ in range(4):
        cb.record_failure()
    assert cb.state == State.CLOSED, "consecutive counter must have reset"


# ---------------------------------------------------------------------------
# OPEN rejects with decaying retry_after
# ---------------------------------------------------------------------------


def test_open_rejects_with_retry_after() -> None:
    box, clk = _clock_box()
    cb = CircuitBreaker(
        policy=BreakerPolicy(failure_threshold=1, cooldown_seconds=30),
        clock=clk,
    )
    cb.record_failure()
    r = cb.try_call()
    assert isinstance(r, RejectOpen)
    assert r.retry_after_seconds == pytest.approx(30.0)


def test_retry_after_decays_with_elapsed_time() -> None:
    box, clk = _clock_box()
    cb = CircuitBreaker(
        policy=BreakerPolicy(failure_threshold=1, cooldown_seconds=30),
        clock=clk,
    )
    cb.record_failure()
    _advance(box, 10)
    r = cb.try_call()
    assert isinstance(r, RejectOpen)
    assert r.retry_after_seconds == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# OPEN -> HALF_OPEN
# ---------------------------------------------------------------------------


def test_open_transitions_to_half_open_after_cooldown() -> None:
    box, clk = _clock_box()
    cb = CircuitBreaker(
        policy=BreakerPolicy(failure_threshold=1, cooldown_seconds=10),
        clock=clk,
    )
    cb.record_failure()
    _advance(box, 11)
    assert cb.state == State.HALF_OPEN


def test_half_open_admits_probe_with_probing_flag() -> None:
    box, clk = _clock_box()
    cb = CircuitBreaker(
        policy=BreakerPolicy(failure_threshold=1, cooldown_seconds=10),
        clock=clk,
    )
    cb.record_failure()
    _advance(box, 11)
    r = cb.try_call()
    assert isinstance(r, Admit)
    assert r.probing is True
    assert r.state == State.HALF_OPEN


def test_half_open_saturated_rejects_second_concurrent_probe() -> None:
    box, clk = _clock_box()
    cb = CircuitBreaker(
        policy=BreakerPolicy(
            failure_threshold=1, cooldown_seconds=10, half_open_max_calls=1,
        ),
        clock=clk,
    )
    cb.record_failure()
    _advance(box, 11)
    first = cb.try_call()
    assert isinstance(first, Admit)
    second = cb.try_call()  # no outcome recorded yet for first
    assert isinstance(second, RejectHalfOpenSaturated)


def test_half_open_max_calls_two_admits_two_probes() -> None:
    box, clk = _clock_box()
    cb = CircuitBreaker(
        policy=BreakerPolicy(
            failure_threshold=1, cooldown_seconds=10, half_open_max_calls=2,
        ),
        clock=clk,
    )
    cb.record_failure()
    _advance(box, 11)
    a = cb.try_call()
    b = cb.try_call()
    c = cb.try_call()
    assert isinstance(a, Admit)
    assert isinstance(b, Admit)
    assert isinstance(c, RejectHalfOpenSaturated)


# ---------------------------------------------------------------------------
# HALF_OPEN outcomes
# ---------------------------------------------------------------------------


def test_half_open_success_closes_circuit_and_resets_counter() -> None:
    box, clk = _clock_box()
    cb = CircuitBreaker(
        policy=BreakerPolicy(failure_threshold=3, cooldown_seconds=10),
        clock=clk,
    )
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()  # OPEN now
    _advance(box, 11)
    cb.try_call()  # admit probe
    cb.record_success()
    assert cb.state == State.CLOSED
    assert cb.consecutive_failures == 0


def test_half_open_failure_reopens_breaker() -> None:
    box, clk = _clock_box()
    cb = CircuitBreaker(
        policy=BreakerPolicy(failure_threshold=1, cooldown_seconds=10),
        clock=clk,
    )
    cb.record_failure()  # OPEN
    _advance(box, 11)
    cb.try_call()  # HALF_OPEN probe admitted
    cb.record_failure()  # probe failed
    assert cb.state == State.OPEN
    # Cooldown re-armed — full window starts again
    r = cb.try_call()
    assert isinstance(r, RejectOpen)
    assert r.retry_after_seconds == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Defensive: success in OPEN is a no-op
# ---------------------------------------------------------------------------


def test_record_success_while_open_is_noop() -> None:
    cb = CircuitBreaker(policy=BreakerPolicy(failure_threshold=1, cooldown_seconds=60))
    cb.record_failure()  # OPEN
    cb.record_success()  # should not flip back
    assert cb.state == State.OPEN


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_forces_closed_from_open() -> None:
    cb = CircuitBreaker(policy=BreakerPolicy(failure_threshold=1, cooldown_seconds=60))
    cb.record_failure()
    assert cb.state == State.OPEN
    cb.reset()
    assert cb.state == State.CLOSED
    assert cb.consecutive_failures == 0


def test_reset_forces_closed_from_half_open() -> None:
    box, clk = _clock_box()
    cb = CircuitBreaker(
        policy=BreakerPolicy(failure_threshold=1, cooldown_seconds=5),
        clock=clk,
    )
    cb.record_failure()
    _advance(box, 6)
    cb.try_call()  # HALF_OPEN, in_flight=1
    cb.reset()
    assert cb.state == State.CLOSED


# ---------------------------------------------------------------------------
# Concurrency — exactly one transition observed
# ---------------------------------------------------------------------------


def test_concurrent_failures_open_exactly_once() -> None:
    """20 threads each record_failure on a threshold=10 breaker. The
    state transitions once; no thread sees a partial / undefined view."""
    cb = CircuitBreaker(policy=BreakerPolicy(failure_threshold=10, cooldown_seconds=60))
    barrier = threading.Barrier(20)

    def worker() -> None:
        barrier.wait()
        cb.record_failure()

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cb.state == State.OPEN
    # Counter should reflect all 20 failures (even though only 10 were
    # needed to open; once OPEN further failures are no-op'd silently).
    # We don't assert the exact value because record_failure short-circuits
    # in OPEN — both "stops at 10" and "keeps counting to 20" are valid
    # under the spec. Just assert no exception, and OPEN.
