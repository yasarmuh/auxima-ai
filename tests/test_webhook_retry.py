"""Tests for ``auxima_ai.webhooks.retry`` — backoff + decision policy.

Coverage:
  - 2xx -> DeliverySuccess.
  - Each transient status (408/425/429/500/502/503/504) -> RetryScheduled.
  - Other 4xx (400/401/403/404/422) -> GiveUpPermanent.
  - Network failure (status_code=None) -> RetryScheduled.
  - attempt == max_attempts -> GiveUpExhausted (don't burn the last spin).
  - Retry-After header (delta-seconds) overrides the backoff.
  - Retry-After malformed / negative / HTTP-date -> ignored (falls back).
  - Retry-After clamped to max_seconds.
  - Backoff is non-negative and within [0, expo].
  - Backoff caps at max_seconds.
  - Backoff is deterministic with an injected RNG.
  - RetryPolicy validation rejects bad params.
  - All decision types are frozen.
"""
from __future__ import annotations

from typing import Any

import pytest

from auxima_ai.webhooks.retry import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_SECONDS,
    DeliverySuccess,
    GiveUpExhausted,
    GiveUpPermanent,
    RetryPolicy,
    RetryScheduled,
    compute_delay,
    evaluate,
)


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy_is_constructable() -> None:
    p = RetryPolicy()
    assert p.base_seconds == 1.0
    assert p.factor == 2.0
    assert p.max_seconds == DEFAULT_MAX_SECONDS
    assert p.max_attempts == DEFAULT_MAX_ATTEMPTS


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"base_seconds": 0}, "base_seconds"),
        ({"base_seconds": -1}, "base_seconds"),
        ({"factor": 0.5}, "factor"),
        ({"max_seconds": 0.1, "base_seconds": 1.0}, "max_seconds"),
        ({"max_attempts": 0}, "max_attempts"),
        ({"max_attempts": -3}, "max_attempts"),
    ],
)
def test_policy_validation_rejects_bad_params(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RetryPolicy(**kwargs)


def test_policy_is_frozen() -> None:
    p = RetryPolicy()
    with pytest.raises((AttributeError, TypeError)):
        p.base_seconds = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [200, 201, 202, 204, 207, 299])
def test_2xx_is_success(code: int) -> None:
    d = evaluate(attempt=1, status_code=code)
    assert isinstance(d, DeliverySuccess)
    assert d.attempt == 1


# ---------------------------------------------------------------------------
# Transient -> RetryScheduled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [408, 425, 429, 500, 502, 503, 504])
def test_transient_status_schedules_retry(code: int) -> None:
    d = evaluate(attempt=1, status_code=code, rng=lambda a, b: 0.5)
    assert isinstance(d, RetryScheduled)
    assert d.next_attempt == 2
    assert d.delay_seconds >= 0
    assert str(code) in d.reason


def test_network_failure_none_status_is_transient() -> None:
    d = evaluate(attempt=1, status_code=None, rng=lambda a, b: 0.5)
    assert isinstance(d, RetryScheduled)
    assert "network" in d.reason


# ---------------------------------------------------------------------------
# Non-retryable -> GiveUpPermanent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [400, 401, 403, 404, 405, 410, 422, 451])
def test_other_4xx_gives_up_permanent(code: int) -> None:
    d = evaluate(attempt=1, status_code=code)
    assert isinstance(d, GiveUpPermanent)
    assert d.status_code == code


@pytest.mark.parametrize("code", [301, 302, 304])
def test_3xx_treated_as_permanent_in_v1(code: int) -> None:
    """3xx isn't in the transient set; we don't follow redirects automatically
    for outbound webhooks (would be a confused-deputy / SSRF amplifier)."""
    d = evaluate(attempt=1, status_code=code)
    assert isinstance(d, GiveUpPermanent)


# ---------------------------------------------------------------------------
# Exhaustion
# ---------------------------------------------------------------------------


def test_max_attempts_reached_gives_up_exhausted() -> None:
    policy = RetryPolicy(max_attempts=3)
    d = evaluate(attempt=3, status_code=500, policy=policy)
    assert isinstance(d, GiveUpExhausted)
    assert d.attempt == 3


def test_max_attempts_minus_one_still_retries() -> None:
    policy = RetryPolicy(max_attempts=3)
    d = evaluate(attempt=2, status_code=500, policy=policy, rng=lambda a, b: 0.0)
    assert isinstance(d, RetryScheduled)
    assert d.next_attempt == 3


def test_max_attempts_one_means_no_retries_ever() -> None:
    policy = RetryPolicy(max_attempts=1)
    d = evaluate(attempt=1, status_code=500, policy=policy)
    assert isinstance(d, GiveUpExhausted)


# ---------------------------------------------------------------------------
# Retry-After honored
# ---------------------------------------------------------------------------


def test_retry_after_delta_seconds_overrides_backoff() -> None:
    """If the server says Retry-After: 5, we wait 5, not the jittered backoff."""
    d = evaluate(
        attempt=1,
        status_code=429,
        retry_after_header="5",
        rng=lambda a, b: 999.0,  # ensure jitter would be different
    )
    assert isinstance(d, RetryScheduled)
    assert d.delay_seconds == 5.0


def test_retry_after_clamped_to_max_seconds() -> None:
    policy = RetryPolicy(max_seconds=30.0)
    d = evaluate(
        attempt=1,
        status_code=429,
        retry_after_header="9999",
        policy=policy,
    )
    assert isinstance(d, RetryScheduled)
    assert d.delay_seconds == 30.0


@pytest.mark.parametrize(
    "bad_value",
    [
        "",
        "  ",
        "-3",
        "not-a-number",
        "Wed, 21 Oct 2015 07:28:00 GMT",  # HTTP-date form — deliberately unsupported
    ],
)
def test_retry_after_malformed_falls_back_to_backoff(bad_value: str) -> None:
    """Bad Retry-After is ignored; backoff used instead."""
    d = evaluate(
        attempt=1,
        status_code=503,
        retry_after_header=bad_value,
        rng=lambda a, b: 0.5,
    )
    assert isinstance(d, RetryScheduled)
    assert d.delay_seconds == 0.5


def test_retry_after_none_falls_back_to_backoff() -> None:
    d = evaluate(
        attempt=1,
        status_code=503,
        retry_after_header=None,
        rng=lambda a, b: 0.5,
    )
    assert isinstance(d, RetryScheduled)
    assert d.delay_seconds == 0.5


# ---------------------------------------------------------------------------
# compute_delay properties
# ---------------------------------------------------------------------------


def test_compute_delay_is_within_jitter_window() -> None:
    """Full-jitter picks uniformly in [0, expo]; our deterministic RNG returns
    exactly the high end -> delay == expo for that attempt."""
    policy = RetryPolicy(base_seconds=1.0, factor=2.0, max_seconds=600.0, max_attempts=10)
    # attempt 1: expo = 1; attempt 2: expo = 2; attempt 3: expo = 4; ...
    for attempt in (1, 2, 3, 5):
        delay = compute_delay(attempt, policy, rng=lambda a, b: b)
        assert delay == policy.base_seconds * (policy.factor ** (attempt - 1))


def test_compute_delay_is_zero_at_low_end_of_jitter() -> None:
    delay = compute_delay(1, RetryPolicy(), rng=lambda a, b: 0.0)
    assert delay == 0.0


def test_compute_delay_caps_at_max_seconds() -> None:
    policy = RetryPolicy(base_seconds=1.0, factor=10.0, max_seconds=5.0)
    delay = compute_delay(10, policy, rng=lambda a, b: b)
    assert delay == 5.0


def test_compute_delay_rejects_attempt_zero() -> None:
    with pytest.raises(ValueError, match="attempt"):
        compute_delay(0, RetryPolicy())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_evaluate_rejects_attempt_zero() -> None:
    with pytest.raises(ValueError, match="attempt"):
        evaluate(attempt=0, status_code=200)


# ---------------------------------------------------------------------------
# Frozen invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        DeliverySuccess(attempt=1),
        RetryScheduled(next_attempt=2, delay_seconds=1.0, reason="x"),
        GiveUpPermanent(attempt=1, status_code=400, reason="x"),
        GiveUpExhausted(attempt=8, status_code=500),
    ],
)
def test_decision_objects_are_frozen(obj: Any) -> None:
    with pytest.raises((AttributeError, TypeError)):
        obj.attempt = 99  # type: ignore[misc]
