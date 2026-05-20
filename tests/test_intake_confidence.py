"""Confidence → auto-accept / hold-for-review decision (P1-10 / AC-01).

A pure decision over an extraction confidence score: at or above the
threshold the quote auto-accepts; below it, it's held for broker review.
Returns a typed sum-type (not a bare bool) so the caller routes 1:1, mirroring
the IntakeOutcome pattern in service.py.

The default threshold (0.8) is an engineering placeholder — the
IA-defensible number is GAP-1 / user-owned and must NOT be pinned here.
"""
from __future__ import annotations

import math

import pytest

from auxima_ai.intake.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    AutoAccept,
    ConfidenceError,
    HoldForReview,
    decide,
)


def test_above_threshold_auto_accepts() -> None:
    d = decide(0.95)
    assert isinstance(d, AutoAccept)
    assert d.score == 0.95
    assert d.threshold == DEFAULT_CONFIDENCE_THRESHOLD


def test_exactly_at_threshold_auto_accepts() -> None:
    assert isinstance(decide(DEFAULT_CONFIDENCE_THRESHOLD), AutoAccept)


def test_below_threshold_holds_for_review() -> None:
    d = decide(0.5)
    assert isinstance(d, HoldForReview)
    assert d.score == 0.5


def test_custom_threshold() -> None:
    assert isinstance(decide(0.6, threshold=0.5), AutoAccept)
    assert isinstance(decide(0.4, threshold=0.5), HoldForReview)


def test_boundaries_0_and_1() -> None:
    assert isinstance(decide(0.0, threshold=0.0), AutoAccept)  # 0 >= 0
    assert isinstance(decide(1.0), AutoAccept)
    assert isinstance(decide(0.0), HoldForReview)


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -5.0])
def test_score_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ConfidenceError):
        decide(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_score_rejected(bad: float) -> None:
    with pytest.raises(ConfidenceError):
        decide(bad)


@pytest.mark.parametrize("bad", ["0.9", None, [], {}])
def test_non_numeric_score_rejected(bad: object) -> None:
    with pytest.raises(ConfidenceError):
        decide(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_t", [-0.1, 1.1, math.nan])
def test_invalid_threshold_rejected(bad_t: float) -> None:
    with pytest.raises(ConfidenceError):
        decide(0.9, threshold=bad_t)
