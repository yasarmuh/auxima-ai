"""Tests for the length-aware confidence function (P1-10 §4.2 / GAP-7,12)."""
from __future__ import annotations

import pytest

from auxima_ai.intake.confidence import (
    MIN_RELIABLE_TEXT_CHARS,
    ConfidenceError,
    compute_confidence,
    decide,
)


def test_full_length_text_trusts_model_confidence() -> None:
    # char_count >= MIN_RELIABLE → length_factor 1.0 → final == model_confidence
    final = compute_confidence(0.9, MIN_RELIABLE_TEXT_CHARS)
    assert final == pytest.approx(0.9)


def test_above_threshold_text_caps_at_one() -> None:
    final = compute_confidence(0.75, MIN_RELIABLE_TEXT_CHARS * 10)
    assert final == pytest.approx(0.75)


def test_garbage_high_confidence_is_dragged_down() -> None:
    # The core guarantee: a model that returns 1.0 over a near-empty extraction
    # must NOT auto-accept. 10 chars / 200 = 0.05 factor → 0.05 final.
    final = compute_confidence(1.0, 10)
    assert final == pytest.approx(10 / MIN_RELIABLE_TEXT_CHARS)
    # and it falls below the default auto-accept bar
    from auxima_ai.intake.confidence import HoldForReview

    assert isinstance(decide(final), HoldForReview)


def test_zero_chars_gives_zero_confidence() -> None:
    assert compute_confidence(1.0, 0) == 0.0


def test_linear_scaling_below_threshold() -> None:
    final = compute_confidence(0.8, MIN_RELIABLE_TEXT_CHARS // 2)
    assert final == pytest.approx(0.4)  # 0.8 * 0.5


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_bad_model_confidence_raises(bad: float) -> None:
    with pytest.raises(ConfidenceError):
        compute_confidence(bad, 200)


def test_negative_char_count_raises() -> None:
    with pytest.raises(ConfidenceError):
        compute_confidence(0.5, -1)


def test_bool_char_count_rejected() -> None:
    with pytest.raises(ConfidenceError):
        compute_confidence(0.5, True)  # type: ignore[arg-type]


def test_nonpositive_min_reliable_raises() -> None:
    with pytest.raises(ConfidenceError):
        compute_confidence(0.5, 100, min_reliable_chars=0)
