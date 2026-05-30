"""Confidence-threshold decision for intake extraction (P1-10 / AC-01).

After the LLM extracts fields with a confidence score, the sidecar decides
whether the result auto-accepts or is held for broker review. This is the
pure decision layer — a typed sum-type (mirroring service.py's IntakeOutcome)
so the caller maps the decision 1:1 onto its action (the Frappe-side
Approval-Inbox auto-create vs hold-for-review is the cross-repo half).

Threshold note: :data:`DEFAULT_CONFIDENCE_THRESHOLD` (0.8) is an engineering
placeholder. The IA-defensible auto-accept bar is GAP-1 / user-owned and must
be pinned by the broker-grading process, NOT hard-coded here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

#: Engineering placeholder — NOT the regulatory bar (GAP-1, user-owned).
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.8

#: At/above this many extracted non-whitespace characters the length factor is
#: 1.0 (the model's self-confidence is trusted as-is). Below it, confidence is
#: scaled down linearly so a near-empty extraction can never auto-accept — this
#: is the §4.2 "garbage-but-high-confidence is impossible" guarantee (GAP-7/12).
MIN_RELIABLE_TEXT_CHARS: Final[int] = 200


class ConfidenceError(ValueError):
    """Invalid confidence score or threshold (out of [0,1] or non-finite)."""


@dataclass(frozen=True)
class AutoAccept:
    """score >= threshold — the extraction may be applied without review."""

    score: float
    threshold: float


@dataclass(frozen=True)
class HoldForReview:
    """score < threshold — the extraction must be held for broker review."""

    score: float
    threshold: float


ConfidenceDecision = AutoAccept | HoldForReview


def _validate_unit_interval(name: str, value: object) -> float:
    """Coerce + range-check a value to a finite float in [0.0, 1.0]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfidenceError(f"{name} must be a real number; got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise ConfidenceError(f"{name} must be finite; got {value!r}")
    if not (0.0 <= v <= 1.0):
        raise ConfidenceError(f"{name} must be in [0.0, 1.0]; got {v}")
    return v


def compute_confidence(
    model_confidence: float,
    extracted_text_chars: int,
    *,
    min_reliable_chars: int = MIN_RELIABLE_TEXT_CHARS,
) -> float:
    """Combine the model's self-confidence with an extracted-text-length factor.

    ``final = model_confidence * length_factor`` where
    ``length_factor = min(1.0, extracted_text_chars / min_reliable_chars)``.

    This is the "defined function whose inputs include the extracted-text
    length" the AC requires (§P1-10, GAP-7/12): a model that returns high
    self-confidence over an empty/near-empty extraction is scaled below the
    auto-accept bar, so garbage-but-high-confidence is impossible. The result
    is always a finite float in ``[0, 1]``.

    Raises :class:`ConfidenceError` on a non-unit-interval ``model_confidence``,
    a negative char count, or a non-positive ``min_reliable_chars`` (fail loud —
    a bad input must never silently produce an auto-accept).
    """
    mc = _validate_unit_interval("model_confidence", model_confidence)
    if isinstance(extracted_text_chars, bool) or not isinstance(extracted_text_chars, int):
        raise ConfidenceError(
            f"extracted_text_chars must be an int; got {type(extracted_text_chars).__name__}"
        )
    if extracted_text_chars < 0:
        raise ConfidenceError(f"extracted_text_chars must be >= 0; got {extracted_text_chars}")
    if min_reliable_chars <= 0:
        raise ConfidenceError(f"min_reliable_chars must be > 0; got {min_reliable_chars}")
    length_factor = min(1.0, extracted_text_chars / min_reliable_chars)
    return mc * length_factor


def decide(
    score: float, *, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> ConfidenceDecision:
    """Return :class:`AutoAccept` if ``score >= threshold`` else :class:`HoldForReview`.

    Both ``score`` and ``threshold`` must be finite numbers in ``[0.0, 1.0]``;
    anything else raises :class:`ConfidenceError` (fail loud — a NaN score
    must never silently auto-accept or silently hold).
    """
    s = _validate_unit_interval("score", score)
    t = _validate_unit_interval("threshold", threshold)
    return AutoAccept(s, t) if s >= t else HoldForReview(s, t)


__all__ = (
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "MIN_RELIABLE_TEXT_CHARS",
    "AutoAccept",
    "ConfidenceDecision",
    "ConfidenceError",
    "HoldForReview",
    "compute_confidence",
    "decide",
)
