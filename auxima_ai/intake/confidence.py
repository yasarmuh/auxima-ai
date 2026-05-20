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
    "AutoAccept",
    "ConfidenceDecision",
    "ConfidenceError",
    "HoldForReview",
    "decide",
)
