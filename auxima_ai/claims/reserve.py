# Copyright (c) 2026, Auxilium Tech and contributors
"""Initial-reserve suggestion — pure Decimal, fail-closed, parameterised (P3-01).

The factor table and floor are COMMERCIAL defaults (an adjuster-calibration knob), not
regulator facts — override per call. Mirrors the auxima app's pure-engine convention
(Decimal 2dp ROUND_HALF_UP, ValueError on bad input); deliberately NOT an LLM step: an
initial reserve is a reproducible financial control, not a creative draft.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

_2DP = Decimal("0.01")

# Commercial defaults: uplift on the insured's first estimate by line volatility.
DEFAULT_RESERVE_FACTORS: dict[str, Decimal] = {
	"motor": Decimal("1.0"),
	"property": Decimal("1.1"),
	"medical": Decimal("1.2"),
	"liability": Decimal("1.5"),
	"marine": Decimal("1.15"),
	"engineering": Decimal("1.1"),
	"other": Decimal("1.25"),
}
MIN_RESERVE = Decimal("1000.00")  # floor so a 0-estimate FNOL still books a case reserve


@dataclass(frozen=True)
class ReserveSuggestion:
	suggested_reserve: Decimal
	basis: str


def suggest_reserve(
	estimated_amount: Decimal | str | int,
	loss_type: str,
	*,
	factors: dict[str, Decimal] | None = None,
	minimum: Decimal = MIN_RESERVE,
) -> ReserveSuggestion:
	"""max(estimate × line factor, floor), 2dp. Fail-closed on unknown line or negative input."""
	table = factors if factors is not None else DEFAULT_RESERVE_FACTORS
	if loss_type not in table:
		raise ValueError(f"no reserve factor for loss_type {loss_type!r} (fail-closed)")
	estimate = Decimal(str(estimated_amount))
	if estimate < 0:
		raise ValueError("estimated_amount cannot be negative")
	raw = (estimate * table[loss_type]).quantize(_2DP, rounding=ROUND_HALF_UP)
	if raw < minimum:
		return ReserveSuggestion(
			suggested_reserve=minimum,
			basis=f"floor {minimum} (estimate {estimate} × {loss_type} factor {table[loss_type]} below floor)",
		)
	return ReserveSuggestion(
		suggested_reserve=raw,
		basis=f"estimate {estimate} × {loss_type} factor {table[loss_type]}",
	)
