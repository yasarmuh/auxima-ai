# Copyright (c) 2026, Auxilium Tech and contributors
"""ClaimsCrew wire schemas (P3-01) — FNOL in, advisory triage/reserve/routing out.

Advisory-only by design: the crew RECOMMENDS (severity, initial reserve, sub-crew route); it
never writes a Frappe record — the broker accepts in the Desk/portal and the auxima app's own
fail-closed Claim controller does the bookkeeping (CLAUDE.md §4; advisory-AI-not-into-
immutable-artefacts). Money rides as str-encoded Decimal, never float.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, Field, field_validator

LOSS_TYPES = ("motor", "property", "medical", "liability", "marine", "engineering", "other")

Severity = Literal["low", "medium", "high"]
Complexity = Literal["fast_track", "standard", "complex"]


class FNOLRequest(BaseModel):
	"""First Notice of Loss as the Frappe app reports it (REST; no frappe import)."""

	tenant_id: str = Field(min_length=1)
	claim_ref: str = Field(min_length=1, description="The auxima Claim name (CLM-…)")
	loss_type: Literal["motor", "property", "medical", "liability", "marine", "engineering", "other"]
	incident_date: str = Field(description="ISO date")
	reported_date: str = Field(description="ISO date")
	description: str = Field(description="FNOL narrative — may carry health/PII; never clouded")
	estimated_amount: str = Field(default="0", description="Decimal string, >= 0")
	policy_ref: str | None = None
	currency: str = "SAR"

	@field_validator("estimated_amount")
	@classmethod
	def _non_negative_decimal(cls, v: str) -> str:
		try:
			amount = Decimal(v)
		except InvalidOperation as e:
			raise ValueError(f"estimated_amount is not a decimal: {v!r}") from e
		if amount < 0:
			raise ValueError("estimated_amount cannot be negative")
		return v


class TriageAssessment(BaseModel):
	"""Advisory triage of the FNOL — LLM-drafted (local-only) or heuristic fallback."""

	severity: Severity
	complexity: Complexity
	fraud_indicators: list[str] = Field(default_factory=list)
	source: Literal["llm", "heuristic"] = "llm"


class ReserveSuggestionOut(BaseModel):
	"""Deterministic initial-reserve recommendation (pure Decimal engine, parameterised)."""

	suggested_reserve: str
	basis: str


class ClaimsProcessOutcome(BaseModel):
	"""The crew's verdict. status=rejected means FNOL validation failed CLOSED (no LLM ran)."""

	status: Literal["ok", "rejected"]
	claim_ref: str
	audit_trail: list[str] = Field(default_factory=list)
	triage: TriageAssessment | None = None
	reserve: ReserveSuggestionOut | None = None
	subcrew: str | None = None
	health_data: bool = False
	degraded: bool = False
	reason: str | None = None
