# Copyright (c) 2026, Auxilium Tech and contributors
"""Wire schemas for the multi-turn FNOL intake (P3-01c).

The turn request is what a channel adapter (portal web form, WhatsApp bridge, phone
transcriber) posts per reporter message; the outcome tells it what to ask next (bilingual)
or carries the ClaimsCrew verdict once the FNOL is complete. ``message`` may carry health
narrative and PII — it is echoed into prompts only via local-only LLM calls and is NEVER
echoed back in the outcome (``collected`` holds structured fields only).
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from auxima_ai.claims.schema import ClaimsProcessOutcome

MAX_MESSAGE_CHARS = 4000


def _validate_iso_date(v: str) -> str:
	from datetime import date

	date.fromisoformat(v)  # raises ValueError on garbage → pydantic 422 at the boundary
	return v


class ProvidedFields(BaseModel):
	"""Structured field answers a channel adapter already knows (portal form inputs)."""

	loss_type: (
		Literal["motor", "property", "medical", "liability", "marine", "engineering", "other"]
		| None
	) = None
	incident_date: str | None = None
	estimated_amount: str | None = None

	@field_validator("incident_date")
	@classmethod
	def _iso(cls, v: str | None) -> str | None:
		return None if v is None else _validate_iso_date(v)

	@field_validator("estimated_amount")
	@classmethod
	def _non_negative_decimal(cls, v: str | None) -> str | None:
		if v is None:
			return None
		try:
			amount = Decimal(v)
		except InvalidOperation as e:
			raise ValueError(f"estimated_amount is not a decimal: {v!r}") from e
		if amount < 0:
			raise ValueError("estimated_amount cannot be negative")
		return v


class FNOLTurnRequest(BaseModel):
	"""One reporter message in an intake session (session_id = the channel's thread id)."""

	tenant_id: str = Field(min_length=1)
	session_id: str = Field(min_length=1, max_length=120)
	channel: Literal["web", "whatsapp", "phone"]
	message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
	fields: ProvidedFields | None = None
	today: str | None = Field(
		default=None, description="ISO date of the turn (tests/replays); server date when absent"
	)

	@field_validator("today")
	@classmethod
	def _iso(cls, v: str | None) -> str | None:
		return None if v is None else _validate_iso_date(v)


class FNOLTurnOutcome(BaseModel):
	"""What the channel adapter does next: ask ``next_question`` (bilingual) while
	``collecting``, or surface the embedded crew ``outcome`` once ``processed``."""

	session_id: str
	status: Literal["collecting", "processed"]
	missing: list[str] = Field(default_factory=list)
	next_question: dict[str, str] | None = None
	collected: dict[str, str | None] = Field(default_factory=dict)
	warnings: list[str] = Field(default_factory=list)
	turns: int
	degraded: bool = False
	outcome: ClaimsProcessOutcome | None = None
