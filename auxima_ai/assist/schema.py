"""Pydantic models for ``POST /v1/assist/draft-email``.

Strict (``extra="forbid"``) so a wire typo is a loud 422. The request carries
the record context the draft is grounded in plus optional few-shot style
examples (the learning loop feeds the user's own past ``draft -> sent`` edits
back in here). The response is the drafted subject + body.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StyleExample(BaseModel):
	"""One past email the user actually sent — a few-shot style anchor.

	Supplied by the learning loop (slice 3). ``instruction`` is the purpose that
	produced it (optional), ``subject``/``body`` are what the user finally sent.
	"""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	instruction: str | None = Field(None, max_length=2000)
	subject: str = Field(..., min_length=1, max_length=300)
	body: str = Field(..., min_length=1, max_length=8000)


class DraftEmailRequest(BaseModel):
	"""Body of ``POST /v1/assist/draft-email``."""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	tenant_id: str = Field(..., min_length=1, max_length=128)
	purpose: str = Field(
		..., min_length=1, max_length=2000,
		description="What the broker wants the email to achieve (the instruction).",
	)
	recipient_name: str | None = Field(None, max_length=300)
	recipient_role: str | None = Field(None, max_length=200)
	company_name: str | None = Field(None, max_length=300)
	sender_name: str | None = Field(None, max_length=200)
	language: str = Field("en", pattern="^(en|ar)$")
	tone: str = Field("professional", max_length=60)
	examples: list[StyleExample] = Field(default_factory=list, max_length=8)
	model_id: str | None = Field(None, max_length=128, description="Optional model override.")


class DraftEmailFields(BaseModel):
	"""The shape the LLM must return — validated before we trust it."""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	subject: str = Field(..., min_length=1, max_length=300)
	body: str = Field(..., min_length=1, max_length=8000)


class DraftEmailResponse(BaseModel):
	"""Body of a successful (or degraded) draft response."""

	model_config = ConfigDict(extra="forbid")

	subject: str
	body: str
	language: str
	degraded: bool = False
	model_version: str = ""
	prompt_tokens: int = 0
	completion_tokens: int = 0
	latency_ms: int = 0


__all__ = ("DraftEmailFields", "DraftEmailRequest", "DraftEmailResponse", "StyleExample")
