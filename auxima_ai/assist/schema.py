"""Pydantic models for ``POST /v1/assist/draft-email``.

Strict (``extra="forbid"``) so a wire typo is a loud 422. The request carries
the record context the draft is grounded in plus optional few-shot style
examples (the learning loop feeds the user's own past ``draft -> sent`` edits
back in here). The response is the drafted subject + body.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class DraftNoteRequest(BaseModel):
	"""Body of ``POST /v1/assist/draft-note`` — comments, error-help, general text.

	``kind`` picks the prompt framing:
	  - ``comment``     : a short internal note/comment on a CRM record.
	  - ``error_help``  : explain a blocked action + suggest concrete next steps.
	  - ``general``     : free-form short text from the instruction + context.
	``context`` is a flat str->str map of facts (UNTRUSTED — record/error data).
	"""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	tenant_id: str = Field(..., min_length=1, max_length=128)
	kind: str = Field("general", pattern="^(comment|error_help|general)$")
	instruction: str = Field(..., min_length=1, max_length=2000)
	context: dict[str, str] = Field(default_factory=dict)
	language: str = Field("en", pattern="^(en|ar)$")
	model_id: str | None = Field(None, max_length=128)

	@field_validator("context")
	@classmethod
	def _bound_context(cls, v: dict[str, str]) -> dict[str, str]:
		if len(v) > 40:
			raise ValueError("context has too many keys (max 40)")
		for key, val in v.items():
			if len(str(key)) > 80 or len(str(val)) > 4000:
				raise ValueError(f"context entry {key!r} too large")
		return v


class DraftNoteFields(BaseModel):
	"""The shape the LLM must return for a note."""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	text: str = Field(..., min_length=1, max_length=8000)


class DraftNoteResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	text: str
	kind: str
	language: str
	degraded: bool = False
	model_version: str = ""
	prompt_tokens: int = 0
	completion_tokens: int = 0
	latency_ms: int = 0


class FieldSpec(BaseModel):
	"""One EMPTY field we want a suggestion for."""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	fieldname: str = Field(..., min_length=1, max_length=140)
	label: str = Field("", max_length=200)
	fieldtype: str = Field("Data", max_length=40)


class SuggestFieldsRequest(BaseModel):
	"""Body of ``POST /v1/assist/suggest-fields``.

	``fields`` are the empty fields to suggest; ``current_values`` are the
	already-filled fields used as grounding context. The model is told to only
	suggest where it can reasonably infer and never to invent verifiable facts.
	"""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	tenant_id: str = Field(..., min_length=1, max_length=128)
	doctype: str = Field(..., min_length=1, max_length=140)
	fields: list[FieldSpec] = Field(..., min_length=1, max_length=40)
	current_values: dict[str, str] = Field(default_factory=dict)
	language: str = Field("en", pattern="^(en|ar)$")
	model_id: str | None = Field(None, max_length=128)

	@field_validator("current_values")
	@classmethod
	def _bound_values(cls, v: dict[str, str]) -> dict[str, str]:
		for key, val in v.items():
			if len(str(key)) > 140 or len(str(val)) > 4000:
				raise ValueError(f"current_values entry {key!r} too large")
		return v


class SuggestFieldsResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	suggestions: dict[str, str]
	degraded: bool = False
	model_version: str = ""
	prompt_tokens: int = 0
	completion_tokens: int = 0
	latency_ms: int = 0


# --- recommendation drafter (WT-G14, Article 9(b) note) -----------------------------------


class RecommendationCandidate(BaseModel):
	"""One insurer offer on the panel — the normalised comparison row (step 09) plus the
	optional due-diligence score (step 10 / WT-G09, absent until that engine exists)."""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	insurer: str = Field(..., min_length=1, max_length=200)
	premium: float | None = Field(None, ge=0, description="Net premium in `currency`.")
	currency: str = Field("SAR", max_length=8)
	dd_score: float | None = Field(None, ge=0, le=100, description="Due-diligence score 0-100 (optional).")
	dd_grade: str | None = Field(None, max_length=8)
	terms: list[str] = Field(default_factory=list, max_length=20, description="Key cover terms / flags.")


class RecommendationRequest(BaseModel):
	"""Body of ``POST /v1/assist/draft-recommendation``.

	``recommended_insurer`` must be one of ``candidates`` (the broker's pick, or the top DD score).
	``commission_pct`` is mandatory — it is disclosed in the generated note (Article 9(b)).
	"""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	tenant_id: str = Field(..., min_length=1, max_length=128)
	recommended_insurer: str = Field(..., min_length=1, max_length=200)
	candidates: list[RecommendationCandidate] = Field(..., min_length=1, max_length=12)
	client_needs: list[str] = Field(default_factory=list, max_length=20, description="Demands & needs signals (step 03).")
	commission_pct: float = Field(..., ge=0, le=100, description="Broker commission %, disclosed in the note.")
	language: str = Field("en", pattern="^(en|ar)$")
	retention_years: int = Field(10, ge=1, le=30, description="Record-retention years cited (Art. 24).")
	model_id: str | None = Field(None, max_length=128)

	@model_validator(mode="after")
	def _recommended_must_be_a_candidate(self) -> "RecommendationRequest":
		names = {c.insurer for c in self.candidates}
		if self.recommended_insurer not in names:
			raise ValueError(f"recommended_insurer {self.recommended_insurer!r} is not among the candidates")
		return self


class RecommendationFields(BaseModel):
	"""The shape the LLM must return — the reasoning + the citation bullets. The mandatory legal
	block is NOT generated by the model; the service appends it deterministically."""

	model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

	reasoning: str = Field(..., min_length=20, max_length=4000)
	citations: list[str] = Field(..., min_length=1, max_length=5)


class LegalCheck(BaseModel):
	"""Deterministic post-check on the assembled note — the WT-G14 'legal-line check'."""

	model_config = ConfigDict(extra="forbid")

	insurer_named: bool
	commission_disclosed: bool
	art9b_present: bool
	passed: bool


class RecommendationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	recommended_insurer: str
	body_md: str
	citations: list[str]
	legal_check: LegalCheck
	language: str
	degraded: bool = False
	model_version: str = ""
	prompt_tokens: int = 0
	completion_tokens: int = 0
	latency_ms: int = 0


__all__ = (
	"DraftEmailFields",
	"DraftEmailRequest",
	"DraftEmailResponse",
	"DraftNoteFields",
	"DraftNoteRequest",
	"DraftNoteResponse",
	"FieldSpec",
	"LegalCheck",
	"RecommendationCandidate",
	"RecommendationFields",
	"RecommendationRequest",
	"RecommendationResponse",
	"StyleExample",
	"SuggestFieldsRequest",
	"SuggestFieldsResponse",
)
