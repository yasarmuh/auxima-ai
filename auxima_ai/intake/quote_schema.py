"""Wire models for ``POST /v1/intake/extract-quote`` (P1-10).

The quote-intake endpoint takes a base64-encoded insurer-quote PDF and returns
the extracted premium / coverage / exclusions plus a length-aware confidence
score and an auto-accept/hold decision (see ``quote_service``).

Money is **Decimal, never float** (CLAUDE.md §5 / acceptance §5.2) — serialised
as a JSON string on the wire (JSON has no native Decimal), the same convention
``IntakeResponse.cost`` already uses. Strict ``extra="forbid"`` on both models so
a typo in the body is a loud 422, never silently dropped.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from auxima_ai.cost.pricing import known_models

#: ~25 MB raw PDF (the §4.2 / S-46 cap) is ~34 MB base64. Cap the encoded string
#: a little above that so an oversized upload is rejected at the edge before the
#: document classifier even runs.
_MAX_DOCUMENT_B64_CHARS = 35_000_000


class QuoteIntakeRequest(BaseModel):
    """Body of ``POST /v1/intake/extract-quote``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    document_b64: str = Field(..., min_length=1, max_length=_MAX_DOCUMENT_B64_CHARS)
    model_id: str = Field(default="ollama/qwen2.5:32b", min_length=1)

    @field_validator("model_id")
    @classmethod
    def _model_id_is_sanctioned(cls, v: str) -> str:
        """Reject an unsanctioned model_id at the wire edge (P1-10 H-1, defense-in-depth).

        ``model_id`` is client-supplied, so it is a client error — not a server fault — to
        pass one we do not run. The allow-list is the pricing table's :func:`known_models`
        (the single source of truth). Without this, an unknown id raises UnknownModelError
        deep in the enforcer and surfaces as a 500; here it is a clean 422 before any
        document download/classification. The residency enforcer (ADR-GA2/GA3) remains the
        backstop that governs WHICH sanctioned model a given tenant may egress to.
        """
        allowed = known_models()
        if v not in allowed:
            raise ValueError(
                f"model_id {v!r} is not a sanctioned model; known: {sorted(allowed)}"
            )
        return v


class QuoteIntakeResponse(BaseModel):
    """Body of a successful quote extraction."""

    model_config = ConfigDict(extra="forbid")

    activity_id: str
    model_id: str
    provider: str
    fields: dict[str, Any]  # the extracted quote fields (sans model_confidence)
    confidence: float  # final, length-aware confidence in [0, 1]
    decision: str  # "auto_accept" | "hold_for_review"
    doc_class: str  # DocumentClass value (e.g. "pdf_valid")
    text_source: str  # TextSource value ("native" | "ocr")
    page_count: int
    char_count: int
    model_version: str
    cost: str  # Decimal serialised as string
    period_total: str
    redaction_applied: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


__all__ = ("QuoteIntakeRequest", "QuoteIntakeResponse")
