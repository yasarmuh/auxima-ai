"""Pydantic models for the intake.extract request + response.

Strict mode (``extra="forbid"``) on both request and response so a
typo in the wire body is a loud 422 — never silently dropped.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IntakeRequest(BaseModel):
    """Body of ``POST /v1/intake/extract``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    lead_text: str = Field(..., min_length=1, max_length=50_000)
    model_id: str = Field(default="ollama/qwen2.5:32b", min_length=1)


class IntakeResponse(BaseModel):
    """Body of a successful response."""

    model_config = ConfigDict(extra="forbid")

    activity_id: str
    model_id: str
    provider: str
    fields: dict[str, Any]
    cost: str  # Decimal serialised as string (JSON has no native Decimal)
    period_total: str
    redaction_applied: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


__all__ = ("IntakeRequest", "IntakeResponse")
