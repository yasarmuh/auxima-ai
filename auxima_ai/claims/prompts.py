# Copyright (c) 2026, Auxilium Tech and contributors
"""Triage prompt + strict response validation for the ClaimsCrew (P3-01).

The triage prompt carries the FNOL narrative VERBATIM — it can contain health information
(special-category) and personal data, which is exactly why the service pins the call
``local_only=True`` unconditionally (see service.py). Validation mirrors the assist pattern:
Pydantic-strict, any deviation raises and the caller degrades to the heuristic.
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from auxima_ai.claims.schema import TriageAssessment


def build_triage_prompt(
	*, loss_type: str, incident_date: str, reported_date: str, description: str,
	estimated_amount: str, currency: str,
) -> str:
	return (
		"You are a senior insurance claims triager at a KSA broker.\n"
		"Assess the First Notice of Loss below. Reply with ONLY a JSON object of shape\n"
		'{"severity": "low|medium|high", "complexity": "fast_track|standard|complex",\n'
		' "fraud_indicators": ["..."]}\n'
		"fraud_indicators lists concrete red flags found in THIS notice (empty list if none).\n"
		"Do not invent facts. Do not add fields.\n\n"
		f"loss_type: {loss_type}\n"
		f"incident_date: {incident_date}\n"
		f"reported_date: {reported_date}\n"
		f"estimated_amount: {estimated_amount} {currency}\n"
		f"narrative: {description}\n"
	)


def validate_triage_response(payload: Any) -> TriageAssessment:
	"""Strict-validate the LLM payload; raise ValueError on any deviation (caller degrades)."""
	if not isinstance(payload, dict):
		raise ValueError(f"triage payload must be a JSON object; got {type(payload).__name__}")
	try:
		return TriageAssessment.model_validate({**payload, "source": "llm"})
	except ValidationError as e:
		raise ValueError(f"triage payload failed validation: {e.error_count()} errors") from e
