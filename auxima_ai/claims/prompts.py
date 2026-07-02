# Copyright (c) 2026, Auxilium Tech and contributors
"""Triage prompt + strict response validation for the ClaimsCrew (P3-01).

The triage prompt carries the FNOL narrative VERBATIM — it can contain health information
(special-category) and personal data, which is exactly why the service pins the call
``local_only=True`` unconditionally (see service.py). Validation mirrors the assist pattern:
Pydantic-strict, any deviation raises and the caller degrades to the heuristic.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from auxima_ai.claims.schema import LOSS_TYPES, TriageAssessment


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


def build_fnol_extract_prompt(*, message: str, missing: list[str], today: str) -> str:
	"""Extraction prompt for the multi-turn intake (P3-01c). Carries the reporter's message
	VERBATIM — may contain health data/PII, which is why the intake pins local_only=True."""
	return (
		"You extract structured fields from an insurance First-Notice-of-Loss message\n"
		"(English or Arabic). Reply with ONLY a JSON object containing AT MOST these keys,\n"
		f"and only when the message clearly states them: {', '.join(missing)}.\n"
		'Shapes: "loss_type" one of motor|property|medical|liability|marine|engineering|other;\n'
		'"incident_date" ISO yyyy-mm-dd (resolve relative words against today given below);\n'
		'"estimated_amount" a plain decimal number in SAR.\n'
		"Omit any field the message does not state. Do not invent values. Do not add fields.\n\n"
		f"today: {today}\n"
		f"message: {message}\n"
	)


def validate_extract_payload(payload: Any, *, today: date) -> dict[str, str]:
	"""Per-field validation of the extract payload — invalid values are DROPPED, not fatal
	(a hallucinated date must not kill a correctly extracted loss_type). Non-dict raises."""
	if not isinstance(payload, dict):
		raise ValueError(f"extract payload must be a JSON object; got {type(payload).__name__}")
	valid: dict[str, str] = {}
	if payload.get("loss_type") in LOSS_TYPES:
		valid["loss_type"] = payload["loss_type"]
	if isinstance(payload.get("incident_date"), str):
		try:
			if date.fromisoformat(payload["incident_date"]) <= today:
				valid["incident_date"] = payload["incident_date"]
		except ValueError:
			pass  # dropped: not a date
	if isinstance(payload.get("estimated_amount"), (str, int, float)):
		try:
			amount = Decimal(str(payload["estimated_amount"]))
			if amount >= 0:
				# store the validated Decimal's string, never the raw payload object —
				# a JSON float would smuggle float-repr artifacts into a money field
				valid["estimated_amount"] = str(amount)
		except InvalidOperation:
			pass  # dropped: not a decimal
	return valid
