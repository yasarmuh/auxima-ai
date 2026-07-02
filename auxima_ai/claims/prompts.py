# Copyright (c) 2026, Auxilium Tech and contributors
"""Triage prompt + strict response validation for the ClaimsCrew (P3-01).

The triage prompt carries the FNOL narrative VERBATIM — it can contain health information
(special-category) and personal data, which is exactly why the service pins the call
``local_only=True`` unconditionally (see service.py). Validation mirrors the assist pattern:
Pydantic-strict, any deviation raises and the caller degrades to the heuristic.
"""
from __future__ import annotations

from datetime import date
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
	VERBATIM — may contain health data/PII, which is why the intake pins local_only=True.

	Hardened for small models (0.5B-tier benchmark, 2026-07-02): the explicit return-{} rule
	plus one-shot examples stopped qwen2.5:0.5b from inventing fields on an empty message —
	without them every sub-1B candidate padded all three fields with plausible values.
	Shape rules are scoped to the fields actually asked for (money is never one of them —
	amounts are deterministic-extractor territory)."""
	shapes = {
		"loss_type": '- "loss_type": one of motor|property|medical|liability|marine|'
		"engineering|other\n",
		"incident_date": '- "incident_date": ISO yyyy-mm-dd; resolve relative words like '
		'"yesterday" against today\n',
	}
	return (
		"You extract fields from an insurance First-Notice-of-Loss message (English or "
		"Arabic).\n"
		"Reply with ONLY a JSON object. Rules:\n"
		f"- Allowed keys (include a key ONLY if the message clearly states it): "
		f"{', '.join(missing)}\n"
		+ "".join(shapes[f] for f in missing if f in shapes)
		+ "- If the message does not clearly state a field, DO NOT include that key.\n"
		"- If nothing is clearly stated, reply with exactly {}\n"
		"- Never guess. An empty object is a correct answer.\n\n"
		"Example 1 message: 'my car was hit yesterday'\n"
		'Example 1 reply: {"loss_type": "motor", "incident_date": "<resolved yesterday>"}\n'
		"Example 2 message: 'something happened, details later'\n"
		"Example 2 reply: {}\n\n"
		f"today: {today}\n"
		f"message: {message}\n"
	)


def validate_extract_payload(payload: Any, *, today: date) -> dict[str, str]:
	"""Per-field validation of the extract payload — invalid values are DROPPED, not fatal
	(a hallucinated date must not kill a correctly extracted loss_type). Non-dict raises.

	Padding guards (small-model benchmark, 2026-07-02): an LLM "other" loss type and an LLM
	incident_date equal to TODAY are dropped as indistinguishable-from-padding — the
	deterministic extractor already catches a reporter literally answering "other"/"أخرى"
	or saying "today"/"اليوم" BEFORE the LLM is asked, so from the LLM these values are
	almost surely invented. Money is dropped UNCONDITIONALLY (live: a 0.5B garbled "20
	thousand riyals" into 200000 — magnitude-wrong money passes shape checks; amounts are
	deterministic-extractor territory). Cost of a false drop = one re-ask; cost of a false
	accept = a fabricated fact in the FNOL."""
	if not isinstance(payload, dict):
		raise ValueError(f"extract payload must be a JSON object; got {type(payload).__name__}")
	valid: dict[str, str] = {}
	if payload.get("loss_type") in LOSS_TYPES and payload["loss_type"] != "other":
		valid["loss_type"] = payload["loss_type"]
	if isinstance(payload.get("incident_date"), str):
		try:
			if date.fromisoformat(payload["incident_date"]) < today:
				valid["incident_date"] = payload["incident_date"]
		except ValueError:
			pass  # dropped: not a date
	return valid
