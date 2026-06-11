# Copyright (c) 2026, Auxilium Tech and contributors
"""ClaimsCrew LangGraph state machine (P3-01) — FNOL → triage → reserve → line-routing.

The load-bearing facts: the graph walks validate→triage→reserve→route in order and records the
walk in an audit trail; FNOL validation FAILS CLOSED (rejected outcome, the LLM is never
called); the triage LLM step is ADVISORY and pinned ``local_only=True`` unconditionally (a
claim narrative can carry HEALTH data — special-category; the regex redactor has no NER); LLM
unavailability degrades to a deterministic heuristic (never a crash, flagged ``degraded``);
the reserve suggestion is pure Decimal and parameterised (commercial defaults, not regulator
facts); routing maps Policy.line to a P3-04 sub-crew label (skeleton).
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from auxima_ai.claims.reserve import DEFAULT_RESERVE_FACTORS, MIN_RESERVE, suggest_reserve
from auxima_ai.claims.schema import FNOLRequest
from auxima_ai.claims.service import ClaimsCrewService
from auxima_ai.intake.llm import LLMResponse


def _fnol(**overrides) -> FNOLRequest:
	base = {
		"tenant_id": "t-claims",
		"claim_ref": "CLM-T-001",
		"loss_type": "motor",
		"incident_date": "2026-06-01",
		"reported_date": "2026-06-03",
		"description": "Rear-end collision on King Fahd Road, bumper and boot damage.",
		"estimated_amount": "20000.00",
	}
	base.update(overrides)
	return FNOLRequest(**base)


TRIAGE_PAYLOAD = {"severity": "medium", "complexity": "standard", "fraud_indicators": []}


class RecordingInvoke:
	"""Stub for AssistService._invoke — records kwargs, returns a canned triage payload."""

	def __init__(self, payload: dict | None = None, fail: bool = False):
		self.payload = TRIAGE_PAYLOAD if payload is None else payload
		self.fail = fail
		self.calls: list[dict] = []

	def __call__(self, *, tenant_id: str, model_id: str, prompt: str, local_only: bool = False, **kw):
		self.calls.append({
			"tenant_id": tenant_id, "model_id": model_id, "prompt": prompt, "local_only": local_only,
		})
		if self.fail:
			raise RuntimeError("all providers down")
		return LLMResponse(payload=self.payload, prompt_tokens=10, completion_tokens=10, latency_ms=5)


# --- happy path -------------------------------------------------------------------------------


def test_happy_path_walks_all_stages_in_order():
	invoke = RecordingInvoke()
	svc = ClaimsCrewService(invoke=invoke)
	out = svc.process(_fnol())
	assert out.status == "ok"
	assert out.audit_trail == ["validate_fnol", "triage", "reserve_suggest", "route_line"]
	assert out.triage is not None and out.triage.severity == "medium"
	assert out.degraded is False


def test_reserve_uses_pure_engine_and_loss_factor():
	svc = ClaimsCrewService(invoke=RecordingInvoke())
	out = svc.process(_fnol(estimated_amount="20000.00", loss_type="motor"))
	expected = suggest_reserve(Decimal("20000.00"), "motor")
	assert out.reserve is not None
	assert Decimal(out.reserve.suggested_reserve) == expected.suggested_reserve
	assert out.reserve.basis  # the basis is explained, never a bare number


def test_route_line_maps_loss_type_to_subcrew():
	svc = ClaimsCrewService(invoke=RecordingInvoke())
	assert svc.process(_fnol(loss_type="motor")).subcrew == "motor"
	assert svc.process(_fnol(loss_type="medical")).subcrew == "medical"
	assert svc.process(_fnol(loss_type="marine")).subcrew == "specialty"


# --- the egress pin (PDPL/GDPR — health-bearing narrative) -------------------------------------


def test_triage_is_always_local_only():
	invoke = RecordingInvoke()
	svc = ClaimsCrewService(invoke=invoke)
	svc.process(_fnol())
	assert len(invoke.calls) == 1
	assert invoke.calls[0]["local_only"] is True


def test_medical_claim_also_local_only_and_health_flagged():
	invoke = RecordingInvoke()
	svc = ClaimsCrewService(invoke=invoke)
	out = svc.process(_fnol(loss_type="medical", description="ER admission after fall"))
	assert invoke.calls[0]["local_only"] is True
	assert out.health_data is True


# --- fail-closed validation (LLM never called) --------------------------------------------------


def test_reported_before_incident_rejected_no_llm():
	invoke = RecordingInvoke()
	svc = ClaimsCrewService(invoke=invoke)
	out = svc.process(_fnol(incident_date="2026-06-10", reported_date="2026-06-01"))
	assert out.status == "rejected"
	assert "incident" in (out.reason or "").lower()
	assert invoke.calls == []  # fail-closed BEFORE any LLM step


def test_blank_description_rejected():
	out = ClaimsCrewService(invoke=RecordingInvoke()).process(_fnol(description="   "))
	assert out.status == "rejected"


def test_negative_estimate_rejected_at_schema():
	with pytest.raises(Exception):
		_fnol(estimated_amount="-5")


# --- degrade-safe triage ------------------------------------------------------------------------


def test_llm_down_degrades_to_heuristic():
	svc = ClaimsCrewService(invoke=RecordingInvoke(fail=True))
	out = svc.process(_fnol(estimated_amount="900000.00"))
	assert out.status == "ok"
	assert out.degraded is True
	assert out.triage is not None and out.triage.severity == "high"  # amount heuristic
	assert out.triage.source == "heuristic"


def test_llm_garbage_degrades_not_crashes():
	svc = ClaimsCrewService(invoke=RecordingInvoke(payload={"text": "not the triage shape"}))
	out = svc.process(_fnol())
	assert out.status == "ok"
	assert out.degraded is True
	assert out.triage.source == "heuristic"


# --- pure reserve engine ------------------------------------------------------------------------


def test_reserve_minimum_floor():
	r = suggest_reserve(Decimal("0"), "motor")
	assert r.suggested_reserve == MIN_RESERVE


def test_reserve_factor_table_is_parameterised():
	r = suggest_reserve(Decimal("10000.00"), "liability",
	                    factors={"liability": Decimal("2.0")})
	assert r.suggested_reserve == Decimal("20000.00")


def test_reserve_unknown_loss_type_fails_closed():
	with pytest.raises(ValueError):
		suggest_reserve(Decimal("100"), "satellite")


def test_default_factors_cover_all_schema_loss_types():
	from auxima_ai.claims.schema import LOSS_TYPES
	assert set(LOSS_TYPES) <= set(DEFAULT_RESERVE_FACTORS)
