# Copyright (c) 2026, Auxilium Tech and contributors
"""ClaimsCrew HTTP adapter — outcome→status mapping + unconfigured fail-closed (P3-01)."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auxima_ai.claims.router import (
	reset_claims_service,
	router,
	set_claims_service,
)
from auxima_ai.claims.service import ClaimsCrewService
from auxima_ai.intake.llm import LLMResponse

TRIAGE = {"severity": "low", "complexity": "fast_track", "fraud_indicators": []}


def _stub_invoke(**kw):
	return LLMResponse(payload=TRIAGE, prompt_tokens=1, completion_tokens=1, latency_ms=1)


def _app() -> TestClient:
	app = FastAPI()
	app.include_router(router)
	return TestClient(app)


def _fnol_body(**overrides) -> dict:
	body = {
		"tenant_id": "t-router", "claim_ref": "CLM-R-1", "loss_type": "property",
		"incident_date": "2026-06-01", "reported_date": "2026-06-02",
		"description": "Warehouse fire, partial stock loss", "estimated_amount": "75000",
	}
	body.update(overrides)
	return body


def test_process_returns_recommendations():
	set_claims_service(ClaimsCrewService(invoke=_stub_invoke))
	try:
		r = _app().post("/v1/claims/process", json=_fnol_body())
		assert r.status_code == 200
		data = r.json()
		assert data["status"] == "ok"
		assert data["subcrew"] == "property"
		assert data["reserve"]["suggested_reserve"] == "82500.00"  # 75000 x 1.1
		assert data["audit_trail"] == [
			"validate_fnol", "triage", "reserve_suggest", "route_line", "subcrew_actions",
		]
	finally:
		reset_claims_service()


def test_rejected_fnol_is_422():
	set_claims_service(ClaimsCrewService(invoke=_stub_invoke))
	try:
		r = _app().post("/v1/claims/process", json=_fnol_body(reported_date="2026-05-01"))
		assert r.status_code == 422
		assert "incident" in str(r.json()["detail"]["reason"]).lower()
	finally:
		reset_claims_service()


def test_unconfigured_crew_is_503_fail_closed():
	reset_claims_service()
	r = _app().post("/v1/claims/process", json=_fnol_body())
	assert r.status_code == 503


def test_schema_invalid_body_is_422():
	set_claims_service(ClaimsCrewService(invoke=_stub_invoke))
	try:
		r = _app().post("/v1/claims/process", json=_fnol_body(estimated_amount="-1"))
		assert r.status_code == 422
	finally:
		reset_claims_service()
