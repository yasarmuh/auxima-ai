# Copyright (c) 2026, Auxilium Tech and contributors
"""FNOL intake HTTP adapter — POST /v1/claims/fnol/turn (P3-01c)."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auxima_ai.claims.intake import FNOLIntakeService
from auxima_ai.claims.intake_router import (
	reset_fnol_intake_service,
	router,
	set_fnol_intake_service,
)
from auxima_ai.claims.schema import ClaimsProcessOutcome


class _StubClaims:
	def process(self, request):
		return ClaimsProcessOutcome(status="ok", claim_ref=request.claim_ref, audit_trail=["stub"])


def _app() -> TestClient:
	app = FastAPI()
	app.include_router(router)
	return TestClient(app)


def _body(**overrides) -> dict:
	body = {
		"tenant_id": "t-http", "session_id": "s-http", "channel": "web",
		"message": "A fire broke out at the warehouse", "today": "2026-07-02",
	}
	body.update(overrides)
	return body


def test_unconfigured_is_503():
	reset_fnol_intake_service()
	assert _app().post("/v1/claims/fnol/turn", json=_body()).status_code == 503


def test_collecting_turn_returns_bilingual_question():
	set_fnol_intake_service(FNOLIntakeService(claims=_StubClaims()))
	try:
		r = _app().post("/v1/claims/fnol/turn", json=_body())
		assert r.status_code == 200
		data = r.json()
		assert data["status"] == "collecting"
		assert data["next_question"]["en"] and data["next_question"]["ar"]
	finally:
		reset_fnol_intake_service()


def test_completed_turn_embeds_crew_outcome():
	set_fnol_intake_service(FNOLIntakeService(claims=_StubClaims()))
	try:
		r = _app().post("/v1/claims/fnol/turn", json=_body(
			fields={"loss_type": "property", "incident_date": "2026-06-28",
					"estimated_amount": "50000"},
		))
		assert r.status_code == 200
		data = r.json()
		assert data["status"] == "processed"
		assert data["outcome"]["claim_ref"] == "FNOL-s-http"
	finally:
		reset_fnol_intake_service()


def test_turn_cap_is_409():
	set_fnol_intake_service(FNOLIntakeService(claims=_StubClaims(), max_turns=1))
	try:
		client = _app()
		assert client.post("/v1/claims/fnol/turn", json=_body()).status_code == 200
		assert client.post("/v1/claims/fnol/turn", json=_body()).status_code == 409
	finally:
		reset_fnol_intake_service()


def test_empty_message_is_422():
	set_fnol_intake_service(FNOLIntakeService(claims=_StubClaims()))
	try:
		assert _app().post("/v1/claims/fnol/turn", json=_body(message="")).status_code == 422
	finally:
		reset_fnol_intake_service()
