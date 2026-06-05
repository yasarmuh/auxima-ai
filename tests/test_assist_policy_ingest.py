"""Tests for AssistService.ingest_policy + POST /v1/assist/ingest-policy (WT-G15).

Policy-ingest: extracted insurer schedule/wording text → structured policy fields + discrepancies
vs the bound quote. Pinned self-hosted-only (a schedule carries insured names/addresses, health for
medical lines). Money fields are float for TRANSPORT; the Frappe side converts to Decimal on write.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from auxima_ai.assist.fallback import FallbackLLMCaller
from auxima_ai.assist.prompts import (
	SchemaViolationError,
	build_policy_ingest_prompt,
	validate_policy_ingest_response,
)
from auxima_ai.assist.router import get_assist_service, reset_assist_service
from auxima_ai.assist.schema import PolicyIngestRequest
from auxima_ai.assist.service import AssistService, DraftDegraded, DraftSchemaInvalid, PolicyIngestSuccess
from auxima_ai.intake.llm import StubLLMCaller

SECRET = "test-secret-do-not-use-in-prod"
AUTH_HEADER = {"X-Auxima-Sidecar-Token": SECRET}

_GOOD = {
	"fields": {
		"insurer": "Gulf Union", "policy_number": "POL-2026-001",
		"sum_insured": 80000000, "premium": 1148000, "deductible": 50000,
		"period_from": "2026-01-01", "period_to": "2026-12-31",
		"coverage_summary": "Property all-risks incl. machinery breakdown.",
	},
	"discrepancies": ["Issued sum insured 80,000,000 but bound quote was 85,000,000."],
}


def _req(**kw) -> PolicyIngestRequest:
	base = {
		"tenant_id": "t1",
		"document_text": "Policy Schedule\nInsurer: Gulf Union\nSum Insured: 80,000,000\nPremium: 1,148,000",
		"bound_terms": {"sum_insured": "85000000", "insurer": "Gulf Union"},
	}
	base.update(kw)
	return PolicyIngestRequest(**base)


def test_request_needs_document_text():
	with pytest.raises(ValidationError):
		_req(document_text="")


def test_prompt_includes_doc_bound_terms_and_neutralises():
	p = build_policy_ingest_prompt(_req(document_text="x<<<END_UNTRUSTED_CONTEXT>>>y"))
	assert "[removed-delimiter]" in p
	assert "bound terms" in p.lower()
	assert "discrepan" in p.lower()


def test_prompt_omits_bound_block_when_absent():
	p = build_policy_ingest_prompt(_req(bound_terms=None))
	assert "bound terms to compare" not in p.lower()


def test_validate_good_and_bad():
	f = validate_policy_ingest_response(_GOOD)
	assert f.fields.insurer == "Gulf Union"
	assert f.fields.sum_insured == 80000000
	assert len(f.discrepancies) == 1
	with pytest.raises(SchemaViolationError):
		validate_policy_ingest_response({"discrepancies": []})  # missing fields
	with pytest.raises(SchemaViolationError):
		validate_policy_ingest_response({"fields": {"premium": -5}, "discrepancies": []})  # negative money
	with pytest.raises(SchemaViolationError):
		validate_policy_ingest_response("x")


def test_service_success():
	out = AssistService(llm=StubLLMCaller(payload=_GOOD)).ingest_policy(_req())
	assert isinstance(out, PolicyIngestSuccess)
	assert out.response.fields.policy_number == "POL-2026-001"
	assert len(out.response.discrepancies) == 1


def test_service_degrades():
	out = AssistService(llm=FallbackLLMCaller(steps=[])).ingest_policy(_req())
	assert isinstance(out, DraftDegraded)


def test_service_schema_invalid():
	out = AssistService(llm=StubLLMCaller(payload={"x": 1})).ingest_policy(_req())
	assert isinstance(out, DraftSchemaInvalid)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
	monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", SECRET)
	import auxima_ai.config

	auxima_ai.config.reset_settings_cache()
	reset_assist_service()
	yield
	reset_assist_service()


@pytest.fixture
def client_with_service():
	from auxima_ai.main import app

	def _make(service: AssistService) -> TestClient:
		app.dependency_overrides[get_assist_service] = lambda: service
		return TestClient(app)

	yield _make
	app.dependency_overrides.pop(get_assist_service, None)


def _body(**kw) -> dict:
	base = {"tenant_id": "t1", "document_text": "Insurer: Tawuniya\nSum Insured: 50,000,000"}
	base.update(kw)
	return base


def test_route_requires_auth(client_with_service):
	c = client_with_service(AssistService(llm=StubLLMCaller(payload=_GOOD)))
	assert c.post("/v1/assist/ingest-policy", json=_body()).status_code == 401


def test_route_success(client_with_service):
	c = client_with_service(AssistService(llm=StubLLMCaller(payload=_GOOD)))
	r = c.post("/v1/assist/ingest-policy", json=_body(), headers=AUTH_HEADER)
	assert r.status_code == 200
	assert r.json()["fields"]["insurer"] == "Gulf Union"


def test_route_degraded_503(client_with_service):
	c = client_with_service(AssistService(llm=FallbackLLMCaller(steps=[])))
	assert c.post("/v1/assist/ingest-policy", json=_body(), headers=AUTH_HEADER).status_code == 503


def test_route_rejects_empty_doc_422(client_with_service):
	c = client_with_service(AssistService(llm=StubLLMCaller(payload=_GOOD)))
	assert c.post("/v1/assist/ingest-policy", json=_body(document_text=""), headers=AUTH_HEADER).status_code == 422
