"""Tests for AssistService.suggest_fields + POST /v1/assist/suggest-fields."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from auxima_ai.assist.fallback import FallbackLLMCaller
from auxima_ai.assist.prompts import build_suggest_fields_prompt, validate_suggest_fields_response
from auxima_ai.assist.router import get_assist_service, reset_assist_service
from auxima_ai.assist.schema import SuggestFieldsRequest
from auxima_ai.assist.service import AssistService, DraftDegraded, SuggestFieldsSuccess
from auxima_ai.intake.llm import StubLLMCaller

SECRET = "test-secret-do-not-use-in-prod"
AUTH_HEADER = {"X-Auxima-Sidecar-Token": SECRET}


def _req(**kw) -> SuggestFieldsRequest:
	base = {
		"tenant_id": "t1",
		"doctype": "Customer",
		"fields": [{"fieldname": "industry", "label": "Industry"}, {"fieldname": "notes", "label": "Notes"}],
		"current_values": {"customer_name_en": "Riyadh Logistics Co"},
	}
	base.update(kw)
	return SuggestFieldsRequest(**base)


# --- validation (the safety layer) ----------------------------------------


def test_validate_keeps_only_allowed_fields():
	out = validate_suggest_fields_response(
		{"suggestions": {"industry": "Logistics", "cr_number": "1010123456", "notes": "  "}},
		allowed={"industry", "notes"},
	)
	# cr_number dropped (not requested — model must not fill verifiable facts);
	# notes dropped (empty after strip); industry kept.
	assert out == {"industry": "Logistics"}


def test_validate_rejects_missing_suggestions():
	import pytest as _pytest

	from auxima_ai.assist.prompts import SchemaViolationError

	with _pytest.raises(SchemaViolationError):
		validate_suggest_fields_response({"nope": {}}, allowed={"industry"})


def test_prompt_lists_empty_fields_and_warns_against_invention():
	p = build_suggest_fields_prompt(_req())
	assert "industry" in p and "notes" in p
	assert "never invent verifiable facts" in p.lower()


# --- service --------------------------------------------------------------


def test_service_success_filters_to_requested():
	svc = AssistService(llm=StubLLMCaller(payload={"suggestions": {"industry": "Logistics", "bogus": "x"}}))
	out = svc.suggest_fields(_req())
	assert isinstance(out, SuggestFieldsSuccess)
	assert out.response.suggestions == {"industry": "Logistics"}


def test_service_empty_suggestions_is_success():
	# The model inferring nothing is a valid, non-degraded outcome.
	svc = AssistService(llm=StubLLMCaller(payload={"suggestions": {}}))
	out = svc.suggest_fields(_req())
	assert isinstance(out, SuggestFieldsSuccess)
	assert out.response.suggestions == {}


def test_service_degrades():
	out = AssistService(llm=FallbackLLMCaller(steps=[])).suggest_fields(_req())
	assert isinstance(out, DraftDegraded)


# --- route ----------------------------------------------------------------


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
	base = {
		"tenant_id": "t1",
		"doctype": "Customer",
		"fields": [{"fieldname": "industry", "label": "Industry"}],
		"current_values": {"customer_name_en": "Riyadh Logistics Co"},
	}
	base.update(kw)
	return base


def test_route_requires_auth(client_with_service):
	client = client_with_service(AssistService(llm=StubLLMCaller(payload={"suggestions": {}})))
	assert client.post("/v1/assist/suggest-fields", json=_body()).status_code == 401


def test_route_success(client_with_service):
	svc = AssistService(llm=StubLLMCaller(payload={"suggestions": {"industry": "Logistics"}}))
	client = client_with_service(svc)
	r = client.post("/v1/assist/suggest-fields", json=_body(), headers=AUTH_HEADER)
	assert r.status_code == 200
	assert r.json()["suggestions"] == {"industry": "Logistics"}


def test_route_rejects_empty_fields_422(client_with_service):
	svc = AssistService(llm=StubLLMCaller(payload={"suggestions": {}}))
	client = client_with_service(svc)
	r = client.post("/v1/assist/suggest-fields", json=_body(fields=[]), headers=AUTH_HEADER)
	assert r.status_code == 422
