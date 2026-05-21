"""Tests for AssistService.draft_email + the POST /v1/assist/draft-email route."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from auxima_ai.assist.fallback import FallbackLLMCaller
from auxima_ai.assist.router import get_assist_service, reset_assist_service
from auxima_ai.assist.service import (
	AssistService,
	DraftDegraded,
	DraftEmailSuccess,
	DraftSchemaInvalid,
)
from auxima_ai.assist.schema import DraftEmailRequest
from auxima_ai.intake.llm import StubLLMCaller

SECRET = "test-secret-do-not-use-in-prod"
AUTH_HEADER = {"X-Auxima-Sidecar-Token": SECRET}


def _req(**kw) -> DraftEmailRequest:
	base = {"tenant_id": "t1", "purpose": "introduce our motor fleet cover", "recipient_name": "Baqar"}
	base.update(kw)
	return DraftEmailRequest(**base)


# --- service level ---------------------------------------------------------


def test_service_success():
	svc = AssistService(llm=StubLLMCaller(payload={"subject": "Motor fleet cover", "body": "Dear Baqar, ... Regards."}))
	out = svc.draft_email(_req())
	assert isinstance(out, DraftEmailSuccess)
	assert out.response.subject == "Motor fleet cover"
	assert out.response.degraded is False


def test_service_degrades_when_all_providers_unavailable():
	# An empty fallback chain raises AllProvidersUnavailable -> graceful degrade.
	svc = AssistService(llm=FallbackLLMCaller(steps=[]))
	out = svc.draft_email(_req())
	assert isinstance(out, DraftDegraded)


def test_service_schema_invalid_when_model_returns_wrong_shape():
	svc = AssistService(llm=StubLLMCaller(payload={"headline": "oops"}))
	out = svc.draft_email(_req())
	assert isinstance(out, DraftSchemaInvalid)


def test_service_threads_language_through():
	svc = AssistService(llm=StubLLMCaller(payload={"subject": "مرحبا", "body": "نص الرسالة"}))
	out = svc.draft_email(_req(language="ar"))
	assert isinstance(out, DraftEmailSuccess)
	assert out.response.language == "ar"


# --- route level -----------------------------------------------------------


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
	base = {"tenant_id": "t1", "purpose": "introduce motor fleet cover", "recipient_name": "Baqar"}
	base.update(kw)
	return base


def test_route_requires_auth(client_with_service):
	client = client_with_service(AssistService(llm=StubLLMCaller(payload={"subject": "s", "body": "b"})))
	r = client.post("/v1/assist/draft-email", json=_body())  # no auth header
	assert r.status_code == 401


def test_route_success_200(client_with_service):
	svc = AssistService(llm=StubLLMCaller(payload={"subject": "Fleet cover", "body": "Dear Baqar..."}))
	client = client_with_service(svc)
	r = client.post("/v1/assist/draft-email", json=_body(), headers=AUTH_HEADER)
	assert r.status_code == 200
	data = r.json()
	assert data["subject"] == "Fleet cover"
	assert data["degraded"] is False


def test_route_degraded_503(client_with_service):
	svc = AssistService(llm=FallbackLLMCaller(steps=[]))
	client = client_with_service(svc)
	r = client.post("/v1/assist/draft-email", json=_body(), headers=AUTH_HEADER)
	assert r.status_code == 503
	assert r.json()["degraded"] is True
	assert r.headers.get("Retry-After") == "30"


def test_route_schema_invalid_502(client_with_service):
	svc = AssistService(llm=StubLLMCaller(payload={"nope": 1}))
	client = client_with_service(svc)
	r = client.post("/v1/assist/draft-email", json=_body(), headers=AUTH_HEADER)
	assert r.status_code == 502


def test_route_rejects_unknown_field_422(client_with_service):
	svc = AssistService(llm=StubLLMCaller(payload={"subject": "s", "body": "b"}))
	client = client_with_service(svc)
	r = client.post("/v1/assist/draft-email", json=_body(bogus="x"), headers=AUTH_HEADER)
	assert r.status_code == 422
