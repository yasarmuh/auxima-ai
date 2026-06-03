"""Tests for the ``POST /v1/intake/extract-quote`` FastAPI route (P1-10)."""
from __future__ import annotations

import base64
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from auxima_ai.cost.ledger import InMemoryCostLedger
from auxima_ai.cost.pricing import reset_pricing_table
from auxima_ai.idempotency.store import InMemoryIdempotencyStore
from auxima_ai.intake.llm import StubLLMCaller
from auxima_ai.intake.pdf_text import StubPdfTextExtractor
from auxima_ai.intake.quote_router import get_quote_intake_service, reset_quote_intake_service
from auxima_ai.intake.quote_service import QuoteIntakeService
from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy
from auxima_ai.ratelimit.bucket import PerTenantRateLimiter

SECRET = "test-secret-do-not-use-in-prod"
AUTH_HEADER = {"X-Auxima-Sidecar-Token": SECRET}

_QUOTE_PAYLOAD = {
    "insurer_name": "Tawuniya",
    "premium": "12500.00",
    "currency": "SAR",
    "sum_insured": "1000000.00",
    "deductible": "500.00",
    "coverage": ["Own damage"],
    "exclusions": ["War"],
    "valid_until": "2026-12-31",
    "model_confidence": 0.95,
}


def _policy(*, tier=TierPolicy.OLLAMA_THEN_PAID_CLOUD):
    return TenantPolicy(
        tenant_id="tenant-acme", tier=tier, region="INTL",
        monthly_ceiling=Decimal("100"), rate_capacity=1000.0, rate_refill_per_second=100.0,
    )


def _build_service(*, policy=None, llm=None, pdf_extractor=None) -> QuoteIntakeService:
    enf = PolicyEnforcer(
        ledger=InMemoryCostLedger(),
        rate_limiter=PerTenantRateLimiter(capacity=1000.0, refill_per_second=100.0),
    )
    enf.set_policy(policy or _policy())
    return QuoteIntakeService(
        enforcer=enf,
        idempotency=InMemoryIdempotencyStore(),
        llm=llm or StubLLMCaller(payload=_QUOTE_PAYLOAD),
        pdf_extractor=pdf_extractor or StubPdfTextExtractor(text="Q" * 300),
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", SECRET)
    import auxima_ai.config

    auxima_ai.config.reset_settings_cache()
    reset_quote_intake_service()
    reset_pricing_table()
    yield
    reset_quote_intake_service()
    reset_pricing_table()


@pytest.fixture
def client_with_service():
    from auxima_ai.main import app

    def _make(service: QuoteIntakeService) -> TestClient:
        app.dependency_overrides[get_quote_intake_service] = lambda: service
        return TestClient(app)

    yield _make
    app.dependency_overrides.pop(get_quote_intake_service, None)


def _pdf_b64(body: bytes = b"quote body") -> str:
    return base64.b64encode(b"%PDF-1.4\n" + body + b"\n%%EOF\n").decode("ascii")


def _body(*, tenant="tenant-acme", document_b64=None, model_id="ollama/qwen2.5:32b") -> dict:
    return {
        "tenant_id": tenant,
        "document_b64": document_b64 if document_b64 is not None else _pdf_b64(),
        "model_id": model_id,
    }


def test_requires_auth_token(client_with_service) -> None:
    client = client_with_service(_build_service())
    r = client.post("/v1/intake/extract-quote", json=_body(),
                    headers={"Idempotency-Key": "q-noauth"})
    assert r.status_code == 401


def test_missing_idempotency_key_is_422(client_with_service) -> None:
    client = client_with_service(_build_service())
    r = client.post("/v1/intake/extract-quote", json=_body(), headers=AUTH_HEADER)
    assert r.status_code == 422


def test_unsanctioned_model_id_is_422_not_500(client_with_service) -> None:
    """H-1: a client-supplied model_id we don't run is a clean 422 at the edge — never a 500,
    and never reaches the enforcer/LLM. Auth + idempotency key are valid so only model_id is bad."""
    client = client_with_service(_build_service())
    r = client.post(
        "/v1/intake/extract-quote",
        json=_body(model_id="bogus/unsanctioned-model"),
        headers={**AUTH_HEADER, "Idempotency-Key": "q-badmodel"},
    )
    assert r.status_code == 422
    assert "model_id" in r.text and "sanctioned" in r.text


def test_success_200_with_confidence_and_decision(client_with_service) -> None:
    client = client_with_service(_build_service())
    r = client.post("/v1/intake/extract-quote", json=_body(),
                    headers={**AUTH_HEADER, "Idempotency-Key": "q-1"})
    assert r.status_code == 200
    data = r.json()
    assert data["fields"]["premium"] == "12500.00"
    assert data["decision"] == "auto_accept"
    assert data["confidence"] == pytest.approx(0.95)
    assert data["doc_class"] == "pdf_valid"


def test_replay_sets_header(client_with_service) -> None:
    client = client_with_service(_build_service())
    h = {**AUTH_HEADER, "Idempotency-Key": "q-rep"}
    client.post("/v1/intake/extract-quote", json=_body(), headers=h)
    r2 = client.post("/v1/intake/extract-quote", json=_body(), headers=h)
    assert r2.status_code == 200
    assert r2.headers.get("Idempotent-Replayed") == "true"


def test_corrupt_pdf_is_422_with_reason(client_with_service) -> None:
    client = client_with_service(_build_service())
    bad = base64.b64encode(b"%PDF-1.4\nno eof").decode("ascii")  # missing %%EOF
    r = client.post("/v1/intake/extract-quote", json=_body(document_b64=bad),
                    headers={**AUTH_HEADER, "Idempotency-Key": "q-cor"})
    assert r.status_code == 422
    assert r.json()["reason"] == "corrupt_document"


def test_provider_denied_is_403(client_with_service) -> None:
    client = client_with_service(_build_service(policy=_policy(tier=TierPolicy.OLLAMA_ONLY)))
    r = client.post("/v1/intake/extract-quote", json=_body(model_id="openai/gpt-4o-mini"),
                    headers={**AUTH_HEADER, "Idempotency-Key": "q-deny"})
    assert r.status_code == 403


def test_schema_invalid_is_502(client_with_service) -> None:
    # lead-shaped default stub payload doesn't match the quote schema
    client = client_with_service(_build_service(llm=StubLLMCaller()))
    r = client.post("/v1/intake/extract-quote", json=_body(),
                    headers={**AUTH_HEADER, "Idempotency-Key": "q-bad"})
    assert r.status_code == 502
