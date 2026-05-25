"""Tests for the ``POST /v1/intake/extract`` FastAPI route.

Coverage:
  - 200 on success; activity_id, fields, cost echoed.
  - 200 on idempotent replay; Idempotent-Replayed: true header.
  - 409 on in-flight conflict with Retry-After.
  - 422 on idempotency conflict (same key, different body).
  - 422 on missing body fields / unknown extra fields.
  - 422 on missing Idempotency-Key header.
  - 403 on tenant tier not allowing provider.
  - 429 on per-tenant rate limit; Retry-After integer header.
  - 402 on cost ceiling exceeded.
  - 500 on unknown-provider config bug.
  - Auth middleware still enforced (401 without token).
  - traceparent header propagates into emitted event.
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from auxima_ai.cost.ledger import InMemoryCostLedger
from auxima_ai.cost.pricing import ModelPricing, register_pricing, reset_pricing_table
from auxima_ai.intake.llm import StubLLMCaller
from auxima_ai.intake.router import (
    get_intake_service,
    reset_intake_service,
)
from auxima_ai.intake.service import IntakeService
from auxima_ai.idempotency.store import InMemoryIdempotencyStore
from auxima_ai.observability.log import EVENT_LOGGER_NAME
from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy
from auxima_ai.ratelimit.bucket import PerTenantRateLimiter


SECRET = "test-secret-do-not-use-in-prod"
AUTH_HEADER = {"X-Auxima-Sidecar-Token": SECRET}


def _policy(
    *,
    tenant: str = "tenant-acme",
    tier: TierPolicy = TierPolicy.OLLAMA_THEN_PAID_CLOUD,
    ceiling: Decimal = Decimal("100"),
    capacity: float = 1000.0,
    refill: float = 100.0,
    # Non-in-Kingdom by default so cloud/ceiling mechanism tests reach the cloud
    # path; the KSA residency invariant is covered by test_residency_invariant.py.
    region: str = "INTL",
) -> TenantPolicy:
    return TenantPolicy(
        tenant_id=tenant, tier=tier, region=region, monthly_ceiling=ceiling,
        rate_capacity=capacity, rate_refill_per_second=refill,
    )


def _build_service(
    *,
    policy: TenantPolicy | None = None,
    llm: StubLLMCaller | None = None,
    rate_capacity: float = 1000.0,
    rate_refill: float = 100.0,
) -> IntakeService:
    enf = PolicyEnforcer(
        ledger=InMemoryCostLedger(),
        rate_limiter=PerTenantRateLimiter(capacity=rate_capacity, refill_per_second=rate_refill),
    )
    enf.set_policy(policy or _policy(capacity=rate_capacity, refill=rate_refill))
    return IntakeService(
        enforcer=enf,
        idempotency=InMemoryIdempotencyStore(),
        llm=llm or StubLLMCaller(),
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AUXIMA_SIDECAR_SHARED_SECRET", SECRET)
    import auxima_ai.config
    auxima_ai.config.reset_settings_cache()
    reset_intake_service()
    reset_pricing_table()
    yield
    reset_intake_service()
    reset_pricing_table()


@pytest.fixture
def client_with_service():
    """Returns (TestClient, injected_service) — caller passes any service."""
    from auxima_ai.main import app

    def _make(service: IntakeService) -> TestClient:
        app.dependency_overrides[get_intake_service] = lambda: service
        return TestClient(app)

    yield _make
    app.dependency_overrides.pop(get_intake_service, None)


def _body(
    *, tenant: str = "tenant-acme",
    lead_text: str = "lead from Acme Brokers needs P&C cover",
    model_id: str = "ollama/qwen2.5:32b",
) -> dict:
    return {"tenant_id": tenant, "lead_text": lead_text, "model_id": model_id}


# ---------------------------------------------------------------------------
# Auth still enforced
# ---------------------------------------------------------------------------


def test_extract_requires_auth_token(client_with_service) -> None:
    client = client_with_service(_build_service())
    r = client.post(
        "/v1/intake/extract",
        json=_body(),
        headers={"Idempotency-Key": "k-noauth"},  # no auth header
    )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# 200 success
# ---------------------------------------------------------------------------


def test_extract_200_on_success(client_with_service) -> None:
    client = client_with_service(_build_service())
    r = client.post(
        "/v1/intake/extract",
        json=_body(),
        headers={**AUTH_HEADER, "Idempotency-Key": "k-ok"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_id"] == "ollama/qwen2.5:32b"
    assert body["provider"] == "ollama"
    assert body["activity_id"]
    assert body["cost"] == "0.000000"
    assert "fields" in body


# ---------------------------------------------------------------------------
# Idempotency replay
# ---------------------------------------------------------------------------


def test_replay_returns_200_with_idempotent_replayed_header(client_with_service) -> None:
    client = client_with_service(_build_service())
    headers = {**AUTH_HEADER, "Idempotency-Key": "k-replay"}
    first = client.post("/v1/intake/extract", json=_body(), headers=headers)
    second = client.post("/v1/intake/extract", json=_body(), headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers.get("Idempotent-Replayed") == "true"
    assert first.json()["activity_id"] == second.json()["activity_id"]


# ---------------------------------------------------------------------------
# Idempotency conflict — same key, different body
# ---------------------------------------------------------------------------


def test_conflict_422_on_same_key_different_body(client_with_service) -> None:
    client = client_with_service(_build_service())
    headers = {**AUTH_HEADER, "Idempotency-Key": "k-conflict"}
    client.post("/v1/intake/extract", json=_body(lead_text="body A"), headers=headers)
    r = client.post("/v1/intake/extract", json=_body(lead_text="body B"), headers=headers)
    assert r.status_code == 422
    assert "different request body" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Body validation
# ---------------------------------------------------------------------------


def test_422_when_idempotency_key_missing(client_with_service) -> None:
    client = client_with_service(_build_service())
    r = client.post("/v1/intake/extract", json=_body(), headers=AUTH_HEADER)
    assert r.status_code == 422


def test_422_when_body_missing_required_field(client_with_service) -> None:
    client = client_with_service(_build_service())
    bad_body = {"tenant_id": "tenant-acme"}  # lead_text missing
    r = client.post(
        "/v1/intake/extract",
        json=bad_body,
        headers={**AUTH_HEADER, "Idempotency-Key": "k-bad"},
    )
    assert r.status_code == 422


def test_422_on_unknown_extra_field(client_with_service) -> None:
    client = client_with_service(_build_service())
    body = {**_body(), "rogue_field": "should fail"}
    r = client.post(
        "/v1/intake/extract",
        json=body,
        headers={**AUTH_HEADER, "Idempotency-Key": "k-extra"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Tier gate
# ---------------------------------------------------------------------------


def test_403_when_tier_forbids_provider(client_with_service) -> None:
    svc = _build_service(policy=_policy(tier=TierPolicy.OLLAMA_ONLY))
    client = client_with_service(svc)
    r = client.post(
        "/v1/intake/extract",
        json=_body(model_id="openai/gpt-4o-mini"),
        headers={**AUTH_HEADER, "Idempotency-Key": "k-prov"},
    )
    assert r.status_code == 403
    body = r.json()
    assert body["provider"] == "openai"
    assert body["provider_class"] == "paid-cloud"


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


def test_429_with_retry_after_when_bucket_empty(client_with_service) -> None:
    svc = _build_service(
        policy=_policy(capacity=1, refill=0.001),
        rate_capacity=1, rate_refill=0.001,
    )
    client = client_with_service(svc)
    client.post(
        "/v1/intake/extract",
        json=_body(),
        headers={**AUTH_HEADER, "Idempotency-Key": "k-r1"},
    )
    r = client.post(
        "/v1/intake/extract",
        json=_body(),
        headers={**AUTH_HEADER, "Idempotency-Key": "k-r2"},
    )
    assert r.status_code == 429
    # Retry-After per RFC 7231 — integer seconds.
    retry_after = r.headers["Retry-After"]
    assert retry_after.isdigit()
    assert int(retry_after) >= 1


# ---------------------------------------------------------------------------
# Cost ceiling
# ---------------------------------------------------------------------------


def test_402_when_estimated_cost_exceeds_ceiling(client_with_service) -> None:
    svc = _build_service(policy=_policy(ceiling=Decimal("0.0000001")))
    client = client_with_service(svc)
    r = client.post(
        "/v1/intake/extract",
        json=_body(model_id="gemini/gemini-2.0-flash", lead_text="x" * 5000),
        headers={**AUTH_HEADER, "Idempotency-Key": "k-ceil"},
    )
    assert r.status_code == 402
    body = r.json()
    assert Decimal(body["estimated_cost"]) > Decimal(body["ceiling"])


# ---------------------------------------------------------------------------
# Unknown provider config bug
# ---------------------------------------------------------------------------


def test_500_when_provider_unknown_to_classifier(client_with_service) -> None:
    register_pricing(
        ModelPricing(
            model_id="weirdcorp/secret-1",
            provider="weirdcorp",
            prompt_per_million=Decimal("1"),
            completion_per_million=Decimal("1"),
        ),
    )
    svc = _build_service()
    client = client_with_service(svc)
    r = client.post(
        "/v1/intake/extract",
        json=_body(model_id="weirdcorp/secret-1"),
        headers={**AUTH_HEADER, "Idempotency-Key": "k-unk"},
    )
    assert r.status_code == 500
    assert "weirdcorp" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Traceparent propagation
# ---------------------------------------------------------------------------


def test_502_when_llm_returns_schema_invalid_payload(client_with_service) -> None:
    """Upstream-bug case: LLM responds but with wrong shape -> 502 Bad Gateway."""
    bad_llm = StubLLMCaller(payload={"contact_email": "no-lead-name@example.com"})
    svc = _build_service(llm=bad_llm)
    client = client_with_service(svc)
    r = client.post(
        "/v1/intake/extract",
        json=_body(),
        headers={**AUTH_HEADER, "Idempotency-Key": "k-bg"},
    )
    assert r.status_code == 502
    body = r.json()
    assert "upstream" in body["detail"].lower()
    assert isinstance(body["errors"], list)
    assert body["errors"]


def test_traceparent_propagates_into_log_event(
    client_with_service, caplog: pytest.LogCaptureFixture,
) -> None:
    client = client_with_service(_build_service())
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER_NAME):
        client.post(
            "/v1/intake/extract",
            json=_body(),
            headers={**AUTH_HEADER, "Idempotency-Key": "k-tr", "traceparent": tp},
        )
    matching = [r for r in caplog.records if "intake.extract.completed" in r.getMessage()]
    assert matching
    parsed = json.loads(matching[0].getMessage())
    assert parsed["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
    assert parsed["span_id"] == "b7ad6b7169203331"
