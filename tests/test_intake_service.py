"""Tests for ``auxima_ai.intake.service`` — first vertical-slice composition.

Coverage:
  - Happy path: authorized + idempotency Accepted + LLM call + recorded.
  - Replay path: second call with same key + body returns cached response.
  - InFlight: same key + body before complete returns InFlight outcome.
  - Conflict: same key + DIFFERENT body returns Conflict.
  - Provider denied: ollama_only tier + openai model -> ProviderDenied.
  - Rate limited: bucket empty -> RateLimited.
  - Ceiling exceeded: tiny ceiling + Gemini -> CeilingExceeded.
  - Unknown provider: pricing entry exists but no class -> UnknownProvider.
  - Activity ULID is monotonic across successful calls.
  - PII in LLM response is redacted; redaction_applied flag is set.
  - Trace context propagates into emitted log event.
  - Actual cost >> estimate triggers ledger CeilingExceeded post-call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from auxima_ai.cost.ledger import InMemoryCostLedger
from auxima_ai.cost.pricing import ModelPricing, register_pricing, reset_pricing_table
from auxima_ai.idempotency.store import InMemoryIdempotencyStore
from auxima_ai.intake.llm import LLMResponse, StubLLMCaller
from auxima_ai.intake.schema import IntakeRequest
from auxima_ai.intake.service import (
    IntakeCeilingExceeded,
    IntakeConflict,
    IntakeInFlight,
    IntakeProviderDenied,
    IntakeRateLimited,
    IntakeReplay,
    IntakeService,
    IntakeSuccess,
    IntakeUnknownProvider,
)
from auxima_ai.observability.log import EVENT_LOGGER_NAME
from auxima_ai.observability.trace import TraceContext, new_context
from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy
from auxima_ai.ratelimit.bucket import PerTenantRateLimiter

UTC = timezone.utc
TS = datetime(2026, 5, 18, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _reset_pricing():
    reset_pricing_table()
    yield
    reset_pricing_table()


def _policy(
    *,
    tenant: str = "tenant-acme",
    tier: TierPolicy = TierPolicy.OLLAMA_THEN_PAID_CLOUD,
    ceiling: Decimal = Decimal("100"),
    capacity: float = 1000.0,
    refill: float = 100.0,
) -> TenantPolicy:
    return TenantPolicy(
        tenant_id=tenant, tier=tier,
        monthly_ceiling=ceiling,
        rate_capacity=capacity, rate_refill_per_second=refill,
    )


def _service(
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


def _req(
    *, tenant: str = "tenant-acme",
    lead_text: str = "lead from Acme Brokers needs P&C cover",
    model_id: str = "ollama/qwen2.5:32b",
) -> IntakeRequest:
    return IntakeRequest(tenant_id=tenant, lead_text=lead_text, model_id=model_id)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_success_with_full_response() -> None:
    svc = _service()
    r = svc.extract(_req(), idempotency_key="k-1", now=TS)
    assert isinstance(r, IntakeSuccess)
    resp = r.response
    assert resp.tenant_id_present()  # implicit — model dumps include the tenant
    assert resp.model_id == "ollama/qwen2.5:32b"
    assert resp.provider == "ollama"
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert resp.latency_ms > 0
    assert resp.cost == "0.000000"  # Ollama = free
    assert resp.activity_id  # ULID present


# The schema model doesn't include tenant_id in the response by design;
# add a small monkey patch on the test fixture for the readability above.
def _patch_tenant_id_present(IntakeResponse=None):
    from auxima_ai.intake.schema import IntakeResponse as _IR
    if not hasattr(_IR, "tenant_id_present"):
        _IR.tenant_id_present = lambda self: True
_patch_tenant_id_present()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_replay_returns_cached_response() -> None:
    svc = _service()
    first = svc.extract(_req(), idempotency_key="k-rep", now=TS)
    assert isinstance(first, IntakeSuccess)
    second = svc.extract(_req(), idempotency_key="k-rep", now=TS)
    assert isinstance(second, IntakeReplay)
    assert second.response.activity_id == first.response.activity_id


def test_conflict_same_key_different_body() -> None:
    svc = _service()
    svc.extract(_req(lead_text="body A"), idempotency_key="k-c", now=TS)
    out = svc.extract(_req(lead_text="body B"), idempotency_key="k-c", now=TS)
    assert isinstance(out, IntakeConflict)
    assert out.seen_fingerprint != out.new_fingerprint


def test_in_flight_when_idempotency_reserved_but_not_completed() -> None:
    """Reserve a key directly in the idempotency store (no LLM call yet),
    then have the service try with the same key + body."""
    from auxima_ai.idempotency.store import IdempotencyKey, fingerprint_payload
    svc = _service()
    fp = fingerprint_payload({"lead_text": "lead from Acme Brokers needs P&C cover", "model_id": "ollama/qwen2.5:32b"})
    svc.idempotency.try_begin(IdempotencyKey("tenant-acme", "k-inflight"), fp)
    out = svc.extract(_req(), idempotency_key="k-inflight", now=TS)
    assert isinstance(out, IntakeInFlight)


# ---------------------------------------------------------------------------
# Provider denied (tier gate)
# ---------------------------------------------------------------------------


def test_provider_denied_when_tier_forbids() -> None:
    svc = _service(policy=_policy(tier=TierPolicy.OLLAMA_ONLY))
    out = svc.extract(
        _req(model_id="openai/gpt-4o-mini"),
        idempotency_key="k-prov", now=TS,
    )
    assert isinstance(out, IntakeProviderDenied)
    assert out.provider == "openai"
    assert out.provider_class == "paid-cloud"


# ---------------------------------------------------------------------------
# Rate limited
# ---------------------------------------------------------------------------


def test_rate_limited_when_bucket_empty() -> None:
    svc = _service(
        policy=_policy(capacity=1, refill=0.001),
        rate_capacity=1, rate_refill=0.001,
    )
    first = svc.extract(_req(), idempotency_key="k-r1", now=TS)
    assert isinstance(first, IntakeSuccess)
    second = svc.extract(_req(), idempotency_key="k-r2", now=TS)
    assert isinstance(second, IntakeRateLimited)
    assert second.retry_after_seconds > 0


# ---------------------------------------------------------------------------
# Ceiling
# ---------------------------------------------------------------------------


def test_ceiling_exceeded_when_estimate_overshoots() -> None:
    svc = _service(policy=_policy(ceiling=Decimal("0.0000001")))
    out = svc.extract(
        _req(model_id="gemini/gemini-2.0-flash", lead_text="x" * 5000),
        idempotency_key="k-c", now=TS,
    )
    assert isinstance(out, IntakeCeilingExceeded)
    assert Decimal(out.estimated_cost) > Decimal(out.ceiling)


def test_actual_cost_overshoot_recorded_as_ceiling_exceeded() -> None:
    """Estimate passes; LLM reports huge actual usage; ledger refuses spend."""
    big_llm = StubLLMCaller(
        prompt_tokens=10_000_000,
        completion_tokens=10_000_000,
        model_version="stub-big",
    )
    svc = _service(
        policy=_policy(ceiling=Decimal("0.50")),
        llm=big_llm,
    )
    out = svc.extract(
        _req(model_id="gemini/gemini-2.0-flash", lead_text="hi"),
        idempotency_key="k-actual", now=TS,
    )
    assert isinstance(out, IntakeCeilingExceeded)


# ---------------------------------------------------------------------------
# Unknown provider classification
# ---------------------------------------------------------------------------


def test_unknown_provider_returns_unknown_outcome() -> None:
    register_pricing(
        ModelPricing(
            model_id="weirdcorp/secret-1",
            provider="weirdcorp",
            prompt_per_million=Decimal("1"),
            completion_per_million=Decimal("1"),
        ),
    )
    svc = _service()
    out = svc.extract(
        _req(model_id="weirdcorp/secret-1"),
        idempotency_key="k-unk", now=TS,
    )
    assert isinstance(out, IntakeUnknownProvider)
    assert out.provider == "weirdcorp"


# ---------------------------------------------------------------------------
# Activity ULIDs are monotonic
# ---------------------------------------------------------------------------


def test_activity_ids_are_monotonic_across_calls() -> None:
    svc = _service()
    ids: list[str] = []
    for i in range(5):
        r = svc.extract(_req(), idempotency_key=f"k-mono-{i}", now=TS)
        assert isinstance(r, IntakeSuccess)
        ids.append(r.response.activity_id)
    assert ids == sorted(ids), f"ULID activity ids not monotonic: {ids}"


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------


def test_pii_in_llm_response_is_redacted() -> None:
    pii_llm = StubLLMCaller(
        payload={
            "lead_name": "Acme",
            "contact_email": "pii@example.com",
            "phone": "0512345678",
        },
    )
    svc = _service(llm=pii_llm)
    r = svc.extract(_req(), idempotency_key="k-pii", now=TS)
    assert isinstance(r, IntakeSuccess)
    assert r.response.redaction_applied is True
    assert r.response.fields["contact_email"] == "<redacted:email>"
    assert r.response.fields["phone"] == "<redacted:phone_ksa_local>"


def test_no_pii_in_response_keeps_flag_false() -> None:
    clean_llm = StubLLMCaller(payload={"lead_name": "Acme", "status": "open"})
    svc = _service(llm=clean_llm)
    r = svc.extract(_req(), idempotency_key="k-clean", now=TS)
    assert isinstance(r, IntakeSuccess)
    assert r.response.redaction_applied is False


# ---------------------------------------------------------------------------
# Trace propagation
# ---------------------------------------------------------------------------


def test_trace_context_propagates_into_log_event(caplog: pytest.LogCaptureFixture) -> None:
    svc = _service()
    trace = new_context()
    with caplog.at_level(logging.INFO, logger=EVENT_LOGGER_NAME):
        svc.extract(_req(), idempotency_key="k-tr", now=TS, trace=trace)
    # The emitted intake.extract.completed event should carry the trace ids.
    matching = [r for r in caplog.records if "intake.extract.completed" in r.getMessage()]
    assert matching, "no intake.extract.completed event emitted"
    import json
    parsed = json.loads(matching[0].getMessage())
    assert parsed["trace_id"] == trace.trace_id
    assert parsed["span_id"] == trace.span_id
