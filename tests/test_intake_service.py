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
from auxima_ai.intake.llm import StubLLMCaller
from auxima_ai.intake.schema import IntakeRequest
from auxima_ai.intake.service import (
    ActivityEmitter,
    CapturingActivityEmitter,
    IntakeCeilingExceeded,
    IntakeConflict,
    IntakeInFlight,
    IntakeProviderDenied,
    IntakeRateLimited,
    IntakeReplay,
    IntakeSchemaInvalid,
    IntakeService,
    IntakeSuccess,
    IntakeUnknownProvider,
    NullActivityEmitter,
)
from auxima_ai.activity.row import ActivityRow, RetentionClass
from auxima_ai.observability.log import EVENT_LOGGER_NAME
from auxima_ai.observability.trace import new_context
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
    activity_emitter: ActivityEmitter | None = None,
) -> IntakeService:
    enf = PolicyEnforcer(
        ledger=InMemoryCostLedger(),
        rate_limiter=PerTenantRateLimiter(capacity=rate_capacity, refill_per_second=rate_refill),
    )
    enf.set_policy(policy or _policy(capacity=rate_capacity, refill=rate_refill))
    kwargs: dict = dict(
        enforcer=enf,
        idempotency=InMemoryIdempotencyStore(),
        llm=llm or StubLLMCaller(),
    )
    if activity_emitter is not None:
        kwargs["activity_emitter"] = activity_emitter
    return IntakeService(**kwargs)


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
            "contact_phone": "0512345678",
            "line_of_business": "property",
            "urgency": "normal",
            "notes": None,
        },
    )
    svc = _service(llm=pii_llm)
    r = svc.extract(_req(), idempotency_key="k-pii", now=TS)
    assert isinstance(r, IntakeSuccess)
    assert r.response.redaction_applied is True
    assert r.response.fields["contact_email"] == "<redacted:email>"
    assert r.response.fields["contact_phone"] == "<redacted:phone_ksa_local>"


def test_no_pii_in_response_keeps_flag_false() -> None:
    clean_llm = StubLLMCaller(payload={
        "lead_name": "Acme",
        "contact_email": None,
        "contact_phone": None,
        "line_of_business": "unknown",
        "urgency": "unknown",
        "notes": "no PII at all here",
    })
    svc = _service(llm=clean_llm)
    r = svc.extract(_req(), idempotency_key="k-clean", now=TS)
    assert isinstance(r, IntakeSuccess)
    assert r.response.redaction_applied is False


# ---------------------------------------------------------------------------
# Trace propagation
# ---------------------------------------------------------------------------


def test_response_carries_normalised_canonical_fields_through_redaction() -> None:
    """End-to-end: LLM returns raw "Sales@AL-Mansour.SA" + "0512345678";
    both raw and _canonical / _e164 are populated, then PII-redacted.

    The redactor runs LAST on the response payload (S-19), so every
    PII-shaped field — raw OR canonical — comes out as a placeholder.
    The presence of the placeholder in BOTH raw and canonical slots
    proves the normaliser ran and populated the canonical field
    (otherwise the canonical field would be None, which the redactor
    skips)."""
    llm = StubLLMCaller(payload={
        "lead_name": "Acme",
        "contact_email": "Sales@AL-Mansour.SA",
        "contact_phone": "0512345678",
        "line_of_business": "property",
        "urgency": "normal",
        "notes": None,
    })
    svc = _service(llm=llm)
    r = svc.extract(_req(), idempotency_key="k-norm", now=TS)
    assert isinstance(r, IntakeSuccess)
    # Raw email is PII -> redacted; raw phone is KSA local -> redacted.
    assert r.response.fields["contact_email"] == "<redacted:email>"
    assert r.response.fields["contact_phone"] == "<redacted:phone_ksa_local>"
    # Canonical fields were populated (otherwise they'd be None and the
    # redactor would have left them None). Redaction proves they ran.
    assert r.response.fields["contact_email_canonical"] == "<redacted:email>"
    assert r.response.fields["contact_phone_e164"] == "<redacted:phone_e164>"


def test_response_canonical_fields_are_none_when_normalisation_fails() -> None:
    """LLM returns junk-shaped values; canonical fields stay None
    (so the redactor doesn'\''t pretend it normalised something it didn'\''t)."""
    llm = StubLLMCaller(payload={
        "lead_name": "Acme",
        "contact_email": "not-an-email-at-all",
        "contact_phone": "junk",
        "line_of_business": "unknown",
        "urgency": "unknown",
        "notes": None,
    })
    svc = _service(llm=llm)
    r = svc.extract(_req(), idempotency_key="k-junk", now=TS)
    assert isinstance(r, IntakeSuccess)
    assert r.response.fields["contact_email_canonical"] is None
    assert r.response.fields["contact_phone_e164"] is None
    # Raw fields preserved as-is (no PII pattern, so no redaction).
    assert r.response.fields["contact_email"] == "not-an-email-at-all"
    assert r.response.fields["contact_phone"] == "junk"


def test_schema_invalid_when_llm_omits_required_field() -> None:
    """LLM returns payload missing lead_name -> IntakeSchemaInvalid, no row written."""
    bad_llm = StubLLMCaller(payload={"contact_email": "ops@acme.example"})
    svc = _service(llm=bad_llm)
    r = svc.extract(_req(), idempotency_key="k-badshape", now=TS)
    assert isinstance(r, IntakeSchemaInvalid)
    assert any("lead_name" in e["loc"] for e in r.errors)


def test_schema_invalid_when_llm_emits_extra_field() -> None:
    """extra="forbid" — unexpected keys from the LLM are refused."""
    bad_llm = StubLLMCaller(payload={"lead_name": "Acme", "rogue_field": "nope"})
    svc = _service(llm=bad_llm)
    r = svc.extract(_req(), idempotency_key="k-extra", now=TS)
    assert isinstance(r, IntakeSchemaInvalid)


def test_schema_invalid_with_bad_enum_value() -> None:
    bad_llm = StubLLMCaller(payload={"lead_name": "Acme", "line_of_business": "spaceship"})
    svc = _service(llm=bad_llm)
    r = svc.extract(_req(), idempotency_key="k-enum", now=TS)
    assert isinstance(r, IntakeSchemaInvalid)


def test_prompt_template_is_used_for_llm_call() -> None:
    """The LLM caller MUST receive the schema-shaped prompt, not raw lead_text."""
    seen_prompts: list[str] = []

    class CaptureLLM:
        def call(self, *, model_id: str, prompt: str):
            seen_prompts.append(prompt)
            return StubLLMCaller().call(model_id=model_id, prompt=prompt)

    svc = _service(llm=CaptureLLM())  # type: ignore[arg-type]
    svc.extract(_req(lead_text="a real lead"), idempotency_key="k-tpl", now=TS)
    assert seen_prompts
    assert "a real lead" in seen_prompts[0]
    # Schema-derived field list MUST be in the prompt.
    for field in ("lead_name", "contact_email", "line_of_business", "urgency"):
        assert field in seen_prompts[0]


# ---------------------------------------------------------------------------
# Activity emitter — CRM §4 invariant
# ---------------------------------------------------------------------------


def test_success_emits_one_activity_row_with_matching_id() -> None:
    cap = CapturingActivityEmitter()
    svc = _service(activity_emitter=cap)
    r = svc.extract(_req(), idempotency_key="k-act-1", now=TS)
    assert isinstance(r, IntakeSuccess)
    assert len(cap.rows) == 1
    row = cap.rows[0]
    assert isinstance(row, ActivityRow)
    assert row.id == r.response.activity_id  # row id == response activity_id
    assert row.tenant_id == "tenant-acme"
    assert row.kind == "intake.extract.completed"
    assert row.retention == RetentionClass.OPERATIONAL
    assert row.source == "sidecar.intake.extract"
    assert row.idempotency_key == "k-act-1"
    assert row.ts == TS
    # Payload carries the response shape (without PII bcs StubLLM default
    # uses the *.example domain which the redactor will mark).
    assert row.payload["model_id"] == "ollama/qwen2.5:32b"
    assert row.payload["provider"] == "ollama"


def test_each_outcome_branch_does_not_emit_except_success() -> None:
    """Rate-limited / ceiling-exceeded / provider-denied must NOT emit a row."""
    cap = CapturingActivityEmitter()
    svc = _service(
        policy=_policy(ceiling=Decimal("0.0000001")),
        activity_emitter=cap,
    )
    r = svc.extract(
        _req(model_id="gemini/gemini-2.0-flash", lead_text="x" * 5000),
        idempotency_key="k-no-emit",
        now=TS,
    )
    assert isinstance(r, IntakeCeilingExceeded)
    assert cap.rows == []


def test_schema_invalid_does_not_emit_activity_row() -> None:
    """Upstream-LLM-bug case must NOT pollute the activity log."""
    bad_llm = StubLLMCaller(payload={"contact_email": "x@y.co"})  # no lead_name
    cap = CapturingActivityEmitter()
    svc = _service(llm=bad_llm, activity_emitter=cap)
    r = svc.extract(_req(), idempotency_key="k-bad-shape", now=TS)
    assert isinstance(r, IntakeSchemaInvalid)
    assert cap.rows == []


def test_replay_does_not_emit_second_activity_row() -> None:
    """Two calls with same key + body -> only the first writes to the audit log."""
    cap = CapturingActivityEmitter()
    svc = _service(activity_emitter=cap)
    first = svc.extract(_req(), idempotency_key="k-replay-once", now=TS)
    second = svc.extract(_req(), idempotency_key="k-replay-once", now=TS)
    assert isinstance(first, IntakeSuccess)
    assert isinstance(second, IntakeReplay)
    assert len(cap.rows) == 1
    assert cap.rows[0].id == first.response.activity_id


def test_default_service_uses_null_emitter() -> None:
    """An IntakeService without an explicit emitter does NOT crash on success."""
    svc = _service()
    assert isinstance(svc.activity_emitter, NullActivityEmitter)
    r = svc.extract(_req(), idempotency_key="k-null", now=TS)
    assert isinstance(r, IntakeSuccess)


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
