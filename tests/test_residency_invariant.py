"""ADR-GA2 — region residency hard-invariant (above the per-tenant tier flag).

A KSA / in-Kingdom tenant must NEVER egress to a cloud provider, **regardless**
of its tier flag and **regardless** of whether a cloud API key is configured.
This is the data-residency safety net above the tier policy: insurance customer
PII stays in-Kingdom (Insurance Market Code of Conduct + PDPL Transfer
Regulation). See ``Docs/Planning/decisions/ADR-GA2-llm-egress-residency.md``.

Design: ``TenantPolicy.region`` defaults to ``"KSA"`` (fail-closed — Phase-1 is
KSA-only). A tenant that should be allowed a cloud tier must be EXPLICITLY
marked with a non-in-Kingdom region.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from auxima_ai.policy.enforcer import (
    Authorized,
    PolicyEnforcer,
    ProviderNotAllowed,
    TenantPolicy,
    TierPolicy,
)

_NOW = datetime.now(timezone.utc)


def _policy(tenant_id: str, tier: TierPolicy, region: str = "KSA") -> TenantPolicy:
    return TenantPolicy(
        tenant_id=tenant_id,
        tier=tier,
        region=region,
        monthly_ceiling=Decimal("100"),
        rate_capacity=1000.0,
        rate_refill_per_second=100.0,
    )


def test_ksa_tenant_blocked_from_cloud_even_on_paid_cloud_tier():
    """The misconfiguration case: a KSA tenant wrongly set to a cloud tier
    must still be blocked from every cloud provider_class."""
    e = PolicyEnforcer()
    e.set_policy(_policy("ksa-1", TierPolicy.OLLAMA_THEN_PAID_CLOUD))
    assert e.provider_class_allowed("ksa-1", "self-hosted") is True
    assert e.provider_class_allowed("ksa-1", "free-cloud") is False
    assert e.provider_class_allowed("ksa-1", "paid-cloud") is False


def test_ksa_tenant_try_authorize_denies_cloud_model_regardless_of_tier():
    e = PolicyEnforcer()
    e.set_policy(_policy("ksa-2", TierPolicy.OLLAMA_THEN_PAID_CLOUD))
    decision = e.try_authorize(
        "ksa-2",
        "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=10,
        estimated_completion_tokens=10,
        now=_NOW,
    )
    assert isinstance(decision, ProviderNotAllowed)
    assert decision.provider_class == "free-cloud"


def test_ksa_tenant_self_hosted_still_authorized():
    e = PolicyEnforcer()
    e.set_policy(_policy("ksa-3", TierPolicy.OLLAMA_THEN_PAID_CLOUD))
    decision = e.try_authorize(
        "ksa-3",
        "ollama/llama3.1:8b",
        estimated_prompt_tokens=10,
        estimated_completion_tokens=10,
        now=_NOW,
    )
    assert isinstance(decision, Authorized)


def test_default_region_is_in_kingdom_fail_closed():
    """A tenant constructed WITHOUT an explicit region defaults to in-Kingdom
    and is cloud-blocked (Phase-1 KSA-only fail-closed posture)."""
    e = PolicyEnforcer()
    e.set_policy(
        TenantPolicy(
            tenant_id="d1",
            tier=TierPolicy.OLLAMA_THEN_FREE_CLOUD,
            monthly_ceiling=Decimal("100"),
            rate_capacity=1000.0,
            rate_refill_per_second=100.0,
        )
    )
    assert e.provider_class_allowed("d1", "free-cloud") is False


def test_non_ksa_tenant_cloud_follows_tier():
    """An explicitly non-in-Kingdom tenant is governed by its tier flag again."""
    e = PolicyEnforcer()
    e.set_policy(_policy("intl-1", TierPolicy.OLLAMA_THEN_FREE_CLOUD, region="INTL"))
    assert e.provider_class_allowed("intl-1", "free-cloud") is True
    assert e.provider_class_allowed("intl-1", "paid-cloud") is False  # tier still caps


def test_region_invariant_is_case_insensitive():
    e = PolicyEnforcer()
    e.set_policy(_policy("ksa-lc", TierPolicy.OLLAMA_THEN_PAID_CLOUD, region="ksa"))
    assert e.provider_class_allowed("ksa-lc", "paid-cloud") is False


# ---------------------------------------------------------------------------
# ADR-GA3 — OpenRouter cloud approved for ALL tenants incl. KSA, gated by an
# explicit per-tenant `cloud_egress_approved` flag (default False = fail-closed).
# The relax is deliberately NARROW: an approved in-Kingdom tenant may egress to
# OpenRouter (the contracted provider) ONLY — direct OpenAI/Gemini stays blocked,
# and the tier flag still caps independently. ADR-GA3 §3.1.
# ---------------------------------------------------------------------------


def _approved(tenant_id: str, tier: TierPolicy = TierPolicy.OLLAMA_THEN_PAID_CLOUD,
              region: str = "KSA") -> TenantPolicy:
    return TenantPolicy(
        tenant_id=tenant_id,
        tier=tier,
        region=region,
        monthly_ceiling=Decimal("100"),
        rate_capacity=1000.0,
        rate_refill_per_second=100.0,
        cloud_egress_approved=True,
    )


_OPENROUTER_MODEL = "openrouter/qwen/qwen-2.5-72b-instruct"


def test_approved_ksa_tenant_allowed_openrouter():
    """The ADR-GA3 go-forward: an explicitly approved KSA tenant on a cloud tier
    may egress to OpenRouter."""
    e = PolicyEnforcer()
    e.set_policy(_approved("ksa-or"))
    decision = e.try_authorize(
        "ksa-or", _OPENROUTER_MODEL,
        estimated_prompt_tokens=10, estimated_completion_tokens=10, now=_NOW,
    )
    assert isinstance(decision, Authorized)
    assert decision.provider == "openrouter"


def test_nonapproved_ksa_tenant_still_blocked_from_openrouter():
    """Fail-closed default preserved: a KSA tenant NOT covered by the agreement
    (cloud_egress_approved defaults False) gets no cloud — not even OpenRouter."""
    e = PolicyEnforcer()
    e.set_policy(_policy("ksa-na", TierPolicy.OLLAMA_THEN_PAID_CLOUD))  # approved defaults False
    decision = e.try_authorize(
        "ksa-na", _OPENROUTER_MODEL,
        estimated_prompt_tokens=10, estimated_completion_tokens=10, now=_NOW,
    )
    assert isinstance(decision, ProviderNotAllowed)


def test_approved_ksa_tenant_still_blocked_from_direct_cloud():
    """The agreement is OpenRouter-specific: an approved KSA tenant may NOT egress
    to a direct cloud provider (gemini/openai). Defence-in-depth above the tier."""
    e = PolicyEnforcer()
    e.set_policy(_approved("ksa-direct"))
    decision = e.try_authorize(
        "ksa-direct", "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=10, estimated_completion_tokens=10, now=_NOW,
    )
    assert isinstance(decision, ProviderNotAllowed)


def test_approval_does_not_override_tier_cap():
    """Approval lifts the RESIDENCY gate only; the TIER still governs. An approved
    KSA tenant on `ollama_only` still reaches no cloud (both gates must pass)."""
    e = PolicyEnforcer()
    e.set_policy(_approved("ksa-oo", tier=TierPolicy.OLLAMA_ONLY))
    decision = e.try_authorize(
        "ksa-oo", _OPENROUTER_MODEL,
        estimated_prompt_tokens=10, estimated_completion_tokens=10, now=_NOW,
    )
    assert isinstance(decision, ProviderNotAllowed)


def test_approved_ksa_tenant_self_hosted_unaffected():
    """Approval doesn't change the always-allowed self-hosted path."""
    e = PolicyEnforcer()
    e.set_policy(_approved("ksa-oll"))
    decision = e.try_authorize(
        "ksa-oll", "ollama/llama3.1:8b",
        estimated_prompt_tokens=10, estimated_completion_tokens=10, now=_NOW,
    )
    assert isinstance(decision, Authorized)


def test_approved_ksa_provider_class_allowed_for_assist():
    """The coarse assist gate (provider_class) also opens paid-cloud for an
    approved KSA tenant (the actual provider is controlled by bootstrap wiring +
    redaction); a non-approved KSA tenant stays blocked."""
    e = PolicyEnforcer()
    e.set_policy(_approved("ksa-assist"))
    e.set_policy(_policy("ksa-assist-na", TierPolicy.OLLAMA_THEN_PAID_CLOUD))
    assert e.provider_class_allowed("ksa-assist", "paid-cloud") is True
    assert e.provider_class_allowed("ksa-assist", "self-hosted") is True
    assert e.provider_class_allowed("ksa-assist-na", "paid-cloud") is False


def test_unlisted_openrouter_model_refused_loudly():
    """The model allow-list is the curated OpenRouter pricing set: an unvetted
    OpenRouter model id is refused loudly (UnknownModelError), not silently run."""
    import pytest

    from auxima_ai.cost.pricing import UnknownModelError

    e = PolicyEnforcer()
    e.set_policy(_approved("ksa-unlisted"))
    with pytest.raises(UnknownModelError):
        e.try_authorize(
            "ksa-unlisted", "openrouter/some/unvetted-model-xyz",
            estimated_prompt_tokens=10, estimated_completion_tokens=10, now=_NOW,
        )


def test_non_ksa_approved_flag_is_noop():
    """For a non-in-Kingdom tenant the flag is irrelevant — the tier governs as
    before (no residency restriction to lift)."""
    e = PolicyEnforcer()
    e.set_policy(_approved("intl-or", region="INTL"))
    assert e.provider_class_allowed("intl-or", "paid-cloud") is True
