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
