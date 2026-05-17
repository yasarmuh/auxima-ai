"""Tests for ``auxima_ai.policy.enforcer`` — per-tenant authorisation composition.

Coverage:
  - TenantPolicy validation (tenant_id, tier enum, ceiling Decimal, rate params).
  - Tier gates: OLLAMA_ONLY refuses cloud; OLLAMA_THEN_FREE_CLOUD allows
    Gemini but not OpenAI; OLLAMA_THEN_PAID_CLOUD allows both.
  - Ollama always allowed regardless of tier.
  - Unknown tenant raises UnknownTenantError.
  - Unknown model raises UnknownModelError (loud failure on misconfig).
  - Provider classifier: known/unknown providers handled distinctly.
  - try_authorize order of checks: provider gate before ceiling before rate.
  - Ceiling check uses ESTIMATED cost.
  - Rate-limit consume only happens after ceiling passes.
  - record_spend persists actual cost via the ledger.
  - record_spend can fail with CeilingExceeded when actual >> estimate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from auxima_ai.cost.ledger import CeilingExceeded as LedgerCeilingExceeded, InMemoryCostLedger, Recorded
from auxima_ai.cost.pricing import UnknownModelError
from auxima_ai.policy.enforcer import (
    Authorized,
    CeilingWouldExceed,
    PolicyEnforcer,
    PolicyError,
    ProviderNotAllowed,
    RateLimited,
    TenantPolicy,
    TierPolicy,
    UnknownTenantError,
)
from auxima_ai.ratelimit.bucket import PerTenantRateLimiter

UTC = timezone.utc
TS = datetime(2026, 5, 17, 12, 0, tzinfo=UTC)


def _policy(
    *,
    tenant_id: str = "tenant-acme",
    tier: TierPolicy = TierPolicy.OLLAMA_THEN_PAID_CLOUD,
    monthly_ceiling: Decimal = Decimal("100.00"),
    rate_capacity: float = 1000.0,
    rate_refill_per_second: float = 100.0,
) -> TenantPolicy:
    return TenantPolicy(
        tenant_id=tenant_id,
        tier=tier,
        monthly_ceiling=monthly_ceiling,
        rate_capacity=rate_capacity,
        rate_refill_per_second=rate_refill_per_second,
    )


# ---------------------------------------------------------------------------
# TenantPolicy validation
# ---------------------------------------------------------------------------


def test_policy_valid_construction() -> None:
    p = _policy()
    assert p.tenant_id == "tenant-acme"
    assert p.tier == TierPolicy.OLLAMA_THEN_PAID_CLOUD


@pytest.mark.parametrize("bad", ["", None, 42])
def test_policy_rejects_bad_tenant(bad: object) -> None:
    with pytest.raises(PolicyError, match="tenant_id"):
        TenantPolicy(
            tenant_id=bad,  # type: ignore[arg-type]
            tier=TierPolicy.OLLAMA_ONLY,
            monthly_ceiling=Decimal("1"),
            rate_capacity=1,
            rate_refill_per_second=1,
        )


def test_policy_rejects_non_enum_tier() -> None:
    with pytest.raises(PolicyError, match="tier"):
        TenantPolicy(
            tenant_id="t", tier="ollama_only",  # type: ignore[arg-type]
            monthly_ceiling=Decimal("1"),
            rate_capacity=1, rate_refill_per_second=1,
        )


def test_policy_rejects_float_ceiling() -> None:
    with pytest.raises(PolicyError, match="Decimal"):
        TenantPolicy(
            tenant_id="t", tier=TierPolicy.OLLAMA_ONLY,
            monthly_ceiling=1.0,  # type: ignore[arg-type]
            rate_capacity=1, rate_refill_per_second=1,
        )


def test_policy_rejects_negative_ceiling() -> None:
    with pytest.raises(PolicyError, match=">= 0"):
        TenantPolicy(
            tenant_id="t", tier=TierPolicy.OLLAMA_ONLY,
            monthly_ceiling=Decimal("-1"),
            rate_capacity=1, rate_refill_per_second=1,
        )


@pytest.mark.parametrize("kwargs", [{"rate_capacity": 0}, {"rate_refill_per_second": -1}])
def test_policy_rejects_bad_rate_params(kwargs: dict) -> None:
    args = dict(
        tenant_id="t", tier=TierPolicy.OLLAMA_ONLY,
        monthly_ceiling=Decimal("1"),
        rate_capacity=1, rate_refill_per_second=1,
    )
    args.update(kwargs)
    with pytest.raises(PolicyError):
        TenantPolicy(**args)


def test_policy_is_frozen() -> None:
    p = _policy()
    with pytest.raises((AttributeError, TypeError)):
        p.tenant_id = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tier policy gating
# ---------------------------------------------------------------------------


def test_ollama_always_allowed_regardless_of_tier() -> None:
    enf = PolicyEnforcer()
    for tier in TierPolicy:
        enf.set_policy(_policy(tenant_id=f"t-{tier.value}", tier=tier))
        d = enf.try_authorize(
            f"t-{tier.value}", "ollama/qwen2.5:32b",
            estimated_prompt_tokens=10, estimated_completion_tokens=10, now=TS,
        )
        assert isinstance(d, Authorized), f"Ollama refused on tier {tier}"


def test_ollama_only_refuses_gemini() -> None:
    enf = PolicyEnforcer()
    enf.set_policy(_policy(tier=TierPolicy.OLLAMA_ONLY))
    d = enf.try_authorize(
        "tenant-acme", "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=10, estimated_completion_tokens=10, now=TS,
    )
    assert isinstance(d, ProviderNotAllowed)
    assert d.provider_class == "free-cloud"
    assert d.tier == TierPolicy.OLLAMA_ONLY


def test_free_cloud_tier_allows_gemini_but_not_openai() -> None:
    enf = PolicyEnforcer()
    enf.set_policy(_policy(tier=TierPolicy.OLLAMA_THEN_FREE_CLOUD))
    gemini = enf.try_authorize(
        "tenant-acme", "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=10, estimated_completion_tokens=10, now=TS,
    )
    assert isinstance(gemini, Authorized)
    openai = enf.try_authorize(
        "tenant-acme", "openai/gpt-4o-mini",
        estimated_prompt_tokens=10, estimated_completion_tokens=10, now=TS,
    )
    assert isinstance(openai, ProviderNotAllowed)
    assert openai.provider_class == "paid-cloud"


def test_paid_cloud_tier_allows_both_free_and_paid() -> None:
    enf = PolicyEnforcer()
    enf.set_policy(_policy(tier=TierPolicy.OLLAMA_THEN_PAID_CLOUD))
    for model in ("gemini/gemini-2.0-flash", "openai/gpt-4o-mini"):
        d = enf.try_authorize(
            "tenant-acme", model,
            estimated_prompt_tokens=10, estimated_completion_tokens=10, now=TS,
        )
        assert isinstance(d, Authorized), f"paid tier refused {model}"


# ---------------------------------------------------------------------------
# Cost ceiling — uses ESTIMATE
# ---------------------------------------------------------------------------


def test_ceiling_uses_estimate_and_rejects_overshoot() -> None:
    enf = PolicyEnforcer()
    enf.set_policy(_policy(monthly_ceiling=Decimal("0.05")))
    # 1M+1M tokens at gemini = $0.50 — well over ceiling.
    d = enf.try_authorize(
        "tenant-acme", "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=1_000_000,
        estimated_completion_tokens=1_000_000,
        now=TS,
    )
    assert isinstance(d, CeilingWouldExceed)
    assert d.estimated_cost == Decimal("0.500000")
    assert d.current_total == Decimal("0")
    assert d.ceiling == Decimal("0.05")


def test_ceiling_is_per_tenant() -> None:
    enf = PolicyEnforcer()
    enf.set_policy(_policy(tenant_id="a", monthly_ceiling=Decimal("0.01")))
    enf.set_policy(_policy(tenant_id="b", monthly_ceiling=Decimal("100")))
    a = enf.try_authorize(
        "a", "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=1_000_000, estimated_completion_tokens=1_000_000, now=TS,
    )
    b = enf.try_authorize(
        "b", "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=1_000_000, estimated_completion_tokens=1_000_000, now=TS,
    )
    assert isinstance(a, CeilingWouldExceed)
    assert isinstance(b, Authorized)


# ---------------------------------------------------------------------------
# Rate limit — only consumed after ceiling passes
# ---------------------------------------------------------------------------


def test_rate_limit_denied_when_bucket_empty() -> None:
    limiter = PerTenantRateLimiter(capacity=1, refill_per_second=0.001)
    enf = PolicyEnforcer(rate_limiter=limiter)
    enf.set_policy(_policy())
    # First call drains the bucket.
    first = enf.try_authorize(
        "tenant-acme", "ollama/qwen2.5:32b",
        estimated_prompt_tokens=1, estimated_completion_tokens=1, now=TS,
    )
    assert isinstance(first, Authorized)
    # Second call: rate limited.
    second = enf.try_authorize(
        "tenant-acme", "ollama/qwen2.5:32b",
        estimated_prompt_tokens=1, estimated_completion_tokens=1, now=TS,
    )
    assert isinstance(second, RateLimited)
    assert second.retry_after_seconds > 0


def test_provider_gate_runs_before_rate_limit() -> None:
    """A policy-banned call must NOT burn a rate token."""
    limiter = PerTenantRateLimiter(capacity=1, refill_per_second=0.001)
    enf = PolicyEnforcer(rate_limiter=limiter)
    enf.set_policy(_policy(tier=TierPolicy.OLLAMA_ONLY))
    blocked = enf.try_authorize(
        "tenant-acme", "openai/gpt-4o-mini",
        estimated_prompt_tokens=1, estimated_completion_tokens=1, now=TS,
    )
    assert isinstance(blocked, ProviderNotAllowed)
    # Bucket still full — followup Ollama call succeeds.
    follow = enf.try_authorize(
        "tenant-acme", "ollama/qwen2.5:32b",
        estimated_prompt_tokens=1, estimated_completion_tokens=1, now=TS,
    )
    assert isinstance(follow, Authorized)


def test_ceiling_check_runs_before_rate_limit() -> None:
    """A would-overshoot call must NOT burn a rate token."""
    limiter = PerTenantRateLimiter(capacity=1, refill_per_second=0.001)
    enf = PolicyEnforcer(rate_limiter=limiter)
    enf.set_policy(_policy(monthly_ceiling=Decimal("0.0000001")))
    blocked = enf.try_authorize(
        "tenant-acme", "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=1_000, estimated_completion_tokens=1_000, now=TS,
    )
    assert isinstance(blocked, CeilingWouldExceed)
    # Bucket still full.
    follow = enf.try_authorize(
        "tenant-acme", "ollama/qwen2.5:32b",
        estimated_prompt_tokens=1, estimated_completion_tokens=1, now=TS,
    )
    assert isinstance(follow, Authorized)


# ---------------------------------------------------------------------------
# Unknown tenant / model
# ---------------------------------------------------------------------------


def test_unknown_tenant_raises() -> None:
    enf = PolicyEnforcer()
    with pytest.raises(UnknownTenantError):
        enf.try_authorize(
            "ghost", "ollama/qwen2.5:32b",
            estimated_prompt_tokens=1, estimated_completion_tokens=1, now=TS,
        )


def test_unknown_model_raises_loudly() -> None:
    enf = PolicyEnforcer()
    enf.set_policy(_policy())
    with pytest.raises(UnknownModelError):
        enf.try_authorize(
            "tenant-acme", "weirdcorp/secret-model-9",
            estimated_prompt_tokens=1, estimated_completion_tokens=1, now=TS,
        )


# ---------------------------------------------------------------------------
# set_policy validation
# ---------------------------------------------------------------------------


def test_set_policy_rejects_non_policy() -> None:
    enf = PolicyEnforcer()
    with pytest.raises(PolicyError, match="TenantPolicy"):
        enf.set_policy("not-a-policy")  # type: ignore[arg-type]


def test_set_policy_mirrors_ceiling_into_ledger() -> None:
    ledger = InMemoryCostLedger()
    enf = PolicyEnforcer(ledger=ledger)
    enf.set_policy(_policy(monthly_ceiling=Decimal("42")))
    assert ledger.ceiling_for("tenant-acme") == Decimal("42")


# ---------------------------------------------------------------------------
# record_spend
# ---------------------------------------------------------------------------


def test_record_spend_persists_actual_cost() -> None:
    ledger = InMemoryCostLedger()
    enf = PolicyEnforcer(ledger=ledger)
    enf.set_policy(_policy(monthly_ceiling=Decimal("100")))
    enf.try_authorize(
        "tenant-acme", "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=100, estimated_completion_tokens=100, now=TS,
    )
    r = enf.record_spend(
        tenant_id="tenant-acme",
        model_id="gemini/gemini-2.0-flash",
        prompt_tokens=523,  # actual differs from estimate
        completion_tokens=187,
        latency_ms=873,
        ts=TS,
        model_version="2026-05",
    )
    assert isinstance(r, Recorded)
    assert ledger.entry_count() == 1


def test_record_spend_can_overshoot_ceiling_if_actual_exceeds_estimate() -> None:
    """If actual cost is much higher than estimate, the ledger STILL rejects;
    the enforcer never silently allows over-cap spend, even on already-admitted
    calls."""
    ledger = InMemoryCostLedger()
    enf = PolicyEnforcer(ledger=ledger)
    enf.set_policy(_policy(monthly_ceiling=Decimal("0.05")))
    # Admit on a tiny estimate.
    a = enf.try_authorize(
        "tenant-acme", "gemini/gemini-2.0-flash",
        estimated_prompt_tokens=10, estimated_completion_tokens=10, now=TS,
    )
    assert isinstance(a, Authorized)
    # Actual cost blows past ceiling.
    r = enf.record_spend(
        tenant_id="tenant-acme",
        model_id="gemini/gemini-2.0-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        latency_ms=873,
        ts=TS,
    )
    assert isinstance(r, LedgerCeilingExceeded)
    assert ledger.period_total("tenant-acme", TS) == Decimal("0")
