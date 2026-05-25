"""Per-tenant policy enforcer (CLAUDE §2 + S-19).

The enforcer answers one question per request: **may this tenant make
this LLM call right now?** It composes three pre-existing primitives:

  1. **Tier policy** — does the tenant's tier flag allow the provider
     of ``model_id``? Free-tier-cloud and paid-tier-cloud are explicit
     opt-ins per the tenant's data-egress acknowledgement.
  2. **Rate limit** — is the tenant under their token-bucket quota?
  3. **Cost ceiling** — would this call push the monthly ledger total
     past the tenant's configured ceiling?

If any check fails, the LLM call is REFUSED before any bytes go on the
wire — the tenant never overspends, never gets rate-limited at the
provider, and never invokes a provider their policy bans.

If all three checks pass, the enforcer **provisionally** consumes a
rate-limit token. The caller MUST follow up with :meth:`record_spend`
once the actual LLM cost is known (the ledger records the real cost,
not the estimate). This two-step (admit-then-record) split mirrors
the real cost path: we admit on an estimate, then bill on the truth.

Pure stdlib + project-internal deps; no FastAPI / Frappe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Final

from auxima_ai.cost.ledger import (
    CeilingExceeded as LedgerCeilingExceeded,
    CostLedger,
    InMemoryCostLedger,
    LedgerEntry,
    Recorded,
)
from auxima_ai.cost.pricing import (
    cost_for,
    pricing_for,
)
from auxima_ai.ratelimit.bucket import (
    Allowed,
    Denied,
    PerTenantRateLimiter,
    RateLimiter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PolicyError(ValueError):
    """Base — invalid configuration / inputs raise a subclass of this."""


class UnknownTenantError(PolicyError):
    """Raised when a tenant_id has no registered :class:`TenantPolicy`."""


# ---------------------------------------------------------------------------
# Tier policy
# ---------------------------------------------------------------------------


class TierPolicy(str, Enum):
    """Per-tenant tier flag — CLAUDE §2 LLM-runtime policy."""

    OLLAMA_ONLY = "ollama_only"
    OLLAMA_THEN_FREE_CLOUD = "ollama_then_free_cloud"
    OLLAMA_THEN_PAID_CLOUD = "ollama_then_paid_cloud"


# Tags applied to ModelPricing.provider so the enforcer knows which
# tier-policy gate the provider falls under.
_FREE_CLOUD_PROVIDERS: Final[frozenset[str]] = frozenset({"gemini"})
_PAID_CLOUD_PROVIDERS: Final[frozenset[str]] = frozenset({"openai", "anthropic"})
_SELF_HOSTED_PROVIDERS: Final[frozenset[str]] = frozenset({"ollama"})

#: Regions whose residency rules require in-Kingdom inference. A tenant in one
#: of these regions may use SELF-HOSTED providers only — no cloud egress —
#: regardless of its tier flag or whether a cloud API key is set. This is the
#: KSA data-residency hard-invariant ABOVE the tier policy (ADR-GA2:
#: Insurance Market Code of Conduct localizes customer PII in-Kingdom; PDPL
#: Transfer Regulation has no published adequacy list). Compared upper-cased.
_IN_KINGDOM_REGIONS: Final[frozenset[str]] = frozenset({"KSA", "SA"})


def _classify_provider(provider: str) -> str:
    """Return one of ``self-hosted`` / ``free-cloud`` / ``paid-cloud`` / ``unknown``."""
    if provider in _SELF_HOSTED_PROVIDERS:
        return "self-hosted"
    if provider in _FREE_CLOUD_PROVIDERS:
        return "free-cloud"
    if provider in _PAID_CLOUD_PROVIDERS:
        return "paid-cloud"
    return "unknown"


def _is_in_kingdom(region: str) -> bool:
    """True if ``region`` is an in-Kingdom residency region (case-insensitive)."""
    return region.strip().upper() in _IN_KINGDOM_REGIONS


def _residency_allows(region: str, provider_class: str) -> bool:
    """Region hard-invariant — does ``region`` permit ``provider_class``?

    In-Kingdom regions (KSA) permit **self-hosted only**: every cloud
    provider_class is refused regardless of the tenant's tier. Non-in-Kingdom
    regions impose no residency restriction (the tier policy governs). This sits
    ABOVE :func:`_policy_allows`; a call is permitted only if BOTH gates pass.
    """
    if provider_class == "self-hosted":
        return True
    return not _is_in_kingdom(region)


def _policy_allows(tier: TierPolicy, provider_class: str) -> bool:
    """Pure-function policy gate — does ``tier`` allow ``provider_class``?"""
    if provider_class == "self-hosted":
        return True  # Ollama always allowed on every tier.
    if tier == TierPolicy.OLLAMA_ONLY:
        return False
    if tier == TierPolicy.OLLAMA_THEN_FREE_CLOUD:
        return provider_class == "free-cloud"
    if tier == TierPolicy.OLLAMA_THEN_PAID_CLOUD:
        return provider_class in ("free-cloud", "paid-cloud")
    return False  # defensive


# ---------------------------------------------------------------------------
# Tenant policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantPolicy:
    """The full per-tenant policy snapshot."""

    tenant_id: str
    tier: TierPolicy
    monthly_ceiling: Decimal
    rate_capacity: float
    rate_refill_per_second: float
    #: Residency region. Defaults to ``"KSA"`` (in-Kingdom) — fail-closed for
    #: the Phase-1 KSA-only market: an unmarked tenant is treated as in-Kingdom
    #: and cloud-blocked regardless of tier (ADR-GA2). A tenant that should be
    #: allowed a cloud tier MUST be explicitly marked a non-in-Kingdom region.
    region: str = "KSA"

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id:
            raise PolicyError("tenant_id must be a non-empty string")
        if not isinstance(self.tier, TierPolicy):
            raise PolicyError(f"tier must be TierPolicy; got {type(self.tier).__name__}")
        if not isinstance(self.region, str) or not self.region.strip():
            raise PolicyError("region must be a non-empty string")
        if not isinstance(self.monthly_ceiling, Decimal):
            raise PolicyError("monthly_ceiling must be Decimal")
        if self.monthly_ceiling.is_nan() or self.monthly_ceiling < 0:
            raise PolicyError(
                f"monthly_ceiling must be finite + >= 0; got {self.monthly_ceiling}"
            )
        if self.rate_capacity <= 0:
            raise PolicyError(f"rate_capacity must be > 0; got {self.rate_capacity}")
        if self.rate_refill_per_second <= 0:
            raise PolicyError(
                f"rate_refill_per_second must be > 0; got {self.rate_refill_per_second}"
            )


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Authorized:
    """All checks passed; the LLM call MAY proceed."""

    tenant_id: str
    model_id: str
    provider: str
    estimated_cost: Decimal
    period_total_estimate: Decimal


@dataclass(frozen=True)
class ProviderNotAllowed:
    """The tenant's tier flag forbids this provider class."""

    tenant_id: str
    model_id: str
    provider: str
    provider_class: str
    tier: TierPolicy


@dataclass(frozen=True)
class RateLimited:
    """The tenant's token bucket is empty; retry after the suggested wait."""

    tenant_id: str
    retry_after_seconds: float


@dataclass(frozen=True)
class CeilingWouldExceed:
    """This call's ESTIMATED cost would push the month past the ceiling."""

    tenant_id: str
    estimated_cost: Decimal
    current_total: Decimal
    would_be_total: Decimal
    ceiling: Decimal


@dataclass(frozen=True)
class UnknownProvider:
    """``model_id`` is registered but its provider isn't in any class.

    A pricing entry exists; the enforcer just doesn't know how to gate
    that provider against the tier policy. Refuse rather than guess —
    surfacing the omission is preferable to silently allowing a call
    that the tenant's compliance policy may forbid.
    """

    tenant_id: str
    model_id: str
    provider: str


AuthorizeDecision = (
    Authorized
    | ProviderNotAllowed
    | RateLimited
    | CeilingWouldExceed
    | UnknownProvider
)


# ---------------------------------------------------------------------------
# Enforcer
# ---------------------------------------------------------------------------


@dataclass
class PolicyEnforcer:
    """Composes the per-tenant primitives into one authorise/record loop.

    Constructed once at sidecar startup; tenants registered via
    :meth:`set_policy` whenever the Frappe-side admin updates them.
    """

    ledger: CostLedger = field(default_factory=InMemoryCostLedger)
    rate_limiter: RateLimiter | None = None
    _policies: dict[str, TenantPolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # If no rate limiter is provided we install a permissive default;
        # tests can pass a tight one to exercise the rate-limited path.
        if self.rate_limiter is None:
            self.rate_limiter = PerTenantRateLimiter(
                capacity=1000.0, refill_per_second=100.0,
            )

    # -- policy admin ------------------------------------------------------

    def set_policy(self, policy: TenantPolicy) -> None:
        if not isinstance(policy, TenantPolicy):
            raise PolicyError(
                f"policy must be TenantPolicy; got {type(policy).__name__}"
            )
        self._policies[policy.tenant_id] = policy
        # Mirror the cost ceiling into the ledger so the ledger's
        # try_spend uses the same value the enforcer sees.
        self.ledger.set_ceiling(policy.tenant_id, policy.monthly_ceiling)

    def policy_for(self, tenant_id: str) -> TenantPolicy:
        try:
            return self._policies[tenant_id]
        except KeyError as e:
            raise UnknownTenantError(
                f"no policy registered for tenant {tenant_id!r}"
            ) from e

    # -- tier-only gate (no cost/rate side effects) ------------------------

    def provider_class_allowed(self, tenant_id: str, provider_class: str) -> bool:
        """Pure tier check: may this tenant use this ``provider_class`` now?

        Unlike :meth:`try_authorize` this consumes no rate token and touches
        no ledger — it answers only the CLAUDE §2 egress question (does the
        tenant's tier permit self-hosted / free-cloud / paid-cloud?). The
        assist fallback uses it to SKIP cloud steps for an ``ollama_only``
        tenant before any bytes leave the process.

        **Fail-closed (two ways):** (1) an unregistered tenant is treated as the
        most restrictive policy (self-hosted only); (2) an in-Kingdom (KSA)
        tenant is refused every cloud provider_class regardless of its tier —
        the residency hard-invariant (ADR-GA2). Both let assist still draft via
        Ollama, but never egress to cloud.
        """
        try:
            policy = self.policy_for(tenant_id)
        except UnknownTenantError:
            logger.warning(
                "policy: no policy for tenant %r — failing closed to self-hosted only",
                tenant_id,
            )
            return provider_class == "self-hosted"
        if not _residency_allows(policy.region, provider_class):
            logger.warning(
                "policy: residency invariant — in-Kingdom tenant %r (region %s) "
                "refused %s provider_class regardless of tier %s (ADR-GA2)",
                tenant_id, policy.region, provider_class, policy.tier.value,
            )
            return False
        return _policy_allows(policy.tier, provider_class)

    # -- authorise ---------------------------------------------------------

    def try_authorize(
        self,
        tenant_id: str,
        model_id: str,
        *,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int,
        now: datetime,
    ) -> AuthorizeDecision:
        """Decide if a call may proceed; provisionally reserve a rate-limit token.

        The check order is deliberate:
          1. **Provider gate** — fast string comparison; refuses
             policy-banned providers BEFORE we burn the rate limiter.
          2. **Cost-ceiling check** — uses the estimated cost, so a
             tenant near their cap doesn't even consume rate tokens.
          3. **Rate-limit consume** — last, because once consumed it's
             genuine quota gone.

        Caller MUST follow up with :meth:`record_spend` once the actual
        cost is known so the ledger reflects truth, not the estimate.
        """
        policy = self.policy_for(tenant_id)

        # 1. Provider gate.
        pricing = pricing_for(model_id)  # raises UnknownModelError loudly
        provider_class = _classify_provider(pricing.provider)
        if provider_class == "unknown":
            return UnknownProvider(
                tenant_id=tenant_id,
                model_id=model_id,
                provider=pricing.provider,
            )
        # Residency hard-invariant (ADR-GA2) sits ABOVE the tier gate: an
        # in-Kingdom tenant is refused every cloud provider_class regardless of
        # tier or API-key presence. Reuses ProviderNotAllowed (intake already
        # maps it to a clean denial) + a residency-specific WARNING for audit.
        residency_ok = _residency_allows(policy.region, provider_class)
        if not residency_ok:
            logger.warning(
                "policy: residency invariant — in-Kingdom tenant %r (region %s) "
                "refused %s model %r regardless of tier %s (ADR-GA2)",
                tenant_id, policy.region, provider_class, model_id, policy.tier.value,
            )
        if not residency_ok or not _policy_allows(policy.tier, provider_class):
            return ProviderNotAllowed(
                tenant_id=tenant_id,
                model_id=model_id,
                provider=pricing.provider,
                provider_class=provider_class,
                tier=policy.tier,
            )

        # 2. Cost-ceiling check using the estimate.
        estimated_cost = cost_for(
            model_id, estimated_prompt_tokens, estimated_completion_tokens,
        )
        current_total = self.ledger.period_total(tenant_id, now)
        would_be = current_total + estimated_cost
        if would_be > policy.monthly_ceiling:
            return CeilingWouldExceed(
                tenant_id=tenant_id,
                estimated_cost=estimated_cost,
                current_total=current_total,
                would_be_total=would_be,
                ceiling=policy.monthly_ceiling,
            )

        # 3. Rate-limit consume (atomic — only succeeds if quota is left).
        assert self.rate_limiter is not None  # __post_init__ guarantees
        rate_decision = self.rate_limiter.try_consume(tenant_id)
        if isinstance(rate_decision, Denied):
            return RateLimited(
                tenant_id=tenant_id,
                retry_after_seconds=rate_decision.retry_after_seconds,
            )
        assert isinstance(rate_decision, Allowed)

        return Authorized(
            tenant_id=tenant_id,
            model_id=model_id,
            provider=pricing.provider,
            estimated_cost=estimated_cost,
            period_total_estimate=would_be,
        )

    # -- record actual spend after the call -------------------------------

    def record_spend(
        self,
        *,
        tenant_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        ts: datetime,
        model_version: str = "unknown",
    ) -> Recorded | LedgerCeilingExceeded:
        """Persist the actual cost to the ledger after the call completed.

        Returns the ledger's decision so callers can detect the edge
        case where actual cost exceeded the estimate by enough to push
        past the ceiling (rare but possible — the ledger still rejects
        the entry in that case so the monthly total stays truthful).
        """
        pricing = pricing_for(model_id)
        actual_cost = cost_for(model_id, prompt_tokens, completion_tokens)
        entry = LedgerEntry(
            tenant_id=tenant_id,
            provider=pricing.provider,
            model=model_id,
            model_version=model_version,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost=actual_cost,
            ts=ts,
        )
        return self.ledger.try_spend(entry)


__all__ = (
    "Authorized",
    "AuthorizeDecision",
    "CeilingWouldExceed",
    "PolicyEnforcer",
    "PolicyError",
    "ProviderNotAllowed",
    "RateLimited",
    "TenantPolicy",
    "TierPolicy",
    "UnknownProvider",
    "UnknownTenantError",
)
