"""R5 — the assist path must enforce the CLAUDE §2 per-tenant egress gate.

These are the compliance-critical tests for GAP-AUDIT-2: an ``ollama_only``
tenant's data must NEVER reach a cloud provider, even when a cloud step is
wired and listed first. We deliberately order the cloud step BEFORE the local
step so a green test proves the GATE skipped it (not merely the chain order).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from auxima_ai.assist.schema import DraftEmailRequest
from auxima_ai.assist.service import (
	AssistService,
	DraftDegraded,
	DraftEmailSuccess,
	ProviderStep,
)
from auxima_ai.intake.llm import LLMResponse
from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy


@dataclass
class SpyCaller:
	"""Records every call; optionally fails to exercise the fallback chain."""

	payload: dict | None = None
	fail: bool = False
	calls: list[str] = field(default_factory=list)

	def call(self, *, model_id: str, prompt: str) -> LLMResponse:
		self.calls.append(model_id)
		if self.fail:
			raise RuntimeError(f"{model_id} unavailable")
		return LLMResponse(
			payload=self.payload or {"subject": "draft", "body": "body text"},
			prompt_tokens=1, completion_tokens=1, latency_ms=1, model_version=model_id,
		)


def _enforcer(tenant_id: str | None, tier: TierPolicy | None, region: str = "KSA") -> PolicyEnforcer:
	e = PolicyEnforcer()
	if tenant_id is not None and tier is not None:
		e.set_policy(TenantPolicy(
			tenant_id=tenant_id, tier=tier, region=region,
			monthly_ceiling=Decimal("100"), rate_capacity=1000.0, rate_refill_per_second=100.0,
		))
	return e


def _req(tenant_id: str) -> DraftEmailRequest:
	return DraftEmailRequest(tenant_id=tenant_id, purpose="follow up with the client")


def test_ollama_only_tenant_never_calls_cloud_even_when_cloud_is_first():
	cloud = SpyCaller(payload={"subject": "CLOUD", "body": "cloud body"})
	local = SpyCaller(payload={"subject": "LOCAL", "body": "local body"})
	svc = AssistService(
		enforcer=_enforcer("t1", TierPolicy.OLLAMA_ONLY),
		steps=[
			ProviderStep(cloud, "cloud/gemma:free", "free-cloud"),   # listed FIRST
			ProviderStep(local, "ollama/llama3.1:8b", "self-hosted"),
		],
	)
	out = svc.draft_email(_req("t1"))
	assert isinstance(out, DraftEmailSuccess)
	assert out.response.subject == "LOCAL"      # served by Ollama
	assert cloud.calls == []                    # cloud NEVER invoked
	assert local.calls == ["ollama/llama3.1:8b"]


def test_unknown_tenant_fails_closed_to_local():
	cloud = SpyCaller()
	local = SpyCaller(payload={"subject": "LOCAL", "body": "local body"})
	svc = AssistService(
		enforcer=_enforcer(None, None),  # no policy registered
		steps=[
			ProviderStep(cloud, "cloud/gemma:free", "free-cloud"),
			ProviderStep(local, "ollama/llama3.1:8b", "self-hosted"),
		],
	)
	out = svc.draft_email(_req("ghost-tenant"))
	assert isinstance(out, DraftEmailSuccess)
	assert cloud.calls == []  # fail-closed: unconfirmed tenant gets local only


def test_free_cloud_tenant_falls_through_to_cloud_when_ollama_down():
	# region="INTL": cloud fallthrough is only lawful for a non-in-Kingdom
	# tenant. A KSA tenant would be residency-blocked (test_residency_invariant).
	local = SpyCaller(fail=True)
	cloud = SpyCaller(payload={"subject": "CLOUD", "body": "cloud body"})
	svc = AssistService(
		enforcer=_enforcer("t2", TierPolicy.OLLAMA_THEN_FREE_CLOUD, region="INTL"),
		steps=[
			ProviderStep(local, "ollama/llama3.1:8b", "self-hosted"),  # Ollama-first, fails
			ProviderStep(cloud, "cloud/gemma:free", "free-cloud"),
		],
	)
	out = svc.draft_email(_req("t2"))
	assert isinstance(out, DraftEmailSuccess)
	assert out.response.subject == "CLOUD"
	assert local.calls == ["ollama/llama3.1:8b"]  # tried first
	assert cloud.calls == ["cloud/gemma:free"]     # tier permits the fallback


def test_ksa_tenant_on_cloud_tier_is_residency_blocked_from_cloud():
	"""ADR-GA2 fix #1 at the service layer: even MISCONFIGURED to a paid-cloud
	tier, a KSA (in-Kingdom) tenant's assist call skips every cloud step and is
	served in-Kingdom by Ollama (residency hard-invariant above the tier flag)."""
	cloud = SpyCaller(payload={"subject": "CLOUD", "body": "cloud body"})
	local = SpyCaller(payload={"subject": "LOCAL", "body": "local body"})
	svc = AssistService(
		enforcer=_enforcer("ksa-x", TierPolicy.OLLAMA_THEN_PAID_CLOUD, region="KSA"),
		steps=[
			ProviderStep(cloud, "cloud/gemma:free", "free-cloud"),   # listed FIRST
			ProviderStep(local, "ollama/llama3.1:8b", "self-hosted"),
		],
	)
	out = svc.draft_email(_req("ksa-x"))
	assert isinstance(out, DraftEmailSuccess)
	assert out.response.subject == "LOCAL"     # served in-Kingdom by Ollama
	assert cloud.calls == []                   # residency-blocked despite paid-cloud tier


def test_ollama_only_with_only_a_cloud_step_degrades_cleanly():
	cloud = SpyCaller()
	svc = AssistService(
		enforcer=_enforcer("t3", TierPolicy.OLLAMA_ONLY),
		steps=[ProviderStep(cloud, "cloud/gemma:free", "free-cloud")],
	)
	out = svc.draft_email(_req("t3"))
	assert isinstance(out, DraftDegraded)  # all steps policy-skipped
	assert cloud.calls == []
