"""R3 — data minimisation before cloud egress (GDPR/PDPL).

Local (self-hosted) steps must receive the FULL prompt (best draft quality, no
egress). Cloud steps must receive a prompt with structured identifiers
(email / phone / national-id / CR / IBAN) redacted first.

Known residual (documented, not tested): redact.py is regex-based and does NOT
remove names/company (no NER); those still reach an opted-in cloud tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from auxima_ai.assist.schema import WordingDiffRequest, WordingOffer
from auxima_ai.assist.service import AssistService, ProviderStep, WordingDiffSuccess
from auxima_ai.intake.llm import LLMResponse
from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy

_EMAIL = "ops@acme.example"
_KSA_PHONE = "0500123456"
_E164 = "+966500000000"

# The redaction-before-cloud contract is a property of `_invoke`, exercised here via the one
# deliberately cloud-eligible endpoint (wording_diff). The personal-free-text endpoints
# (draft_email, …) are pinned self-hosted-only (H-1) and never reach a cloud step, so they
# can no longer serve as the vehicle for a cloud-redaction test.


@dataclass
class PromptSpy:
	"""Captures the prompt each call receives; optionally fails to advance the chain."""

	fail: bool = False
	prompts: list[str] = field(default_factory=list)

	def call(self, *, model_id: str, prompt: str) -> LLMResponse:
		self.prompts.append(prompt)
		if self.fail:
			raise RuntimeError(f"{model_id} unavailable")
		return LLMResponse(
			payload={"differences": ["flood cover differs"], "flags": []},
			prompt_tokens=1, completion_tokens=1, latency_ms=1, model_version=model_id,
		)


def _req(tenant_id: str) -> WordingDiffRequest:
	# Identifiers embedded in the wording text drive the redaction assertions.
	return WordingDiffRequest(
		tenant_id=tenant_id,
		offers=[
			WordingOffer(insurer="A", wording=f"contact {_EMAIL} or {_KSA_PHONE} / {_E164}; flood excluded"),
			WordingOffer(insurer="B", wording="flood included"),
		],
	)


def _enforcer(tenant_id: str, tier: TierPolicy, region: str = "KSA") -> PolicyEnforcer:
	e = PolicyEnforcer()
	e.set_policy(TenantPolicy(
		tenant_id=tenant_id, tier=tier, region=region,
		monthly_ceiling=Decimal("100"), rate_capacity=1000.0, rate_refill_per_second=100.0,
	))
	return e


def test_cloud_step_receives_redacted_prompt():
	local = PromptSpy(fail=True)        # Ollama down -> chain advances to cloud
	cloud = PromptSpy()
	svc = AssistService(
		# region="INTL": a cloud step is only reachable for a non-in-Kingdom
		# tenant; a KSA tenant is residency-blocked (test_residency_invariant).
		enforcer=_enforcer("t1", TierPolicy.OLLAMA_THEN_FREE_CLOUD, region="INTL"),
		steps=[
			ProviderStep(local, "ollama/llama3.1:8b", "self-hosted"),
			ProviderStep(cloud, "cloud/gemma:free", "free-cloud"),
		],
	)
	out = svc.wording_diff(_req("t1"))
	assert isinstance(out, WordingDiffSuccess)
	sent = cloud.prompts[0]
	# Raw identifiers must NOT have left the process.
	assert _EMAIL not in sent
	assert _KSA_PHONE not in sent
	assert _E164 not in sent
	# Replaced by typed placeholders.
	assert "<redacted:email>" in sent
	assert "<redacted:phone_ksa_local>" in sent
	assert "<redacted:phone_e164>" in sent


def test_local_step_receives_full_unredacted_prompt():
	"""Best quality on the in-Kingdom path: no redaction for self-hosted."""
	local = PromptSpy()
	svc = AssistService(
		enforcer=_enforcer("t2", TierPolicy.OLLAMA_THEN_FREE_CLOUD),
		steps=[ProviderStep(local, "ollama/llama3.1:8b", "self-hosted")],
	)
	out = svc.wording_diff(_req("t2"))
	assert isinstance(out, WordingDiffSuccess)
	sent = local.prompts[0]
	assert _EMAIL in sent          # full context preserved locally
	assert _KSA_PHONE in sent
	assert "<redacted:" not in sent
