"""Data-minimisation by design (GDPR Art 25 / PDPL): the free-text agents that can carry
special-category data (a D&N call transcript may contain HEALTH information for medical lines;
a SoV may carry insured names/addresses) are pinned to **self-hosted-only** egress.

The regex redactor removes only structured identifiers (email/phone/national-id/CR/IBAN) — it has
no NER, so it cannot strip health narrative or names. Rather than rely on it for unstructured
sensitive text, ``summarise_dn`` and ``extract_sov`` refuse cloud egress entirely: a raw transcript
or schedule never leaves to a cloud provider even for a ``cloud_egress_approved`` tenant. If no
self-hosted model is available the call degrades cleanly (the broker records it manually).

Contrast: ``wording_diff`` compares insurer policy WORDING (commercial template language, not personal
data) and stays cloud-eligible — a deliberate boundary, asserted below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from auxima_ai.assist.schema import (
	DNSummaryRequest,
	SoVExtractRequest,
	WordingDiffRequest,
	WordingOffer,
)
from auxima_ai.assist.service import (
	AssistService,
	DNSummarySuccess,
	DraftDegraded,
	ProviderStep,
	SoVExtractSuccess,
	WordingDiffSuccess,
)
from auxima_ai.intake.llm import LLMResponse
from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy


@dataclass
class SpyCaller:
	"""Records prompts; returns a fixed payload or fails to advance the fallback chain."""

	payload: dict[str, Any]
	fail: bool = False
	prompts: list[str] = field(default_factory=list)

	def call(self, *, model_id: str, prompt: str) -> LLMResponse:
		self.prompts.append(prompt)
		if self.fail:
			raise RuntimeError(f"{model_id} unavailable")
		return LLMResponse(
			payload=self.payload, prompt_tokens=1, completion_tokens=1, latency_ms=1, model_version=model_id,
		)


_DN_PAYLOAD = {"needs": ["chronic care cover"], "coverage_gaps": []}
_SOV_PAYLOAD = {"line_items": [{"description": "warehouse", "value": 1.0, "category": None}], "total_value": 1.0}
_WORDING_PAYLOAD = {"differences": ["flood excluded vs included"], "flags": []}


def _cloud_tenant() -> PolicyEnforcer:
	"""A tenant that IS allowed cloud egress (non-in-Kingdom + paid-cloud tier + approved)."""
	e = PolicyEnforcer()
	e.set_policy(TenantPolicy(
		tenant_id="cloudy", tier=TierPolicy.OLLAMA_THEN_PAID_CLOUD, region="INTL",
		cloud_egress_approved=True,
		monthly_ceiling=Decimal("100"), rate_capacity=1000.0, rate_refill_per_second=100.0,
	))
	return e


def _steps(local: SpyCaller, cloud: SpyCaller) -> list[ProviderStep]:
	return [
		ProviderStep(local, "ollama/llama3.1:8b", "self-hosted"),
		ProviderStep(cloud, "openrouter/some-model", "paid-cloud"),
	]


# --- D&N summariser: health-bearing transcript is pinned to self-hosted ---------------------

def test_dn_summary_never_egresses_to_cloud_even_when_local_fails():
	"""Self-hosted DOWN + a cloud-allowed tenant: a normal agent would fall through to cloud.
	The D&N summariser must NOT — the transcript stays in-Kingdom; the call degrades instead."""
	local = SpyCaller(_DN_PAYLOAD, fail=True)
	cloud = SpyCaller(_DN_PAYLOAD)
	svc = AssistService(enforcer=_cloud_tenant(), steps=_steps(local, cloud))
	req = DNSummaryRequest(tenant_id="cloudy", transcript="patient has diabetes, needs chronic cover")
	out = svc.summarise_dn(req)
	assert isinstance(out, DraftDegraded)          # degraded, not silently sent to cloud
	assert cloud.prompts == []                      # the transcript never left to cloud


def test_dn_summary_uses_self_hosted_when_available():
	local = SpyCaller(_DN_PAYLOAD)
	cloud = SpyCaller(_DN_PAYLOAD)
	svc = AssistService(enforcer=_cloud_tenant(), steps=_steps(local, cloud))
	out = svc.summarise_dn(DNSummaryRequest(tenant_id="cloudy", transcript="some notes"))
	assert isinstance(out, DNSummarySuccess)
	assert len(local.prompts) == 1
	assert cloud.prompts == []                      # cloud never touched


# --- SoV extraction: same pin -----------------------------------------------------------------

def test_sov_extract_never_egresses_to_cloud():
	local = SpyCaller(_SOV_PAYLOAD, fail=True)
	cloud = SpyCaller(_SOV_PAYLOAD)
	svc = AssistService(enforcer=_cloud_tenant(), steps=_steps(local, cloud))
	out = svc.extract_sov(SoVExtractRequest(tenant_id="cloudy", text="warehouse 25m, owner Mr X, Riyadh"))
	assert isinstance(out, DraftDegraded)
	assert cloud.prompts == []


# --- Wording-diff: deliberately NOT pinned (commercial template text, not personal data) -------

def test_wording_diff_stays_cloud_eligible():
	"""Documents the boundary: policy WORDING is not personal data, so the wording-diff agent
	may fall through to an approved cloud provider when the local model is down."""
	local = SpyCaller(_WORDING_PAYLOAD, fail=True)
	cloud = SpyCaller(_WORDING_PAYLOAD)
	svc = AssistService(enforcer=_cloud_tenant(), steps=_steps(local, cloud))
	req = WordingDiffRequest(
		tenant_id="cloudy",
		offers=[WordingOffer(insurer="A", wording="flood excluded"), WordingOffer(insurer="B", wording="flood included")],
	)
	out = svc.wording_diff(req)
	assert isinstance(out, WordingDiffSuccess)
	assert len(cloud.prompts) == 1                  # fell through to cloud — allowed for wording
