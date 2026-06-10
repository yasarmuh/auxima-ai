"""H-2 (audit 2026-06-10): the assist path must enforce rate-limit + AI-Run-Log + cost ceiling.

Before this, AssistService._invoke ran the LLM with NO rate limit (Ollama-only tenants could DoS
the local GPU; approved tenants could spam cloud), NO cost-ceiling gate, and emitted NO AI Run
Log row. These pin the parallel lightweight gate (design B): the explicit provider_class tag is
unchanged; the gate reuses the enforcer's model-INDEPENDENT primitives (rate limiter + ledger
period total) and emits one activity row per served call.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from auxima_ai.activity.row import ActivityRow
from auxima_ai.assist.schema import WordingDiffRequest, WordingOffer
from auxima_ai.assist.service import AssistService, DraftDegraded, ProviderStep, WordingDiffSuccess
from auxima_ai.cost.ledger import InMemoryCostLedger
from auxima_ai.intake.llm import LLMResponse
from auxima_ai.policy.enforcer import PolicyEnforcer, TenantPolicy, TierPolicy
from auxima_ai.ratelimit.bucket import PerTenantRateLimiter

_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
_WORDING_PAYLOAD = {"differences": ["flood differs"], "flags": []}


class _Caller:
	def __init__(self, payload: dict[str, Any], fail: bool = False) -> None:
		self.payload, self.fail, self.calls = payload, fail, []

	def call(self, *, model_id: str, prompt: str) -> LLMResponse:
		self.calls.append(model_id)
		if self.fail:
			raise RuntimeError(f"{model_id} down")
		return LLMResponse(payload=self.payload, prompt_tokens=3, completion_tokens=2, latency_ms=4, model_version=model_id)


class _CapturingEmitter:
	def __init__(self) -> None:
		self.rows: list[ActivityRow] = []

	def emit(self, row: ActivityRow) -> None:
		self.rows.append(row)


def _wording_req() -> WordingDiffRequest:
	return WordingDiffRequest(
		tenant_id="t1",
		offers=[WordingOffer(insurer="A", wording="flood excluded"), WordingOffer(insurer="B", wording="flood included")],
	)


def _enforcer(*, rate_capacity: float = 1000.0, ledger: InMemoryCostLedger | None = None, region: str = "INTL") -> PolicyEnforcer:
	e = PolicyEnforcer(
		rate_limiter=PerTenantRateLimiter(capacity=rate_capacity, refill_per_second=0.001),
		ledger=ledger or InMemoryCostLedger(),
	)
	e.set_policy(TenantPolicy(
		tenant_id="t1", tier=TierPolicy.OLLAMA_THEN_PAID_CLOUD, region=region,
		cloud_egress_approved=True, monthly_ceiling=Decimal("100"),
		rate_capacity=rate_capacity, rate_refill_per_second=0.001,
	))
	return e


def _steps(local: _Caller, cloud: _Caller) -> list[ProviderStep]:
	return [
		ProviderStep(local, "ollama/llama3.1:8b", "self-hosted"),
		ProviderStep(cloud, "openrouter/x:free", "paid-cloud"),
	]


# --- rate limit ---------------------------------------------------------------

def test_rate_limited_call_degrades_and_does_not_invoke_any_provider():
	# capacity 1, no refill: the FIRST call consumes the only token; the SECOND is rate-limited.
	local, cloud = _Caller(_WORDING_PAYLOAD), _Caller(_WORDING_PAYLOAD)
	svc = AssistService(enforcer=_enforcer(rate_capacity=1.0), steps=_steps(local, cloud))
	first = svc.wording_diff(_wording_req())
	assert isinstance(first, WordingDiffSuccess)
	local.calls.clear()
	second = svc.wording_diff(_wording_req())
	assert isinstance(second, DraftDegraded)      # rate-limited -> clean degrade
	assert local.calls == [] and cloud.calls == []  # no provider invoked at all


def test_every_call_consumes_a_rate_token_even_all_local():
	# Ollama-only-ish: local serves, but the rate token is still consumed (local-GPU DoS bound).
	local, cloud = _Caller(_WORDING_PAYLOAD), _Caller(_WORDING_PAYLOAD)
	svc = AssistService(enforcer=_enforcer(rate_capacity=1.0), steps=_steps(local, cloud))
	assert isinstance(svc.wording_diff(_wording_req()), WordingDiffSuccess)  # 1st ok (local)
	assert isinstance(svc.wording_diff(_wording_req()), DraftDegraded)        # 2nd rate-limited


# --- cost ceiling -------------------------------------------------------------

def test_over_ceiling_skips_cloud_but_serves_local():
	# Pre-load the ledger to the ceiling; the cloud step is skipped, local still serves.
	ledger = InMemoryCostLedger()
	ledger.set_ceiling("t1", Decimal("100"))
	from auxima_ai.cost.ledger import LedgerEntry
	ledger.try_spend(LedgerEntry(
		tenant_id="t1", provider="openrouter", model="x", model_version="v",
		prompt_tokens=1, completion_tokens=1, latency_ms=1, cost=Decimal("100"), ts=_NOW,
	))
	local, cloud = _Caller(_WORDING_PAYLOAD), _Caller(_WORDING_PAYLOAD)
	svc = AssistService(enforcer=_enforcer(ledger=ledger), steps=_steps(local, cloud))
	out = svc._invoke(tenant_id="t1", model_id="ollama/llama3.1:8b",
	                  prompt="compare A vs B flood", now=_NOW)
	assert out.model_version == "ollama/llama3.1:8b"  # served locally
	assert cloud.calls == []                           # cloud skipped (over ceiling)


def test_over_ceiling_with_local_down_degrades():
	ledger = InMemoryCostLedger()
	ledger.set_ceiling("t1", Decimal("100"))
	from auxima_ai.cost.ledger import LedgerEntry
	ledger.try_spend(LedgerEntry(
		tenant_id="t1", provider="openrouter", model="x", model_version="v",
		prompt_tokens=1, completion_tokens=1, latency_ms=1, cost=Decimal("100"), ts=_NOW,
	))
	local, cloud = _Caller(_WORDING_PAYLOAD, fail=True), _Caller(_WORDING_PAYLOAD)
	svc = AssistService(enforcer=_enforcer(ledger=ledger), steps=_steps(local, cloud))
	out = svc.wording_diff(_wording_req())
	assert isinstance(out, DraftDegraded)  # local down + cloud ceiling-blocked -> degrade
	assert cloud.calls == []


# --- AI Run Log ---------------------------------------------------------------

def test_served_call_emits_one_activity_row():
	emitter = _CapturingEmitter()
	local, cloud = _Caller(_WORDING_PAYLOAD), _Caller(_WORDING_PAYLOAD)
	svc = AssistService(enforcer=_enforcer(), steps=_steps(local, cloud), activity_emitter=emitter)
	assert isinstance(svc.wording_diff(_wording_req()), WordingDiffSuccess)
	assert len(emitter.rows) == 1
	row = emitter.rows[0]
	assert row.tenant_id == "t1"
	assert row.payload["model"] == "ollama/llama3.1:8b"
	assert row.payload["provider_class"] == "self-hosted"


def test_degraded_call_emits_no_activity_row():
	emitter = _CapturingEmitter()
	local, cloud = _Caller(_WORDING_PAYLOAD, fail=True), _Caller(_WORDING_PAYLOAD, fail=True)
	svc = AssistService(enforcer=_enforcer(), steps=_steps(local, cloud), activity_emitter=emitter)
	assert isinstance(svc.wording_diff(_wording_req()), DraftDegraded)
	assert emitter.rows == []  # nothing served -> nothing logged
