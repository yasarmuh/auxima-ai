# Copyright (c) 2026, Auxilium Tech and contributors
"""P3-01c FNOLIntakeService — the checkpointed multi-turn FNOL intake graph.

Load-bearing facts pinned here: state survives across turns per (tenant, session) thread and
tenants can NEVER resume each other's sessions; field precedence is caller-structured >
deterministic extractor > LLM; the LLM extract step is local_only ALWAYS and degrades without
crashing the turn; the amount is skippable; a completed session hands EXACTLY ONE assembled
FNOL to ClaimsCrewService.process and repeats its outcome idempotently on later turns; the
turn cap and the session LRU both fail closed.
"""
from __future__ import annotations

import pytest

from auxima_ai.claims.intake import (
	FNOLIntakeService,
	IntakeSessionExhausted,
)
from auxima_ai.claims.intake_schema import FNOLTurnRequest, ProvidedFields
from auxima_ai.claims.schema import ClaimsProcessOutcome
from auxima_ai.intake.llm import LLMResponse

TODAY = "2026-07-02"


class _StubClaims:
	"""Duck-typed ClaimsCrewService — records the assembled FNOL, returns a canned outcome."""

	def __init__(self):
		self.requests = []

	def process(self, request):
		self.requests.append(request)
		return ClaimsProcessOutcome(
			status="ok", claim_ref=request.claim_ref, audit_trail=["stub"],
		)


def _turn(session="s-1", tenant="t-intake", message="narrative", **kw) -> FNOLTurnRequest:
	return FNOLTurnRequest(
		tenant_id=tenant, session_id=session, channel="whatsapp",
		message=message, today=TODAY, **kw,
	)


def _service(invoke=None, **kw) -> tuple[FNOLIntakeService, _StubClaims]:
	claims = _StubClaims()
	return FNOLIntakeService(claims=claims, invoke=invoke, **kw), claims


class TestSingleTurnCompletion:
	def test_structured_fields_complete_in_one_turn(self):
		svc, claims = _service()
		out = svc.turn(_turn(
			message="Delivery van rear-ended at a red light",
			fields=ProvidedFields(
				loss_type="motor", incident_date="2026-06-28", estimated_amount="15000",
			),
		))
		assert out.status == "processed"
		assert out.outcome is not None and out.outcome.status == "ok"
		assert len(claims.requests) == 1
		req = claims.requests[0]
		assert req.claim_ref == "FNOL-s-1"
		assert req.loss_type == "motor"
		assert req.incident_date == "2026-06-28"
		assert req.estimated_amount == "15000"
		assert req.reported_date == TODAY
		assert "rear-ended" in req.description


class TestMultiTurn:
	def test_collects_across_turns_with_bilingual_questions(self):
		svc, claims = _service()
		out1 = svc.turn(_turn(message="Something terrible happened at our premises"))
		assert out1.status == "collecting"
		assert "loss_type" in out1.missing and "incident_date" in out1.missing
		assert out1.next_question is not None
		assert out1.next_question["en"] and out1.next_question["ar"]

		out2 = svc.turn(_turn(message="It was a fire at the warehouse, on 2026-06-28"))
		assert out2.status == "collecting"
		assert out2.missing == ["estimated_amount"]
		assert out2.next_question["field"] == "estimated_amount"
		assert out2.collected["loss_type"] == "property"
		assert out2.collected["incident_date"] == "2026-06-28"

		out3 = svc.turn(_turn(message="skip"))
		assert out3.status == "processed"
		assert claims.requests[0].estimated_amount == "0"
		# the narrative accumulated every free-text turn
		assert "terrible" in claims.requests[0].description
		assert "warehouse" in claims.requests[0].description

	def test_bare_number_accepted_when_amount_was_asked(self):
		svc, claims = _service()
		svc.turn(_turn(message="Warehouse fire on 2026-06-28"))
		out = svc.turn(_turn(message="50,000"))
		assert out.status == "processed"
		assert claims.requests[0].estimated_amount == "50000"

	def test_reported_date_pins_to_first_turn(self):
		svc, claims = _service()
		svc.turn(FNOLTurnRequest(
			tenant_id="t-intake", session_id="s-d", channel="whatsapp",
			message="A fire at the plant", today="2026-07-01",
		))
		out = svc.turn(FNOLTurnRequest(
			tenant_id="t-intake", session_id="s-d", channel="whatsapp",
			message="it happened yesterday, about SAR 20,000", today="2026-07-02",
		))
		assert out.status == "processed"
		req = claims.requests[0]
		assert req.reported_date == "2026-07-01", "reported = first contact"
		assert req.incident_date == "2026-07-01", "'yesterday' resolves against ITS turn's today"


class TestIsolationAndCaps:
	def test_tenant_scoped_sessions_never_cross(self):
		svc, _claims = _service()
		svc.turn(_turn(tenant="tenant-a", message="fire at warehouse on 2026-06-28"))
		out_b = svc.turn(_turn(tenant="tenant-b", message="hello"))
		assert out_b.turns == 1
		assert out_b.collected.get("loss_type") is None, "tenant-b must not see tenant-a's state"

	def test_turn_cap_fails_closed(self):
		svc, _claims = _service(max_turns=2)
		svc.turn(_turn(message="msg one"))
		svc.turn(_turn(message="msg two"))
		with pytest.raises(IntakeSessionExhausted):
			svc.turn(_turn(message="msg three"))

	def test_lru_evicts_oldest_session(self):
		svc, _claims = _service(max_sessions=2)
		svc.turn(_turn(session="s-a", message="fire on 2026-06-28"))
		svc.turn(_turn(session="s-b", message="msg"))
		svc.turn(_turn(session="s-c", message="msg"))  # evicts s-a
		out = svc.turn(_turn(session="s-a", message="hello again"))
		assert out.turns == 1, "evicted session restarts from scratch"

	def test_eviction_failure_never_breaks_the_triggering_turn(self):
		svc, _claims = _service(max_sessions=1)

		def boom(thread_id):
			raise RuntimeError("checkpointer backend down")

		svc.checkpointer.delete_thread = boom
		svc.turn(_turn(session="s-a", message="msg"))
		out = svc.turn(_turn(session="s-b", message="msg"))  # eviction of s-a fails
		assert out.turns == 1, "the INNOCENT user's turn proceeds despite the failed eviction"

	def test_completion_is_idempotent(self):
		svc, claims = _service()
		first = svc.turn(_turn(
			message="van crash",
			fields=ProvidedFields(
				loss_type="motor", incident_date="2026-06-28", estimated_amount="1000",
			),
		))
		again = svc.turn(_turn(message="did you get my report?"))
		assert again.status == "processed"
		assert again.outcome == first.outcome
		assert len(claims.requests) == 1, "process runs exactly once per session"


class TestProvidedFieldValidation:
	def test_future_incident_date_warned_and_ignored(self):
		svc, _claims = _service()
		out = svc.turn(_turn(
			message="fire at the site",
			fields=ProvidedFields(loss_type="property", incident_date="2026-08-01"),
		))
		assert out.status == "collecting"
		assert "incident_date" in out.missing
		assert any("incident_date" in w for w in out.warnings)


class TestLLMExtract:
	def test_llm_fills_fields_the_extractor_missed(self):
		calls = []

		def invoke(**kw):
			calls.append(kw)
			return LLMResponse(
				payload={"loss_type": "motor", "incident_date": "2026-06-28"},
				prompt_tokens=1, completion_tokens=1, latency_ms=1,
			)

		svc, _claims = _service(invoke=invoke)
		out = svc.turn(_turn(message="my Toyota got rear-ended near the exit"))
		assert out.collected["loss_type"] == "motor"
		assert out.collected["incident_date"] == "2026-06-28"
		assert len(calls) == 1
		assert calls[0]["local_only"] is True, "free-text narrative NEVER egresses"
		assert "Toyota" in calls[0]["prompt"]

	def test_llm_not_called_when_deterministic_suffices(self):
		calls = []

		def invoke(**kw):
			calls.append(kw)
			return LLMResponse(payload={}, prompt_tokens=1, completion_tokens=1, latency_ms=1)

		svc, _claims = _service(invoke=invoke)
		svc.turn(_turn(
			message="hello",
			fields=ProvidedFields(
				loss_type="motor", incident_date="2026-06-28", estimated_amount="1000",
			),
		))
		assert calls == [], "nothing missing after structured fields → no LLM call"

	def test_llm_failure_degrades_never_crashes(self):
		def invoke(**kw):
			raise RuntimeError("ollama down")

		svc, _claims = _service(invoke=invoke)
		out = svc.turn(_turn(message="my Toyota got rear-ended"))
		assert out.status == "collecting"
		assert out.degraded is True

	def test_llm_float_amount_is_decimal_normalized(self):
		# a JSON number materialises as a Python float — the stored value must be the
		# validated Decimal's string, never the raw float repr (money is Decimal, never float)
		def invoke(**kw):
			return LLMResponse(
				payload={"estimated_amount": 1500.5},
				prompt_tokens=1, completion_tokens=1, latency_ms=1,
			)

		svc, claims = _service(invoke=invoke)
		out = svc.turn(_turn(
			message="no numbers in prose",
			fields=ProvidedFields(loss_type="motor", incident_date="2026-06-28"),
		))
		assert out.status == "processed"
		assert claims.requests[0].estimated_amount == "1500.5"

	def test_llm_garbage_payload_is_dropped(self):
		def invoke(**kw):
			return LLMResponse(
				payload={"loss_type": "spaceship", "incident_date": "2099-01-01", "junk": 1},
				prompt_tokens=1, completion_tokens=1, latency_ms=1,
			)

		svc, _claims = _service(invoke=invoke)
		out = svc.turn(_turn(message="my Toyota got rear-ended"))
		assert out.status == "collecting"
		assert out.collected.get("loss_type") is None
		assert out.collected.get("incident_date") is None

	def test_llm_padding_values_are_dropped(self):
		# small-model benchmark 2026-07-02: on partial messages the LLM pads loss_type
		# "other" and TODAY's date — both indistinguishable from padding (a genuine "other"
		# reply and a genuine "today" mention are caught by the DETERMINISTIC extractor
		# before the LLM is asked), so the validator drops them and the intake re-asks.
		def invoke(**kw):
			return LLMResponse(
				payload={"loss_type": "other", "incident_date": TODAY},
				prompt_tokens=1, completion_tokens=1, latency_ms=1,
			)

		svc, _claims = _service(invoke=invoke)
		out = svc.turn(_turn(message="the total damage is around SAR 75,000"))
		assert out.status == "collecting"
		assert out.collected.get("loss_type") is None, "LLM 'other' = padding, re-ask"
		assert out.collected.get("incident_date") is None, "LLM today-date = padding, re-ask"
		assert out.collected["estimated_amount"] == "75000", "deterministic amount still lands"
