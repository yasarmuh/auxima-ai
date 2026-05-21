"""assist.draft-email orchestration — pure-Python, returns a typed outcome.

Deliberately leaner than the intake pipeline: a draft is a cheap, best-effort
*suggestion*, so v1 skips the idempotency/ledger machinery. The one hard rule
is **graceful degradation** — if every model is unavailable the caller gets a
clean ``DraftDegraded`` (the Frappe composer then just opens empty), never a
500 and never a blocked workflow.

(Cost-ceiling gating via the policy enforcer is a deliberate follow-up — wire
it once usage is metered; free/local models make it ~zero in dev.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from auxima_ai.assist.fallback import AllProvidersUnavailable
from auxima_ai.assist.prompts import (
	SchemaViolationError,
	build_draft_email_prompt,
	validate_draft_email_response,
)
from auxima_ai.assist.schema import DraftEmailRequest, DraftEmailResponse
from auxima_ai.intake.llm import LLMCaller, StubLLMCaller
from auxima_ai.observability.log import emit

logger = logging.getLogger(__name__)

#: default dev model chain is configured at the FallbackLLMCaller; this is the
#: logical id passed through when the caller is a single provider / the stub.
DEFAULT_MODEL_ID = "openrouter/google/gemma-4-31b-it:free"


@dataclass(frozen=True)
class DraftEmailSuccess:
	response: DraftEmailResponse


@dataclass(frozen=True)
class DraftDegraded:
	"""Every model was unavailable (rate-limited/down). UI composes manually."""

	reason: str


@dataclass(frozen=True)
class DraftSchemaInvalid:
	"""A model replied but not in the {subject, body} shape — upstream issue."""

	errors: tuple[dict, ...]


DraftOutcome = DraftEmailSuccess | DraftDegraded | DraftSchemaInvalid


@dataclass
class AssistService:
	"""Bundles the LLM caller the draft pipeline needs (injected for tests)."""

	llm: LLMCaller = field(default_factory=StubLLMCaller)

	def draft_email(self, request: DraftEmailRequest) -> DraftOutcome:
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_draft_email_prompt(request)

		try:
			llm_response = self.llm.call(model_id=model_id, prompt=prompt)
		except AllProvidersUnavailable as e:
			emit(
				"warn", "assist.draft_email.degraded",
				fields={"tenant_id": request.tenant_id, "reason": str(e)[:200]},
			)
			return DraftDegraded(reason=str(e))

		try:
			fields = validate_draft_email_response(llm_response.payload)
		except SchemaViolationError as e:
			emit(
				"warn", "assist.draft_email.schema_violation",
				fields={"tenant_id": request.tenant_id, "error_count": len(e.errors)},
			)
			return DraftSchemaInvalid(errors=tuple(e.errors))

		response = DraftEmailResponse(
			subject=fields.subject,
			body=fields.body,
			language=request.language,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.draft_email.completed",
			fields={
				"tenant_id": request.tenant_id,
				"language": request.language,
				"model_version": response.model_version,
				"examples_used": len(request.examples),
				"tokens": response.prompt_tokens + response.completion_tokens,
			},
		)
		return DraftEmailSuccess(response=response)


__all__ = (
	"AssistService",
	"DEFAULT_MODEL_ID",
	"DraftDegraded",
	"DraftEmailSuccess",
	"DraftOutcome",
	"DraftSchemaInvalid",
)
