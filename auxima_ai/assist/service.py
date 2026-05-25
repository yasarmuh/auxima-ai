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
	build_draft_note_prompt,
	build_suggest_fields_prompt,
	validate_draft_email_response,
	validate_draft_note_response,
	validate_suggest_fields_response,
)
from auxima_ai.assist.schema import (
	DraftEmailRequest,
	DraftEmailResponse,
	DraftNoteRequest,
	DraftNoteResponse,
	SuggestFieldsRequest,
	SuggestFieldsResponse,
)
from auxima_ai.intake.llm import LLMCaller, LLMResponse, StubLLMCaller
from auxima_ai.observability.log import emit
from auxima_ai.observability.redact import redact
from auxima_ai.policy.enforcer import PolicyEnforcer

logger = logging.getLogger(__name__)

#: Logical fallback model id for the legacy single-``llm`` path (no enforcer
#: wired). CLAUDE §2 default = self-hosted Ollama; the policy-gated production
#: path uses the per-:class:`ProviderStep` model id, not this.
DEFAULT_MODEL_ID = "ollama/llama3.1:8b"


@dataclass(frozen=True)
class ProviderStep:
    """One ordered provider in the policy-gated assist fallback chain.

    ``provider_class`` is the CLAUDE §2 egress class — ``self-hosted`` /
    ``free-cloud`` / ``paid-cloud`` — tagged explicitly at construction
    (in :func:`auxima_ai.bootstrap.build_assist_service`, which knows which
    caller is local vs cloud) rather than guessed from the model-id string.
    The enforcer gates each step on this class.
    """

    caller: LLMCaller
    model_id: str
    provider_class: str


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


@dataclass(frozen=True)
class DraftNoteSuccess:
	response: DraftNoteResponse


NoteOutcome = DraftNoteSuccess | DraftDegraded | DraftSchemaInvalid


@dataclass(frozen=True)
class SuggestFieldsSuccess:
	response: SuggestFieldsResponse


SuggestOutcome = SuggestFieldsSuccess | DraftDegraded | DraftSchemaInvalid


@dataclass
class AssistService:
	"""Bundles the LLM caller the draft pipeline needs (injected for tests).

	Two modes:
	  * **Policy mode** (production) — ``enforcer`` + ``steps`` are set. Each
	    draft tries the ordered steps, SKIPPING any whose ``provider_class``
	    the tenant's tier forbids, so an ``ollama_only`` tenant never reaches a
	    cloud step (CLAUDE §2). First success wins; all-skipped/all-failed
	    degrades cleanly.
	  * **Legacy mode** (tests / single-provider) — only ``llm`` is set; the
	    single caller is used with no policy gate.
	"""

	llm: LLMCaller = field(default_factory=StubLLMCaller)
	enforcer: PolicyEnforcer | None = None
	steps: list[ProviderStep] | None = None

	def _invoke(self, *, tenant_id: str, model_id: str, prompt: str) -> LLMResponse:
		"""Call the LLM, enforcing per-tenant provider-class policy when wired.

		Raises :class:`AllProvidersUnavailable` if every allowed step fails or
		all steps are policy-skipped — each public method already maps that to
		a clean ``DraftDegraded``.
		"""
		if self.enforcer is not None and self.steps is not None:
			errors: list[tuple[str, str]] = []
			for step in self.steps:
				if not self.enforcer.provider_class_allowed(tenant_id, step.provider_class):
					logger.info(
						"assist: tenant %s tier forbids provider_class %s — skipping step %s",
						tenant_id, step.provider_class, step.model_id,
					)
					continue
				# R3 — data minimisation before cloud egress (GDPR/PDPL): local
				# (self-hosted) steps get the FULL prompt (best quality, no egress);
				# any cloud step gets structured identifiers (email/phone/national-id/
				# CR/IBAN) redacted first. NOTE: redact.py is regex-based and does NOT
				# remove names/company (no NER) — those still reach an opted-in cloud
				# tier; that residual is flagged for R7/counsel.
				step_prompt = prompt
				if step.provider_class != "self-hosted":
					step_prompt, fired = redact(prompt)
					if fired:
						logger.info(
							"assist: redacted structured PII before cloud egress to %s (tenant %s)",
							step.model_id, tenant_id,
						)
				try:
					return step.caller.call(model_id=step.model_id, prompt=step_prompt)
				except Exception as e:  # noqa: BLE001 - any failure advances the chain
					errors.append((step.model_id, f"{type(e).__name__}: {e}"))
					logger.warning("assist provider %s failed, trying next: %s", step.model_id, e)
					continue
			raise AllProvidersUnavailable(errors)
		return self.llm.call(model_id=model_id, prompt=prompt)

	def draft_email(self, request: DraftEmailRequest) -> DraftOutcome:
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_draft_email_prompt(request)

		try:
			llm_response = self._invoke(tenant_id=request.tenant_id, model_id=model_id, prompt=prompt)
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

	def draft_note(self, request: DraftNoteRequest) -> NoteOutcome:
		"""Draft a short note/comment/error-help text; degrade cleanly."""
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_draft_note_prompt(request)

		try:
			llm_response = self._invoke(tenant_id=request.tenant_id, model_id=model_id, prompt=prompt)
		except AllProvidersUnavailable as e:
			emit(
				"warn", "assist.draft_note.degraded",
				fields={"tenant_id": request.tenant_id, "kind": request.kind, "reason": str(e)[:200]},
			)
			return DraftDegraded(reason=str(e))

		try:
			fields = validate_draft_note_response(llm_response.payload)
		except SchemaViolationError as e:
			emit(
				"warn", "assist.draft_note.schema_violation",
				fields={"tenant_id": request.tenant_id, "kind": request.kind, "error_count": len(e.errors)},
			)
			return DraftSchemaInvalid(errors=tuple(e.errors))

		response = DraftNoteResponse(
			text=fields.text,
			kind=request.kind,
			language=request.language,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.draft_note.completed",
			fields={
				"tenant_id": request.tenant_id, "kind": request.kind,
				"language": request.language, "model_version": response.model_version,
			},
		)
		return DraftNoteSuccess(response=response)

	def suggest_fields(self, request: SuggestFieldsRequest) -> SuggestOutcome:
		"""Suggest values for empty fields; degrade cleanly. Suggestion-only."""
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_suggest_fields_prompt(request)
		allowed = {f.fieldname for f in request.fields}

		try:
			llm_response = self._invoke(tenant_id=request.tenant_id, model_id=model_id, prompt=prompt)
		except AllProvidersUnavailable as e:
			emit(
				"warn", "assist.suggest_fields.degraded",
				fields={"tenant_id": request.tenant_id, "doctype": request.doctype, "reason": str(e)[:200]},
			)
			return DraftDegraded(reason=str(e))

		try:
			suggestions = validate_suggest_fields_response(llm_response.payload, allowed)
		except SchemaViolationError as e:
			emit(
				"warn", "assist.suggest_fields.schema_violation",
				fields={"tenant_id": request.tenant_id, "doctype": request.doctype, "error_count": len(e.errors)},
			)
			return DraftSchemaInvalid(errors=tuple(e.errors))

		response = SuggestFieldsResponse(
			suggestions=suggestions,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.suggest_fields.completed",
			fields={
				"tenant_id": request.tenant_id, "doctype": request.doctype,
				"suggested": len(suggestions), "requested": len(allowed),
				"model_version": response.model_version,
			},
		)
		return SuggestFieldsSuccess(response=response)


__all__ = (
	"AssistService",
	"DEFAULT_MODEL_ID",
	"ProviderStep",
	"DraftDegraded",
	"DraftEmailSuccess",
	"DraftNoteSuccess",
	"DraftOutcome",
	"DraftSchemaInvalid",
	"NoteOutcome",
	"SuggestFieldsSuccess",
	"SuggestOutcome",
)
