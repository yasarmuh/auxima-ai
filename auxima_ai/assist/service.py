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
from datetime import datetime, timezone

from auxima_ai.activity.row import RetentionClass, build_activity_row
from auxima_ai.assist.fallback import AllProvidersUnavailable
from auxima_ai.assist.prompts import (
	SchemaViolationError,
	build_draft_email_prompt,
	build_draft_note_prompt,
	build_dn_summary_prompt,
	build_intent_classify_prompt,
	build_policy_ingest_prompt,
	build_recommendation_prompt,
	build_renewal_draft_prompt,
	build_sov_extract_prompt,
	build_suggest_fields_prompt,
	build_wording_diff_prompt,
	validate_dn_summary_response,
	validate_draft_email_response,
	validate_draft_note_response,
	validate_intent_classify_response,
	validate_policy_ingest_response,
	validate_recommendation_response,
	validate_renewal_draft_response,
	validate_sov_extract_response,
	validate_suggest_fields_response,
	validate_wording_diff_response,
)
from auxima_ai.assist.schema import (
	DNSummaryRequest,
	DNSummaryResponse,
	DraftEmailRequest,
	DraftEmailResponse,
	DraftNoteRequest,
	DraftNoteResponse,
	IntentClassifyRequest,
	IntentClassifyResponse,
	LegalCheck,
	PolicyIngestRequest,
	PolicyIngestResponse,
	RenewalDraftRequest,
	RenewalDraftResponse,
	SoVExtractRequest,
	SoVExtractResponse,
	RecommendationFields,
	RecommendationRequest,
	RecommendationResponse,
	SuggestFieldsRequest,
	SuggestFieldsResponse,
	WordingDiffRequest,
	WordingDiffResponse,
)
from auxima_ai.intake.llm import LLMCaller, LLMResponse, StubLLMCaller
from auxima_ai.intake.service import ActivityEmitter, NullActivityEmitter
from auxima_ai.observability.log import emit
from auxima_ai.observability.redact import redact
from auxima_ai.policy.enforcer import PolicyEnforcer
from auxima_ai.ratelimit.bucket import Denied

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


@dataclass(frozen=True)
class RecommendationSuccess:
	response: RecommendationResponse


RecommendationOutcome = RecommendationSuccess | DraftDegraded | DraftSchemaInvalid


@dataclass(frozen=True)
class WordingDiffSuccess:
	response: WordingDiffResponse


WordingDiffOutcome = WordingDiffSuccess | DraftDegraded | DraftSchemaInvalid


@dataclass(frozen=True)
class DNSummarySuccess:
	response: DNSummaryResponse


DNSummaryOutcome = DNSummarySuccess | DraftDegraded | DraftSchemaInvalid


@dataclass(frozen=True)
class SoVExtractSuccess:
	response: SoVExtractResponse


SoVExtractOutcome = SoVExtractSuccess | DraftDegraded | DraftSchemaInvalid


@dataclass(frozen=True)
class IntentClassifySuccess:
	response: IntentClassifyResponse


IntentClassifyOutcome = IntentClassifySuccess | DraftDegraded | DraftSchemaInvalid


@dataclass(frozen=True)
class PolicyIngestSuccess:
	response: PolicyIngestResponse


PolicyIngestOutcome = PolicyIngestSuccess | DraftDegraded | DraftSchemaInvalid


@dataclass(frozen=True)
class RenewalDraftSuccess:
	response: RenewalDraftResponse


RenewalDraftOutcome = RenewalDraftSuccess | DraftDegraded | DraftSchemaInvalid


def _fmt_pct(p: float) -> str:
	"""Trim trailing zeros: 12.5 -> '12.5', 15.0 -> '15'."""
	return f"{p:.4f}".rstrip("0").rstrip(".")


# Mandatory legal block, appended deterministically (never LLM-generated). Both languages keep the
# literal "9(b)" article token so the legal-line check is language-agnostic.
_LEGAL_BLOCK = {
	"en": (
		"\n\n---\n*Commission disclosed: {pct}% (within the Appendix A cap). This recommendation is "
		"prepared under Insurance Authority Implementing Regulations Article 9(b); the comparison "
		"artefact and supporting records are retained for {years} years per Article 24.*"
	),
	"ar": (
		"\n\n---\n*الإفصاح عن العمولة: {pct}% (ضمن حد الملحق أ). أُعدّت هذه التوصية وفقًا للمادة 9(b) "
		"من اللائحة التنفيذية لهيئة التأمين؛ ويُحتفظ بوثيقة المقارنة والسجلات الداعمة لمدة {years} "
		"سنوات وفقًا للمادة 24.*"
	),
}
_WHY_HEADER = {"en": "**Why {insurer}:**", "ar": "**لماذا {insurer}:**"}


def _assemble_recommendation_body(req: RecommendationRequest, fields: RecommendationFields) -> str:
	"""LLM reasoning + citation bullets + the deterministic legal block."""
	header = _WHY_HEADER.get(req.language, _WHY_HEADER["en"]).format(insurer=req.recommended_insurer)
	bullets = "\n".join(f"- {c}" for c in fields.citations)
	legal = _LEGAL_BLOCK.get(req.language, _LEGAL_BLOCK["en"]).format(
		pct=_fmt_pct(req.commission_pct), years=req.retention_years
	)
	return f"{fields.reasoning}\n\n{header}\n{bullets}{legal}"


def _legal_check(req: RecommendationRequest, body_md: str) -> LegalCheck:
	"""Verify the assembled note names the insurer, discloses the commission %, and cites Art 9(b)."""
	insurer_named = req.recommended_insurer in body_md
	commission_disclosed = f"{_fmt_pct(req.commission_pct)}%" in body_md
	art9b_present = "9(b)" in body_md
	return LegalCheck(
		insurer_named=insurer_named,
		commission_disclosed=commission_disclosed,
		art9b_present=art9b_present,
		passed=insurer_named and commission_disclosed and art9b_present,
	)


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
	activity_emitter: ActivityEmitter = field(default_factory=lambda: NullActivityEmitter())

	def _invoke(
		self, *, tenant_id: str, model_id: str, prompt: str, local_only: bool = False,
		now: datetime | None = None,
	) -> LLMResponse:
		"""Call the LLM, enforcing per-tenant provider-class policy when wired.

		``local_only=True`` is a data-minimisation-by-design pin (GDPR Art 25 / PDPL): the prompt
		carries unstructured sensitive text — a D&N call transcript can contain HEALTH information
		(special-category data), a SoV insured names/addresses — that the regex redactor cannot
		sanitise (no NER). Such prompts are NEVER egressed to a cloud provider, even for a
		``cloud_egress_approved`` tenant: every non-self-hosted step is skipped. If no self-hosted
		model is available the chain raises and the call degrades cleanly (broker records manually).

		Raises :class:`AllProvidersUnavailable` if every allowed step fails or
		all steps are policy-skipped — each public method already maps that to
		a clean ``DraftDegraded``.
		"""
		if self.enforcer is not None and self.steps is not None:
			now = now or datetime.now(timezone.utc)
			# H-2: rate-limit EVERY assist call (incl. all-local) so an approved tenant can't spam
			# cloud, and an Ollama-only tenant can't DoS the local GPU. One token per call.
			rate = self.enforcer.rate_limiter.try_consume(tenant_id)
			if isinstance(rate, Denied):
				logger.info("assist: tenant %s rate-limited (retry_after=%ss)", tenant_id, rate.retry_after_seconds)
				raise AllProvidersUnavailable([("rate_limit", f"rate limited; retry_after={rate.retry_after_seconds}s")])
			errors: list[tuple[str, str]] = []
			for step in self.steps:
				if local_only and step.provider_class != "self-hosted":
					logger.info(
						"assist: data-minimisation pin — sensitive free-text NOT egressed to "
						"cloud step %s (provider_class %s, tenant %s); self-hosted only",
						step.model_id, step.provider_class, tenant_id,
					)
					continue
				if not self.enforcer.provider_class_allowed(tenant_id, step.provider_class):
					logger.info(
						"assist: tenant %s tier forbids provider_class %s — skipping step %s",
						tenant_id, step.provider_class, step.model_id,
					)
					continue
				# H-2: cost-ceiling pre-check for CLOUD steps only (self-hosted/Ollama is free, so it
				# never debits and is always allowed). Once the tenant's period spend is at/over the
				# ceiling, skip the cloud step — local serves if available, else degrade.
				if step.provider_class != "self-hosted" and self._cloud_over_ceiling(tenant_id, now):
					logger.info(
						"assist: tenant %s at/over monthly ceiling — skipping cloud step %s",
						tenant_id, step.model_id,
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
					response = step.caller.call(model_id=step.model_id, prompt=step_prompt)
				except Exception as e:  # noqa: BLE001 - any failure advances the chain
					errors.append((step.model_id, f"{type(e).__name__}: {e}"))
					logger.warning("assist provider %s failed, trying next: %s", step.model_id, e)
					continue
				# H-2: AI Run Log — every served assist call emits ONE canonical activity row
				# (CRM §4). Cost is omitted: assist models are unpriced and the cloud one is :free.
				self._emit_run_log(tenant_id, step, response, now)
				return response
			raise AllProvidersUnavailable(errors)
		return self.llm.call(model_id=model_id, prompt=prompt)

	def _cloud_over_ceiling(self, tenant_id: str, now: datetime) -> bool:
		"""True when the tenant's period spend has reached its monthly ceiling (cloud gate).

		Model-independent: reads the ledger period total vs the policy ceiling. An unknown tenant
		has no policy/ceiling → not over (a cloud step is then refused by residency/tier anyway).
		"""
		try:
			policy = self.enforcer.policy_for(tenant_id)
		except Exception:  # noqa: BLE001 - unknown tenant -> no ceiling to enforce here
			return False
		return self.enforcer.ledger.period_total(tenant_id, now) >= policy.monthly_ceiling

	def _emit_run_log(self, tenant_id: str, step: ProviderStep, response: LLMResponse, now: datetime) -> None:
		"""Emit one Auxima Activity row for a served assist call (AI Run Log; CRM §4)."""
		row = build_activity_row(
			tenant_id=tenant_id,
			kind="assist.completed",
			payload={
				"model": step.model_id,
				"provider_class": step.provider_class,
				"model_version": response.model_version,
				"prompt_tokens": response.prompt_tokens,
				"completion_tokens": response.completion_tokens,
				"latency_ms": response.latency_ms,
			},
			retention=RetentionClass.OPERATIONAL,
			source="sidecar.assist",
			ts=now,
		)
		self.activity_emitter.emit(row)

	def draft_email(self, request: DraftEmailRequest) -> DraftOutcome:
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_draft_email_prompt(request)

		try:
			# local_only: the recipient/company names + the `examples` past-sent email bodies can
			# carry personal data (and health narrative for medical-lines brokers) the regex
			# redactor can't strip — never egress to cloud (audit 2026-06-10 H-1).
			llm_response = self._invoke(
				tenant_id=request.tenant_id, model_id=model_id, prompt=prompt, local_only=True,
			)
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
			# local_only: `context` is untrusted record/error data that can carry PII the regex
			# redactor can't strip — never egress to cloud (audit 2026-06-10 H-1).
			llm_response = self._invoke(
				tenant_id=request.tenant_id, model_id=model_id, prompt=prompt, local_only=True,
			)
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
			# local_only: `current_values` are already-filled field values (PII) the regex redactor
			# can't strip — never egress to cloud (audit 2026-06-10 H-1).
			llm_response = self._invoke(
				tenant_id=request.tenant_id, model_id=model_id, prompt=prompt, local_only=True,
			)
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

	def draft_recommendation(self, request: RecommendationRequest) -> RecommendationOutcome:
		"""Draft the Article 9(b) recommendation: LLM writes the reasoning, the service appends
		the mandatory legal block deterministically, then runs the legal-line check. Degrades
		cleanly (broker writes the note manually) if no model is available."""
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_recommendation_prompt(request)

		try:
			# local_only: `client_needs` (demands & needs signals) can carry health/personal data the
			# regex redactor can't strip — never egress to cloud (audit 2026-06-10 H-1).
			llm_response = self._invoke(
				tenant_id=request.tenant_id, model_id=model_id, prompt=prompt, local_only=True,
			)
		except AllProvidersUnavailable as e:
			emit(
				"warn", "assist.draft_recommendation.degraded",
				fields={"tenant_id": request.tenant_id, "reason": str(e)[:200]},
			)
			return DraftDegraded(reason=str(e))

		try:
			fields = validate_recommendation_response(llm_response.payload)
		except SchemaViolationError as e:
			emit(
				"warn", "assist.draft_recommendation.schema_violation",
				fields={"tenant_id": request.tenant_id, "error_count": len(e.errors)},
			)
			return DraftSchemaInvalid(errors=tuple(e.errors))

		body_md = _assemble_recommendation_body(request, fields)
		legal = _legal_check(request, body_md)
		response = RecommendationResponse(
			recommended_insurer=request.recommended_insurer,
			body_md=body_md,
			citations=list(fields.citations),
			legal_check=legal,
			language=request.language,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.draft_recommendation.completed",
			fields={
				"tenant_id": request.tenant_id, "language": request.language,
				"recommended_insurer": request.recommended_insurer,
				"legal_passed": legal.passed, "model_version": response.model_version,
				"tokens": response.prompt_tokens + response.completion_tokens,
			},
		)
		return RecommendationSuccess(response=response)

	def wording_diff(self, request: WordingDiffRequest) -> WordingDiffOutcome:
		"""Surface material differences across insurer offer wordings (WT-G10). Degrades cleanly."""
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_wording_diff_prompt(request)

		try:
			llm_response = self._invoke(tenant_id=request.tenant_id, model_id=model_id, prompt=prompt)
		except AllProvidersUnavailable as e:
			emit(
				"warn", "assist.wording_diff.degraded",
				fields={"tenant_id": request.tenant_id, "reason": str(e)[:200]},
			)
			return DraftDegraded(reason=str(e))

		try:
			fields = validate_wording_diff_response(llm_response.payload)
		except SchemaViolationError as e:
			emit(
				"warn", "assist.wording_diff.schema_violation",
				fields={"tenant_id": request.tenant_id, "error_count": len(e.errors)},
			)
			return DraftSchemaInvalid(errors=tuple(e.errors))

		response = WordingDiffResponse(
			differences=list(fields.differences),
			flags=list(fields.flags),
			language=request.language,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.wording_diff.completed",
			fields={
				"tenant_id": request.tenant_id, "language": request.language,
				"offers": len(request.offers), "differences": len(response.differences),
				"model_version": response.model_version,
			},
		)
		return WordingDiffSuccess(response=response)

	def summarise_dn(self, request: DNSummaryRequest) -> DNSummaryOutcome:
		"""Summarise a D&N call into structured needs + coverage gaps (WT-G13). Degrades cleanly."""
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_dn_summary_prompt(request)

		try:
			# local_only: a D&N transcript can carry HEALTH data (special-category) — never to cloud.
			llm_response = self._invoke(
				tenant_id=request.tenant_id, model_id=model_id, prompt=prompt, local_only=True,
			)
		except AllProvidersUnavailable as e:
			emit("warn", "assist.summarise_dn.degraded", fields={"tenant_id": request.tenant_id, "reason": str(e)[:200]})
			return DraftDegraded(reason=str(e))

		try:
			fields = validate_dn_summary_response(llm_response.payload)
		except SchemaViolationError as e:
			emit("warn", "assist.summarise_dn.schema_violation", fields={"tenant_id": request.tenant_id, "error_count": len(e.errors)})
			return DraftSchemaInvalid(errors=tuple(e.errors))

		response = DNSummaryResponse(
			needs=list(fields.needs),
			coverage_gaps=list(fields.coverage_gaps),
			language=request.language,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.summarise_dn.completed",
			fields={
				"tenant_id": request.tenant_id, "language": request.language,
				"needs": len(response.needs), "gaps": len(response.coverage_gaps),
				"model_version": response.model_version,
			},
		)
		return DNSummarySuccess(response=response)

	def extract_sov(self, request: SoVExtractRequest) -> SoVExtractOutcome:
		"""Structure SoV text into line items (WT-G11). Degrades cleanly."""
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_sov_extract_prompt(request)

		try:
			# local_only: a SoV can carry insured names/addresses (personal data the regex
			# redactor can't strip) — keep the raw schedule in-Kingdom, never to cloud.
			llm_response = self._invoke(
				tenant_id=request.tenant_id, model_id=model_id, prompt=prompt, local_only=True,
			)
		except AllProvidersUnavailable as e:
			emit("warn", "assist.extract_sov.degraded", fields={"tenant_id": request.tenant_id, "reason": str(e)[:200]})
			return DraftDegraded(reason=str(e))

		try:
			fields = validate_sov_extract_response(llm_response.payload)
		except SchemaViolationError as e:
			emit("warn", "assist.extract_sov.schema_violation", fields={"tenant_id": request.tenant_id, "error_count": len(e.errors)})
			return DraftSchemaInvalid(errors=tuple(e.errors))

		response = SoVExtractResponse(
			line_items=fields.line_items,
			total_value=fields.total_value,
			count=len(fields.line_items),
			language=request.language,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.extract_sov.completed",
			fields={"tenant_id": request.tenant_id, "items": response.count, "model_version": response.model_version},
		)
		return SoVExtractSuccess(response=response)

	def classify_intent(self, request: IntentClassifyRequest) -> IntentClassifyOutcome:
		"""Classify an inbound insurer reply (quote/counter/decline/rfi/other) (WT-G08). Degrades cleanly."""
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_intent_classify_prompt(request)

		try:
			# local_only: the insurer message can reference the insured (names/details) — keep in-Kingdom.
			llm_response = self._invoke(
				tenant_id=request.tenant_id, model_id=model_id, prompt=prompt, local_only=True,
			)
		except AllProvidersUnavailable as e:
			emit("warn", "assist.classify_intent.degraded", fields={"tenant_id": request.tenant_id, "reason": str(e)[:200]})
			return DraftDegraded(reason=str(e))

		try:
			fields = validate_intent_classify_response(llm_response.payload)
		except SchemaViolationError as e:
			emit("warn", "assist.classify_intent.schema_violation", fields={"tenant_id": request.tenant_id, "error_count": len(e.errors)})
			return DraftSchemaInvalid(errors=tuple(e.errors))

		response = IntentClassifyResponse(
			intent=fields.intent,
			confidence=fields.confidence,
			rationale=fields.rationale,
			language=request.language,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.classify_intent.completed",
			fields={"tenant_id": request.tenant_id, "intent": response.intent, "model_version": response.model_version},
		)
		return IntentClassifySuccess(response=response)

	def ingest_policy(self, request: PolicyIngestRequest) -> PolicyIngestOutcome:
		"""Extract policy fields from an issued schedule/wording + diff vs bound terms (WT-G15). Degrades cleanly."""
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_policy_ingest_prompt(request)

		try:
			# local_only: a policy schedule carries insured names/addresses (and health for medical
			# lines) the regex redactor can't strip — never egress the document to cloud.
			llm_response = self._invoke(
				tenant_id=request.tenant_id, model_id=model_id, prompt=prompt, local_only=True,
			)
		except AllProvidersUnavailable as e:
			emit("warn", "assist.ingest_policy.degraded", fields={"tenant_id": request.tenant_id, "reason": str(e)[:200]})
			return DraftDegraded(reason=str(e))

		try:
			fields = validate_policy_ingest_response(llm_response.payload)
		except SchemaViolationError as e:
			emit("warn", "assist.ingest_policy.schema_violation", fields={"tenant_id": request.tenant_id, "error_count": len(e.errors)})
			return DraftSchemaInvalid(errors=tuple(e.errors))

		response = PolicyIngestResponse(
			fields=fields.fields,
			discrepancies=list(fields.discrepancies),
			language=request.language,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.ingest_policy.completed",
			fields={
				"tenant_id": request.tenant_id, "discrepancies": len(response.discrepancies),
				"model_version": response.model_version,
			},
		)
		return PolicyIngestSuccess(response=response)

	def draft_renewal(self, request: RenewalDraftRequest) -> RenewalDraftOutcome:
		"""Pre-draft a renewal RFQ + broker brief from the expiring policy + loss experience (WT-G20). Degrades cleanly."""
		model_id = request.model_id or DEFAULT_MODEL_ID
		prompt = build_renewal_draft_prompt(request)

		try:
			# local_only: loss/claims experience can include health data + the policy summary carries
			# insured details — keep the renewal context in-Kingdom.
			llm_response = self._invoke(
				tenant_id=request.tenant_id, model_id=model_id, prompt=prompt, local_only=True,
			)
		except AllProvidersUnavailable as e:
			emit("warn", "assist.draft_renewal.degraded", fields={"tenant_id": request.tenant_id, "reason": str(e)[:200]})
			return DraftDegraded(reason=str(e))

		try:
			fields = validate_renewal_draft_response(llm_response.payload)
		except SchemaViolationError as e:
			emit("warn", "assist.draft_renewal.schema_violation", fields={"tenant_id": request.tenant_id, "error_count": len(e.errors)})
			return DraftSchemaInvalid(errors=tuple(e.errors))

		response = RenewalDraftResponse(
			rfq_subject=fields.rfq_subject,
			rfq_body=fields.rfq_body,
			considerations=list(fields.considerations),
			language=request.language,
			degraded=False,
			model_version=llm_response.model_version,
			prompt_tokens=llm_response.prompt_tokens,
			completion_tokens=llm_response.completion_tokens,
			latency_ms=llm_response.latency_ms,
		)
		emit(
			"info", "assist.draft_renewal.completed",
			fields={
				"tenant_id": request.tenant_id, "considerations": len(response.considerations),
				"model_version": response.model_version,
			},
		)
		return RenewalDraftSuccess(response=response)


__all__ = (
	"AssistService",
	"DEFAULT_MODEL_ID",
	"ProviderStep",
	"DNSummaryOutcome",
	"DNSummarySuccess",
	"IntentClassifyOutcome",
	"IntentClassifySuccess",
	"PolicyIngestOutcome",
	"PolicyIngestSuccess",
	"RenewalDraftOutcome",
	"RenewalDraftSuccess",
	"SoVExtractOutcome",
	"SoVExtractSuccess",
	"DraftDegraded",
	"DraftEmailSuccess",
	"DraftNoteSuccess",
	"DraftOutcome",
	"DraftSchemaInvalid",
	"NoteOutcome",
	"RecommendationOutcome",
	"RecommendationSuccess",
	"SuggestFieldsSuccess",
	"SuggestOutcome",
	"WordingDiffOutcome",
	"WordingDiffSuccess",
)
