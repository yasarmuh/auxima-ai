"""FastAPI router for ``POST /v1/assist/draft-email``.

Thin adapter: parse body → call :class:`AssistService.draft_email` → map the
typed outcome to HTTP. Degradation is a first-class, non-error response so the
Frappe composer can branch on it without treating it as a crash.

    DraftEmailSuccess   -> 200 {subject, body, ...}
    DraftDegraded       -> 503 {detail, degraded: true}   (all models unavailable)
    DraftSchemaInvalid  -> 502 {detail, errors}           (model returned wrong shape)

The service is wired via :func:`get_assist_service` so production injects the
real fallback caller (OpenRouter→Ollama) and tests override with a stub.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from auxima_ai.assist.schema import (
	DNSummaryRequest,
	DNSummaryResponse,
	DraftEmailRequest,
	DraftEmailResponse,
	DraftNoteRequest,
	DraftNoteResponse,
	RecommendationRequest,
	RecommendationResponse,
	SuggestFieldsRequest,
	SuggestFieldsResponse,
	WordingDiffRequest,
	WordingDiffResponse,
)
from auxima_ai.assist.service import (
	AssistService,
	DNSummarySuccess,
	DraftDegraded,
	DraftEmailSuccess,
	DraftNoteSuccess,
	DraftSchemaInvalid,
	RecommendationSuccess,
	SuggestFieldsSuccess,
	WordingDiffSuccess,
)

_service_singleton: AssistService | None = None


def get_assist_service() -> AssistService:
	"""App-wide :class:`AssistService`. Tests override via dependency_overrides."""
	global _service_singleton
	if _service_singleton is None:
		_service_singleton = AssistService()
	return _service_singleton


def set_assist_service(service: AssistService) -> None:
	"""Install a custom service singleton — call at deployment startup."""
	global _service_singleton
	_service_singleton = service


def reset_assist_service() -> None:
	"""Clear the singleton — test-only."""
	global _service_singleton
	_service_singleton = None


router = APIRouter(prefix="/v1/assist", tags=["assist"])


@router.post(
	"/draft-email",
	response_model=DraftEmailResponse,
	summary="Draft an outbound email from record context (best-effort, learns from edits)",
	responses={
		200: {"description": "Drafted subject + body"},
		502: {"description": "Upstream model replied but not in the required shape"},
		503: {"description": "All AI models unavailable — compose manually"},
	},
)
def draft_email(
	body: DraftEmailRequest,
	service: AssistService = Depends(get_assist_service),
):
	"""Draft one email; degrade cleanly if no model is available."""
	outcome = service.draft_email(body)

	if isinstance(outcome, DraftEmailSuccess):
		return JSONResponse(status_code=200, content=outcome.response.model_dump())

	if isinstance(outcome, DraftDegraded):
		return JSONResponse(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			content={
				"detail": "AI drafting is temporarily unavailable; please compose manually.",
				"degraded": True,
				"reason": outcome.reason,
			},
			headers={"Retry-After": "30"},
		)

	if isinstance(outcome, DraftSchemaInvalid):
		return JSONResponse(
			status_code=status.HTTP_502_BAD_GATEWAY,
			content={
				"detail": "the AI model returned an unexpected format; please compose manually.",
				"errors": list(outcome.errors),
			},
		)

	raise AssertionError(f"unhandled draft outcome: {type(outcome).__name__}")  # pragma: no cover


@router.post(
	"/draft-note",
	response_model=DraftNoteResponse,
	summary="Draft a short note/comment, or explain a blocked action (error-help)",
	responses={
		200: {"description": "Drafted note text"},
		502: {"description": "Upstream model replied but not in the required shape"},
		503: {"description": "All AI models unavailable — proceed manually"},
	},
)
def draft_note(
	body: DraftNoteRequest,
	service: AssistService = Depends(get_assist_service),
):
	"""Draft one note (comment / error_help / general); degrade cleanly."""
	outcome = service.draft_note(body)

	if isinstance(outcome, DraftNoteSuccess):
		return JSONResponse(status_code=200, content=outcome.response.model_dump())

	if isinstance(outcome, DraftDegraded):
		return JSONResponse(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			content={"detail": "AI is temporarily unavailable.", "degraded": True, "reason": outcome.reason},
			headers={"Retry-After": "30"},
		)

	if isinstance(outcome, DraftSchemaInvalid):
		return JSONResponse(
			status_code=status.HTTP_502_BAD_GATEWAY,
			content={"detail": "the AI model returned an unexpected format.", "errors": list(outcome.errors)},
		)

	raise AssertionError(f"unhandled note outcome: {type(outcome).__name__}")  # pragma: no cover


@router.post(
	"/suggest-fields",
	response_model=SuggestFieldsResponse,
	summary="Suggest values for empty fields (suggestion-only; user reviews)",
	responses={
		200: {"description": "Suggestions for the requested empty fields (may be empty)"},
		502: {"description": "Upstream model replied but not in the required shape"},
		503: {"description": "All AI models unavailable"},
	},
)
def suggest_fields(
	body: SuggestFieldsRequest,
	service: AssistService = Depends(get_assist_service),
):
	"""Suggest values for empty fields; degrade cleanly."""
	outcome = service.suggest_fields(body)

	if isinstance(outcome, SuggestFieldsSuccess):
		return JSONResponse(status_code=200, content=outcome.response.model_dump())

	if isinstance(outcome, DraftDegraded):
		return JSONResponse(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			content={"detail": "AI is temporarily unavailable.", "degraded": True, "reason": outcome.reason},
			headers={"Retry-After": "30"},
		)

	if isinstance(outcome, DraftSchemaInvalid):
		return JSONResponse(
			status_code=status.HTTP_502_BAD_GATEWAY,
			content={"detail": "the AI model returned an unexpected format.", "errors": list(outcome.errors)},
		)

	raise AssertionError(f"unhandled suggest outcome: {type(outcome).__name__}")  # pragma: no cover


@router.post(
	"/draft-recommendation",
	response_model=RecommendationResponse,
	summary="Draft the Article 9(b) recommendation note (reasoning + citations + legal-line check)",
	responses={
		200: {"description": "Drafted note (body_md) + structured legal_check"},
		502: {"description": "Upstream model replied but not in the required shape"},
		503: {"description": "All AI models unavailable — write the note manually"},
	},
)
def draft_recommendation(
	body: RecommendationRequest,
	service: AssistService = Depends(get_assist_service),
):
	"""Draft one recommendation note; degrade cleanly if no model is available."""
	outcome = service.draft_recommendation(body)

	if isinstance(outcome, RecommendationSuccess):
		return JSONResponse(status_code=200, content=outcome.response.model_dump())

	if isinstance(outcome, DraftDegraded):
		return JSONResponse(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			content={"detail": "AI is temporarily unavailable; please write the note manually.",
			         "degraded": True, "reason": outcome.reason},
			headers={"Retry-After": "30"},
		)

	if isinstance(outcome, DraftSchemaInvalid):
		return JSONResponse(
			status_code=status.HTTP_502_BAD_GATEWAY,
			content={"detail": "the AI model returned an unexpected format.", "errors": list(outcome.errors)},
		)

	raise AssertionError(f"unhandled recommendation outcome: {type(outcome).__name__}")  # pragma: no cover


@router.post(
	"/wording-diff",
	response_model=WordingDiffResponse,
	summary="Diff insurer offer wordings → material differences + flags (WT-G10)",
	responses={
		200: {"description": "Material differences + client-flags"},
		502: {"description": "Upstream model replied but not in the required shape"},
		503: {"description": "All AI models unavailable"},
	},
)
def wording_diff(
	body: WordingDiffRequest,
	service: AssistService = Depends(get_assist_service),
):
	"""Compare >=2 offer wordings; degrade cleanly if no model is available."""
	outcome = service.wording_diff(body)

	if isinstance(outcome, WordingDiffSuccess):
		return JSONResponse(status_code=200, content=outcome.response.model_dump())

	if isinstance(outcome, DraftDegraded):
		return JSONResponse(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			content={"detail": "AI is temporarily unavailable.", "degraded": True, "reason": outcome.reason},
			headers={"Retry-After": "30"},
		)

	if isinstance(outcome, DraftSchemaInvalid):
		return JSONResponse(
			status_code=status.HTTP_502_BAD_GATEWAY,
			content={"detail": "the AI model returned an unexpected format.", "errors": list(outcome.errors)},
		)

	raise AssertionError(f"unhandled wording-diff outcome: {type(outcome).__name__}")  # pragma: no cover


@router.post(
	"/summarise-dn",
	response_model=DNSummaryResponse,
	summary="Summarise a D&N call → structured needs + coverage gaps (WT-G13)",
	responses={
		200: {"description": "Demands & needs + coverage gaps"},
		502: {"description": "Upstream model replied but not in the required shape"},
		503: {"description": "All AI models unavailable"},
	},
)
def summarise_dn(
	body: DNSummaryRequest,
	service: AssistService = Depends(get_assist_service),
):
	"""Summarise a demands & needs call; degrade cleanly if no model is available."""
	outcome = service.summarise_dn(body)

	if isinstance(outcome, DNSummarySuccess):
		return JSONResponse(status_code=200, content=outcome.response.model_dump())

	if isinstance(outcome, DraftDegraded):
		return JSONResponse(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			content={"detail": "AI is temporarily unavailable.", "degraded": True, "reason": outcome.reason},
			headers={"Retry-After": "30"},
		)

	if isinstance(outcome, DraftSchemaInvalid):
		return JSONResponse(
			status_code=status.HTTP_502_BAD_GATEWAY,
			content={"detail": "the AI model returned an unexpected format.", "errors": list(outcome.errors)},
		)

	raise AssertionError(f"unhandled dn-summary outcome: {type(outcome).__name__}")  # pragma: no cover


__all__ = (
	"draft_email",
	"draft_note",
	"draft_recommendation",
	"get_assist_service",
	"reset_assist_service",
	"router",
	"set_assist_service",
	"suggest_fields",
	"summarise_dn",
	"wording_diff",
)
