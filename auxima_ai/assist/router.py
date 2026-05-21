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

from auxima_ai.assist.schema import DraftEmailRequest, DraftEmailResponse
from auxima_ai.assist.service import (
	AssistService,
	DraftDegraded,
	DraftEmailSuccess,
	DraftSchemaInvalid,
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


__all__ = (
	"draft_email",
	"get_assist_service",
	"reset_assist_service",
	"router",
	"set_assist_service",
)
