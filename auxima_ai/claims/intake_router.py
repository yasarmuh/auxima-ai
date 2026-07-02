# Copyright (c) 2026, Auxilium Tech and contributors
"""HTTP adapter for the multi-turn FNOL intake (P3-01c) — POST /v1/claims/fnol/turn.

Same conventions as the claims router: a module singleton installed by bootstrap,
overridable in tests. Advisory-only — a completed session's outcome carries crew
RECOMMENDATIONS; the auxima app creates the actual Claim after broker acceptance.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from auxima_ai.claims.intake import FNOLIntakeService, IntakeSessionExhausted
from auxima_ai.claims.intake_schema import FNOLTurnOutcome, FNOLTurnRequest

_service_singleton: FNOLIntakeService | None = None


def get_fnol_intake_service() -> FNOLIntakeService:
	"""App-wide :class:`FNOLIntakeService`. Tests override via dependency_overrides."""
	if _service_singleton is None:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="fnol intake not configured",
		)
	return _service_singleton


def set_fnol_intake_service(service: FNOLIntakeService) -> None:
	"""Install the singleton — called by bootstrap at deployment startup."""
	global _service_singleton
	_service_singleton = service


def reset_fnol_intake_service() -> None:
	"""Clear the singleton — test-only."""
	global _service_singleton
	_service_singleton = None


router = APIRouter(prefix="/v1/claims", tags=["claims"])


@router.post(
	"/fnol/turn",
	response_model=FNOLTurnOutcome,
	summary="One reporter message in a multi-turn FNOL intake session (checkpointed)",
	responses={
		200: {"description": "Collecting (bilingual next question) or processed (crew outcome)"},
		409: {"description": "Session exceeded its turn cap — file via the Desk"},
		503: {"description": "Intake not configured"},
	},
)
def fnol_turn(
	body: FNOLTurnRequest,
	service: FNOLIntakeService = Depends(get_fnol_intake_service),
):
	try:
		return service.turn(body)
	except IntakeSessionExhausted as e:
		raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
