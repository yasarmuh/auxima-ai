# Copyright (c) 2026, Auxilium Tech and contributors
"""HTTP adapter for the ClaimsCrew (P3-01) — POST /v1/claims/process.

Mirrors the assist router conventions: a module singleton installed by bootstrap
(:func:`set_claims_service`), overridable in tests. The crew is ADVISORY — a rejected FNOL is
a 422 (the caller fixes the notice), a processed one is a 200 carrying recommendations the
broker accepts or discards in the Desk. The crew never writes Frappe records.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from auxima_ai.claims.schema import ClaimsProcessOutcome, FNOLRequest
from auxima_ai.claims.service import ClaimsCrewService

_service_singleton: ClaimsCrewService | None = None


def get_claims_service() -> ClaimsCrewService:
	"""App-wide :class:`ClaimsCrewService`. Tests override via dependency_overrides."""
	if _service_singleton is None:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="claims crew not configured (no LLM chain wired)",
		)
	return _service_singleton


def set_claims_service(service: ClaimsCrewService) -> None:
	"""Install the singleton — called by bootstrap at deployment startup."""
	global _service_singleton
	_service_singleton = service


def reset_claims_service() -> None:
	"""Clear the singleton — test-only."""
	global _service_singleton
	_service_singleton = None


router = APIRouter(prefix="/v1/claims", tags=["claims"])


@router.post(
	"/process",
	response_model=ClaimsProcessOutcome,
	summary="Run the ClaimsCrew state machine over an FNOL (advisory triage/reserve/routing)",
	responses={
		200: {"description": "Crew recommendations (possibly degraded to heuristics)"},
		422: {"description": "FNOL failed validation (fail-closed; nothing was processed)"},
		503: {"description": "Crew not configured"},
	},
)
def process_fnol(
	body: FNOLRequest,
	service: ClaimsCrewService = Depends(get_claims_service),
):
	outcome = service.process(body)
	if outcome.status == "rejected":
		raise HTTPException(
			status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
			detail={"reason": outcome.reason, "audit_trail": outcome.audit_trail},
		)
	return outcome
