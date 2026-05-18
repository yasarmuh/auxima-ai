"""FastAPI router for ``POST /v1/intake/extract``.

Thin adapter: parses the request, pulls the Idempotency-Key + optional
traceparent headers, calls :class:`IntakeService.extract`, and maps
the typed outcome to the right HTTP status + headers.

Status mapping:

    IntakeSuccess           -> 200
    IntakeReplay            -> 200 with ``Idempotent-Replayed: true`` header
    IntakeInFlight          -> 409 with ``Retry-After`` (seconds)
    IntakeConflict          -> 422 (client reused key for different body)
    IntakeProviderDenied    -> 403 (tenant tier forbids provider)
    IntakeRateLimited       -> 429 with ``Retry-After``
    IntakeCeilingExceeded   -> 402 (Payment Required — cost ceiling hit)
    IntakeUnknownProvider   -> 500 (config bug — pricing-classification gap)

The service is wired via :func:`get_intake_service` so production
deployments inject a real :class:`PolicyEnforcer` + LLM caller, while
tests override the dependency to plug in stubs.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse

from auxima_ai.intake.schema import IntakeRequest, IntakeResponse
from auxima_ai.intake.service import (
    IntakeCeilingExceeded,
    IntakeConflict,
    IntakeInFlight,
    IntakeProviderDenied,
    IntakeRateLimited,
    IntakeReplay,
    IntakeSchemaInvalid,
    IntakeService,
    IntakeSuccess,
    IntakeUnknownProvider,
)
from auxima_ai.observability.trace import parse_traceparent


# ---------------------------------------------------------------------------
# Service singleton + dependency provider
# ---------------------------------------------------------------------------

_service_singleton: IntakeService | None = None


def get_intake_service() -> IntakeService:
    """Lazily-constructed app-wide :class:`IntakeService`.

    Tests override via ``app.dependency_overrides[get_intake_service]``.
    Production startup may call :func:`set_intake_service` once to
    install a service pre-wired with the real policy / ledger / LLM
    caller; otherwise a default unconfigured service is built and
    will refuse every call with :class:`UnknownTenantError` until
    policies are registered.
    """
    global _service_singleton
    if _service_singleton is None:
        from auxima_ai.policy.enforcer import PolicyEnforcer
        _service_singleton = IntakeService(enforcer=PolicyEnforcer())
    return _service_singleton


def set_intake_service(service: IntakeService) -> None:
    """Install a custom service singleton — call at deployment startup."""
    global _service_singleton
    _service_singleton = service


def reset_intake_service() -> None:
    """Clear the singleton — test-only."""
    global _service_singleton
    _service_singleton = None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/v1/intake", tags=["intake"])


def _ceil_retry_after(seconds: float) -> str:
    """``Retry-After`` is integer seconds per RFC 7231. Round up to be safe."""
    return str(max(1, math.ceil(seconds)))


@router.post(
    "/extract",
    response_model=IntakeResponse,
    summary="Extract structured lead fields from free-form intake text",
    responses={
        200: {"description": "Extracted; ``Idempotent-Replayed: true`` header set on replays"},
        402: {"description": "Tenant monthly cost ceiling would be exceeded"},
        403: {"description": "Tenant tier forbids this provider"},
        409: {"description": "Same Idempotency-Key already in flight; retry after the hint"},
        422: {"description": "Body validation failed OR same key with different body"},
        429: {"description": "Per-tenant rate limit hit; retry after the hint"},
        500: {"description": "Configuration bug — provider not classified for tier-gate"},
        502: {"description": "Upstream LLM returned a payload that violated the field schema"},
    },
)
def extract(
    body: IntakeRequest,
    idempotency_key: str = Header(
        ...,
        alias="Idempotency-Key",
        min_length=1,
        max_length=255,
        description="Stable opaque identifier — same key + body = idempotent replay.",
    ),
    traceparent: str | None = Header(default=None, alias="traceparent"),
    service: IntakeService = Depends(get_intake_service),
):
    """Process one intake.extract call through the full pipeline."""
    trace = parse_traceparent(traceparent)
    outcome = service.extract(
        body,
        idempotency_key=idempotency_key,
        now=datetime.now(timezone.utc),
        trace=trace,
    )

    if isinstance(outcome, IntakeSuccess):
        return JSONResponse(
            status_code=200,
            content=outcome.response.model_dump(),
        )

    if isinstance(outcome, IntakeReplay):
        return JSONResponse(
            status_code=200,
            content=outcome.response.model_dump(),
            headers={"Idempotent-Replayed": "true"},
        )

    if isinstance(outcome, IntakeInFlight):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": "another worker is processing this Idempotency-Key",
                "key": outcome.key,
            },
            headers={"Retry-After": "1"},
        )

    if isinstance(outcome, IntakeConflict):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Idempotency-Key reused with a different request body",
                "key": outcome.key,
                "seen_fingerprint": outcome.seen_fingerprint,
                "new_fingerprint": outcome.new_fingerprint,
            },
        )

    if isinstance(outcome, IntakeProviderDenied):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "detail": "tenant tier policy does not permit this provider",
                "provider": outcome.provider,
                "provider_class": outcome.provider_class,
            },
        )

    if isinstance(outcome, IntakeRateLimited):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "per-tenant rate limit hit"},
            headers={"Retry-After": _ceil_retry_after(outcome.retry_after_seconds)},
        )

    if isinstance(outcome, IntakeCeilingExceeded):
        return JSONResponse(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            content={
                "detail": "monthly cost ceiling would be exceeded",
                "estimated_cost": outcome.estimated_cost,
                "current_total": outcome.current_total,
                "ceiling": outcome.ceiling,
            },
        )

    if isinstance(outcome, IntakeUnknownProvider):
        # Configuration bug — refuse rather than guess.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"provider {outcome.provider!r} not classified for tier policy",
        )

    if isinstance(outcome, IntakeSchemaInvalid):
        # Upstream LLM returned the wrong shape — neither client nor
        # sidecar bug, but we refuse to write a malformed activity row.
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "detail": "upstream LLM response violated the intake-extract schema",
                "errors": list(outcome.errors),
            },
        )

    # Defensive: every outcome variant must be handled above.
    raise HTTPException(  # pragma: no cover
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"unhandled intake outcome: {type(outcome).__name__}",
    )


__all__ = (
    "extract",
    "get_intake_service",
    "reset_intake_service",
    "router",
    "set_intake_service",
)
