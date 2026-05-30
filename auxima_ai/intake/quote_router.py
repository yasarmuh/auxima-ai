"""FastAPI router for ``POST /v1/intake/extract-quote`` (P1-10).

Thin adapter mirroring ``router.py`` (the lead path): parse the request, pull
the Idempotency-Key + optional traceparent, call
:meth:`QuoteIntakeService.extract_quote`, map the typed outcome to HTTP.

Status mapping:

    QuoteSuccess          -> 200
    QuoteReplay           -> 200 with ``Idempotent-Replayed: true``
    QuoteInFlight         -> 409 with ``Retry-After``
    QuoteConflict         -> 422 (key reused with a different document)
    QuoteProviderDenied   -> 403
    QuoteRateLimited      -> 429 with ``Retry-After``
    QuoteCeilingExceeded  -> 402
    QuoteUnknownProvider  -> 500 (config bug)
    QuoteSchemaInvalid    -> 502 (upstream LLM violated the schema)
    QuoteDocumentFailed   -> 422 with ``{reason, doc_class}`` — the §4.2
                             corrupt/encrypted/oversized/no-text-layer route;
                             the Frappe side reads the reason and routes the
                             Placement to Failed (never a silent success).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse

from auxima_ai.intake.quote_schema import QuoteIntakeRequest, QuoteIntakeResponse
from auxima_ai.intake.quote_service import (
    QuoteCeilingExceeded,
    QuoteConflict,
    QuoteDocumentFailed,
    QuoteInFlight,
    QuoteIntakeService,
    QuoteProviderDenied,
    QuoteRateLimited,
    QuoteReplay,
    QuoteSchemaInvalid,
    QuoteSuccess,
    QuoteUnknownProvider,
)
from auxima_ai.observability.trace import parse_traceparent

_service_singleton: QuoteIntakeService | None = None


def get_quote_intake_service() -> QuoteIntakeService:
    """Lazily-constructed app-wide :class:`QuoteIntakeService`.

    Tests override via ``app.dependency_overrides``. Production startup calls
    :func:`set_quote_intake_service` with a service pre-wired with the real
    policy / ledger / LLM / pdf-extractor / OCR engine.
    """
    global _service_singleton
    if _service_singleton is None:
        from auxima_ai.policy.enforcer import PolicyEnforcer
        _service_singleton = QuoteIntakeService(enforcer=PolicyEnforcer())
    return _service_singleton


def set_quote_intake_service(service: QuoteIntakeService) -> None:
    global _service_singleton
    _service_singleton = service


def reset_quote_intake_service() -> None:
    """Clear the singleton — test-only."""
    global _service_singleton
    _service_singleton = None


router = APIRouter(prefix="/v1/intake", tags=["intake"])


def _ceil_retry_after(seconds: float) -> str:
    return str(max(1, math.ceil(seconds)))


@router.post(
    "/extract-quote",
    response_model=QuoteIntakeResponse,
    summary="Extract structured quote fields from an insurer-quote PDF",
    responses={
        200: {"description": "Extracted; ``Idempotent-Replayed: true`` on replays"},
        402: {"description": "Tenant monthly cost ceiling would be exceeded"},
        403: {"description": "Tenant tier forbids this provider"},
        409: {"description": "Same Idempotency-Key already in flight"},
        422: {"description": "Body invalid, key reused with a different doc, OR the "
                             "document could not be processed (corrupt/encrypted/no-text)"},
        429: {"description": "Per-tenant rate limit hit"},
        500: {"description": "Configuration bug — provider not classified"},
        502: {"description": "Upstream LLM returned a schema-violating payload"},
    },
)
def extract_quote(
    body: QuoteIntakeRequest,
    idempotency_key: str = Header(
        ..., alias="Idempotency-Key", min_length=1, max_length=255,
        description="Stable opaque identifier — same key + document = idempotent replay.",
    ),
    traceparent: str | None = Header(default=None, alias="traceparent"),
    service: QuoteIntakeService = Depends(get_quote_intake_service),
):
    """Process one intake.extract-quote call through the full pipeline."""
    trace = parse_traceparent(traceparent)
    outcome = service.extract_quote(
        body,
        idempotency_key=idempotency_key,
        now=datetime.now(timezone.utc),
        trace=trace,
    )

    if isinstance(outcome, QuoteSuccess):
        return JSONResponse(status_code=200, content=outcome.response.model_dump())

    if isinstance(outcome, QuoteReplay):
        return JSONResponse(
            status_code=200,
            content=outcome.response.model_dump(),
            headers={"Idempotent-Replayed": "true"},
        )

    if isinstance(outcome, QuoteInFlight):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "another worker is processing this Idempotency-Key",
                     "key": outcome.key},
            headers={"Retry-After": "1"},
        )

    if isinstance(outcome, QuoteConflict):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Idempotency-Key reused with a different document",
                     "key": outcome.key,
                     "seen_fingerprint": outcome.seen_fingerprint,
                     "new_fingerprint": outcome.new_fingerprint},
        )

    if isinstance(outcome, QuoteDocumentFailed):
        # §4.2 — the document could not be turned into text. The Frappe side
        # reads ``reason`` and routes the Placement to Failed; never a 200.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "document could not be processed",
                     "reason": outcome.reason,
                     "doc_class": outcome.doc_class},
        )

    if isinstance(outcome, QuoteProviderDenied):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "tenant tier policy does not permit this provider",
                     "provider": outcome.provider,
                     "provider_class": outcome.provider_class},
        )

    if isinstance(outcome, QuoteRateLimited):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "per-tenant rate limit hit"},
            headers={"Retry-After": _ceil_retry_after(outcome.retry_after_seconds)},
        )

    if isinstance(outcome, QuoteCeilingExceeded):
        return JSONResponse(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            content={"detail": "monthly cost ceiling would be exceeded",
                     "estimated_cost": outcome.estimated_cost,
                     "current_total": outcome.current_total,
                     "ceiling": outcome.ceiling},
        )

    if isinstance(outcome, QuoteUnknownProvider):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"provider {outcome.provider!r} not classified for tier policy",
        )

    if isinstance(outcome, QuoteSchemaInvalid):
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "upstream LLM response violated the quote-extract schema",
                     "errors": list(outcome.errors)},
        )

    raise HTTPException(  # pragma: no cover
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"unhandled quote outcome: {type(outcome).__name__}",
    )


__all__ = (
    "extract_quote",
    "get_quote_intake_service",
    "reset_quote_intake_service",
    "router",
    "set_quote_intake_service",
)
