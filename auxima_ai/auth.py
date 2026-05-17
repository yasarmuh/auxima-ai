"""Shared-secret auth — the Frappe-side caller presents X-Auxima-Sidecar-Token.

This is the minimum auth surface for P0-C. The full design (GAP-16: HMAC over body +
timestamp + nonce, replay-protection, rotation) lands in a later slice. For now:
  - Header X-Auxima-Sidecar-Token must equal the configured shared_secret.
  - Missing/empty/wrong → 401.
  - /healthz is excluded (it's used by k8s liveness probes that don't carry auth).

Constant-time comparison via hmac.compare_digest to avoid timing-leak side channels.
"""
from __future__ import annotations

import hmac
from typing import Awaitable, Callable

from fastapi import Request, status
from fastapi.responses import JSONResponse

from auxima_ai.config import get_settings

HEADER_NAME = "X-Auxima-Sidecar-Token"
UNAUTHENTICATED_PATHS = {"/healthz", "/openapi.json", "/docs", "/redoc"}


async def shared_secret_middleware(
    request: Request, call_next: Callable[[Request], Awaitable]
):
    """Reject any non-healthz request whose token header doesn't match the configured secret.

    The shared_secret MUST be set (non-empty); a sidecar started with an empty secret refuses
    every /v1/* request. That's intentional — we'd rather fail closed than accept "" == "".

    Returns a ``JSONResponse`` directly on failure rather than raising
    ``HTTPException``; ``BaseHTTPMiddleware`` does not reliably propagate
    ``HTTPException`` through Starlette's exception middleware across FastAPI
    versions, and silently leaks 500s in tests. Returning the response keeps
    the contract identical regardless of the framework version.
    """
    if request.url.path in UNAUTHENTICATED_PATHS:
        return await call_next(request)

    settings = get_settings()
    expected = (settings.shared_secret or "").strip()
    provided = (request.headers.get(HEADER_NAME) or "").strip()

    if not expected:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "sidecar shared_secret is not configured"},
        )
    if not provided or not hmac.compare_digest(expected, provided):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "missing or invalid sidecar token"},
        )

    return await call_next(request)
