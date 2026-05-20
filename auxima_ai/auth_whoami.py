"""GET /v1/auth/whoami — the active-key_id probe (S-54 R12 / AC-8).

A tiny authenticated endpoint used by the rotation runbook (S-54 §3.4) to
confirm *which* ``key_id`` the sidecar is currently accepting — e.g. after
flipping the Frappe-side signer from primary to secondary, an operator hits
this to verify the cutover took.

It reads the validated ``key_id`` that :mod:`auxima_ai.auth_v1_middleware`
stashes on ``request.state.auth_key_id`` after a successful verify. When the
v1 middleware guards the route, an unauthenticated request never reaches the
handler (the middleware 401s first). The handler keeps its own fail-closed
guard so that, if it is ever mounted WITHOUT the v1 middleware, a request
with no validated key 401s rather than 500-ing on the missing attribute.

Returns ``key_id`` ONLY — never the nonce, timestamp, or HMAC (S-54 R10).
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from auxima_ai.auth_v1 import SCHEME

router = APIRouter()


@router.get("/v1/auth/whoami")
def whoami(request: Request) -> JSONResponse:
    """Return the active ``key_id`` to an authenticated caller; 401 otherwise.

    The middleware sets ``request.state.auth_key_id`` only on a verified
    request, so its presence IS the authentication signal here.
    """
    key_id = getattr(request.state, "auth_key_id", None)
    if not key_id:
        return JSONResponse(
            status_code=401,
            content={"detail": "unauthorized", "reason": "unauthenticated"},
        )
    return JSONResponse(status_code=200, content={"key_id": key_id, "scheme": SCHEME})


__all__ = ("router",)
