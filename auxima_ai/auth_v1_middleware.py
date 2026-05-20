"""FastAPI middleware composing the Auxima-v1 auth core + nonce replay store.

This is the integration layer for S-54 / GAP-16: it wires together the two
pure pieces —
  - :mod:`auxima_ai.auth_v1`     (stateless HMAC + skew verify)
  - :mod:`auxima_ai.auth_nonce`  (stateful replay protection)
— into a single ``http`` middleware and maps every outcome to the S-54 §3.5
HTTP contract.

**Not yet wired into the live app.** :mod:`auxima_ai.main` still uses the
Phase-0 ``shared_secret_middleware``. Cutting the live app over to Auxima-v1
is a coordinated migration (every test that presents ``X-Auxima-Sidecar-
Token`` must move to the signed ``Authorization`` header, and the Frappe-side
signer must ship first). This module + its tests exist so that migration has
a verified, reviewable component to switch to. The factory takes the keyring
+ nonce store as parameters precisely so it can be exercised in isolation.

HTTP contract (S-54 §3.5 + R9):
  - any auth failure (bad scheme/format, unknown key, stale/future ts, bad
    hmac, invalid key) -> 401, body ``{"detail": "...", "reason": "..."}``,
    NO downstream call.
  - nonce replay -> 401 reason=replay.
  - nonce store unreachable -> 503 + ``Retry-After: 5`` (fail closed; we
    cannot verify uniqueness, so we refuse rather than skip the check).
  - success -> the request proceeds; the validated ``key_id`` is stashed on
    ``request.state.auth_key_id`` for the AI Run Log audit (S-54 R7) and
    logged at INFO (key_id ONLY — never the hmac/nonce/secret/timestamp, R10).

Path canonicalisation: the signed path is the URL path PLUS the query string
when present (``/v1/x?since=...``), matching S-54 §3.2. The Frappe-side
signer MUST construct the path identically or every request 401s.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Iterable

from fastapi import Request, status
from fastapi.responses import JSONResponse

from auxima_ai.auth_nonce import (
    DEFAULT_NONCE_TTL_SECONDS,
    NonceReplay,
    NonceStore,
    NonceStoreError,
    NonceStoreUnavailable,
)
from auxima_ai.auth_v1 import (
    DEFAULT_SKEW_SECONDS,
    AuthError,
    Keyring,
    verify_request,
)

logger = logging.getLogger(__name__)

DEFAULT_UNAUTHENTICATED_PATHS: frozenset[str] = frozenset(
    {"/healthz", "/openapi.json", "/docs", "/redoc"}
)


def _request_path_with_query(request: Request) -> str:
    """Reconstruct the signed path: URL path + ``?query`` when present.

    Matches S-54 §3.2 ("the raw URL path including query string ... NOT host,
    NOT scheme"). The Frappe-side signer constructs the identical string.
    """
    path = request.url.path
    query = request.url.query
    return f"{path}?{query}" if query else path


def make_auth_v1_middleware(
    keyring: Keyring,
    nonce_store: NonceStore,
    *,
    skew_seconds: int = DEFAULT_SKEW_SECONDS,
    nonce_ttl_seconds: int = DEFAULT_NONCE_TTL_SECONDS,
    clock=None,
    unauthenticated_paths: Iterable[str] = DEFAULT_UNAUTHENTICATED_PATHS,
) -> Callable[[Request, Callable[[Request], Awaitable]], Awaitable]:
    """Build an ``http`` middleware enforcing the Auxima-v1 scheme.

    Parameters
    ----------
    keyring
        The dual-key :class:`~auxima_ai.auth_v1.Keyring`. An empty keyring
        rejects everything (fail closed).
    nonce_store
        The replay store. Its ``claim`` raising
        :class:`~auxima_ai.auth_nonce.NonceStoreUnavailable` triggers the 503.
    skew_seconds, nonce_ttl_seconds
        S-54 R4 / R5 windows. Defaults 300 / 600.
    clock
        Optional injectable clock (``() -> float``) passed through to
        :func:`~auxima_ai.auth_v1.verify_request` for deterministic tests.
    unauthenticated_paths
        Paths exempt from auth (probes + API docs).
    """
    exempt = frozenset(unauthenticated_paths)
    verify_kwargs = {"skew_seconds": skew_seconds}
    if clock is not None:
        verify_kwargs["clock"] = clock

    async def middleware(
        request: Request, call_next: Callable[[Request], Awaitable]
    ):
        if request.url.path in exempt:
            return await call_next(request)

        # Read the body once. Starlette's BaseHTTPMiddleware caches it so the
        # downstream handler re-reads the SAME bytes the HMAC covers. A
        # transport-level read error (client reset mid-stream) must fail
        # closed with 400, not escape as a 500 (iter-279 review H2).
        try:
            body = await request.body()
        except Exception:
            logger.warning("auth_v1 failed to read request body; rejecting")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "unreadable request body", "reason": "body_read_error"},
            )
        header = request.headers.get("Authorization")
        path = _request_path_with_query(request)

        # Step 1: stateless verify (parse + key + skew + HMAC). Fail closed.
        try:
            token = verify_request(
                header,
                request.method,
                path,
                body,
                keyring,
                **verify_kwargs,
            )
        except AuthError as e:
            # Log auth rejections at WARNING so an attack flood (bad-hmac /
            # unknown-key storm) crosses the same SIEM threshold as a replay
            # (iter-279 review M6). Reason only — never the hmac / nonce /
            # secret / timestamp (S-54 R10).
            logger.warning("auth_v1 reject reason=%s", e.reason)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "unauthorized", "reason": e.reason},
            )

        # Step 2: replay protection (stateful). Fail closed on ANY store
        # error, not only NonceStoreUnavailable: a verified token whose nonce
        # somehow violates the store's input rules, or any future Redis-impl
        # internal error, must 503 (fail closed) rather than escape as a 500
        # (iter-279 review finding-1). With the iter-279 parse-side field
        # validation the input domains are aligned, so InvalidNonceError
        # cannot fire from a verified token — this is defense in depth.
        try:
            claim = nonce_store.claim(token.key_id, token.nonce, nonce_ttl_seconds)
        except (NonceStoreUnavailable, NonceStoreError):
            logger.warning("auth_v1 nonce store error; failing closed (503)")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "nonce store unavailable", "reason": "redis_unreachable"},
                headers={"Retry-After": "5"},
            )

        if isinstance(claim, NonceReplay):
            # Replay = active attack or buggy retry (S-54 §3.5: PAGE).
            logger.warning("auth_v1 reject reason=replay key_id=%s", token.key_id)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "unauthorized", "reason": "replay"},
            )

        # Accepted. Stash key_id for the AI Run Log audit (S-54 R7); log
        # key_id only (R10).
        request.state.auth_key_id = token.key_id
        logger.info("auth_v1 accept key_id=%s", token.key_id)
        return await call_next(request)

    return middleware


__all__ = ("DEFAULT_UNAUTHENTICATED_PATHS", "make_auth_v1_middleware")
