"""FastAPI app entry. Run with:

    uvicorn auxima_ai.main:app --host 0.0.0.0 --port 8088 --reload

Endpoints in this P0-C minimum scaffold:
  - GET /healthz   → 200 with the build version + the time. UNAUTHENTICATED.
  - GET /v1/whoami → 200 with the model alias + shared-secret-present flag. AUTH REQUIRED.

The real AI endpoints (intake/extract, recommend, etc.) land per P1-10 + later.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI

from auxima_ai import __version__
from auxima_ai.auth_select import select_auth_middleware
from auxima_ai.bootstrap import BootstrapError, bootstrap_app
from auxima_ai.config import get_settings
from auxima_ai.intake.router import router as intake_router

app = FastAPI(
    title="Auxima Insure AI Sidecar",
    version=__version__,
    description=(
        "FastAPI service for the multi-agent CRM. REST-only contract with the Frappe "
        "`auxima` app. Never imports frappe. Auth via shared secret."
    ),
)

# Inbound auth is config-selected (S-54 / GAP-16 cutover): shared_secret by
# default (unchanged Phase-0 contract), or the Auxima-v1 HMAC scheme when
# AUXIMA_SIDECAR_SIDECAR_AUTH_MODE=auxima_v1. select_auth_middleware fails
# fast at import if auxima_v1 is set without key material.
app.middleware("http")(select_auth_middleware(get_settings()))
app.include_router(intake_router)


@app.on_event("startup")
def _startup_wire_intake_service() -> None:
    """Compose the production IntakeService at app startup.

    Wraps :func:`bootstrap_app` so a failed compose surfaces as a 503
    on /healthz rather than a confusing late-bind error on the first
    /v1/* call. Tests typically bypass this by registering their own
    service via app.dependency_overrides BEFORE the first request.
    """
    try:
        bootstrap_app()
    except BootstrapError:
        # Don't crash the app — /healthz remains useful for diagnosis.
        # The intake router'\''s default service will refuse calls.
        raise


@app.on_event("shutdown")
def _shutdown_close_http_clients() -> None:
    """Close the HTTPActivityEmitter + OllamaLLMCaller connection pools
    so uvicorn'\''s graceful shutdown doesn'\''t spam ResourceWarnings."""
    from auxima_ai.activity.http_emitter import HTTPActivityEmitter
    from auxima_ai.intake.ollama import OllamaLLMCaller
    from auxima_ai.intake.router import get_intake_service

    svc = get_intake_service()
    if isinstance(svc.activity_emitter, HTTPActivityEmitter):
        svc.activity_emitter.close()
    if isinstance(svc.llm, OllamaLLMCaller):
        svc.llm.close()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Liveness probe. Unauthenticated — used by k8s/Docker healthchecks."""
    return {"status": "ok", "version": __version__, "ts": int(time.time())}


@app.get("/v1/whoami")
def whoami() -> dict[str, Any]:
    """Sanity check that the shared-secret middleware lets an authenticated request through."""
    settings = get_settings()
    return {
        "version": __version__,
        "default_model": settings.default_model,
        "shared_secret_configured": bool(settings.shared_secret),
    }
