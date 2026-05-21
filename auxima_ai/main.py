"""FastAPI app entry. Run with:

    uvicorn auxima_ai.main:app --host 0.0.0.0 --port 8088 --reload

Endpoints in this P0-C minimum scaffold:
  - GET /healthz       → 200 with the build version + time. UNAUTHENTICATED.
  - GET /v1/whoami     → 200 model alias + shared-secret flag. AUTH REQUIRED.
  - GET /v1/auth/whoami→ active Auxima-v1 key_id (S-54 R12). AUTH REQUIRED;
                         meaningful only in auxima_v1 mode (see below).

The app is built by :func:`create_app` so the auth mode + routers can be
wired from explicit settings (tests build per-mode apps without touching
global env). The module-level ``app = create_app()`` is the uvicorn entry
point and is unchanged.

The real AI endpoints (intake/extract, recommend, etc.) land per P1-10 + later.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from auxima_ai import __version__
from auxima_ai.assist.router import router as assist_router
from auxima_ai.auth_select import select_auth_middleware
from auxima_ai.auth_whoami import router as whoami_router
from auxima_ai.bootstrap import BootstrapError, bootstrap_app
from auxima_ai.config import Settings, get_settings
from auxima_ai.intake.router import router as intake_router


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup/shutdown lifecycle (replaces the deprecated on_event handlers).

    Startup: compose the production IntakeService; a failed compose raises so
    it surfaces at startup rather than as a confusing late-bind error on the
    first /v1/* call. Shutdown: close the HTTPActivityEmitter + OllamaLLMCaller
    connection pools so uvicorn's graceful shutdown doesn't spam
    ResourceWarnings. As before, this only runs when the app is entered as a
    context (e.g. `with TestClient(app)`), so plain-construction tests skip it.
    """
    try:
        bootstrap_app()
    except BootstrapError:
        raise
    yield
    from auxima_ai.activity.http_emitter import HTTPActivityEmitter
    from auxima_ai.intake.ollama import OllamaLLMCaller
    from auxima_ai.intake.router import get_intake_service

    svc = get_intake_service()
    if isinstance(svc.activity_emitter, HTTPActivityEmitter):
        svc.activity_emitter.close()
    if isinstance(svc.llm, OllamaLLMCaller):
        svc.llm.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the sidecar app wired for the given (or current) settings.

    Inbound auth is config-selected (S-54 / GAP-16 cutover): ``shared_secret``
    by default (unchanged Phase-0 contract), or the Auxima-v1 HMAC scheme when
    ``sidecar_auth_mode=auxima_v1``. :func:`select_auth_middleware` fails fast
    if auxima_v1 is selected without key material.

    ``/v1/auth/whoami`` is reachable in both modes, but only returns a key_id
    in auxima_v1 mode — in shared_secret mode the v1 middleware isn't active,
    so ``request.state.auth_key_id`` is unset and the route fails closed (401
    reason=unauthenticated). That is the intended behavior: whoami is a v1
    rotation probe.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="Auxima Insure AI Sidecar",
        version=__version__,
        description=(
            "FastAPI service for the multi-agent CRM. REST-only contract with the Frappe "
            "`auxima` app. Never imports frappe."
        ),
        lifespan=_lifespan,
    )

    app.middleware("http")(select_auth_middleware(settings))
    app.include_router(intake_router)
    app.include_router(assist_router)
    app.include_router(whoami_router)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Liveness probe. Unauthenticated — used by k8s/Docker healthchecks."""
        return {"status": "ok", "version": __version__, "ts": int(time.time())}

    @app.get("/v1/whoami")
    def whoami() -> dict[str, Any]:
        """Sanity check that the auth middleware lets an authenticated request through."""
        current = get_settings()
        return {
            "version": __version__,
            "default_model": current.default_model,
            "shared_secret_configured": bool(current.shared_secret),
        }

    return app


# The uvicorn entry point: `uvicorn auxima_ai.main:app`.
app = create_app()
