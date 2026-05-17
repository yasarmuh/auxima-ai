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
from auxima_ai.auth import shared_secret_middleware
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

app.middleware("http")(shared_secret_middleware)
app.include_router(intake_router)


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
