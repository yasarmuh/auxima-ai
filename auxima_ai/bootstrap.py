"""App-startup composition — wires the sidecar from env into one IntakeService.

The intake router uses :func:`auxima_ai.intake.router.get_intake_service`
to fetch the per-process service. In production, :func:`bootstrap_app`
is called once at FastAPI startup; it reads :class:`Settings`, builds
the real :class:`OllamaLLMCaller`, the :class:`PolicyEnforcer` (with
tenants from ``tenants.yaml`` if the path is set), and installs the
composed :class:`IntakeService` via
:func:`auxima_ai.intake.router.set_intake_service`.

The function is idempotent — calling it twice replaces the previous
service — and returns the service it just installed so callers can
hold a handle for graceful shutdown.

Failure policy:
  - Missing tenants file (when ``tenants_path`` is set) raises so the
    sidecar refuses to start. A typo in the env var that silently
    fell back to "no tenants" was the bug that produced a
    "tenant-not-found" outage in prior incidents.
  - Empty ``tenants_path`` is treated as "no tenants yet" — the
    sidecar still starts; every ``/v1/*`` call refuses with
    UnknownTenantError until policies are added by hot-reload.
"""
from __future__ import annotations

import logging
from pathlib import Path

from auxima_ai.activity.http_emitter import HTTPActivityEmitter
from auxima_ai.assist.openrouter import OpenRouterError, OpenRouterLLMCaller
from auxima_ai.assist.router import set_assist_service
from auxima_ai.assist.service import AssistService, ProviderStep
from auxima_ai.config import Settings, get_settings
from auxima_ai.intake.ollama import OllamaLLMCaller
from auxima_ai.intake.router import set_intake_service
from auxima_ai.intake.service import (
    ActivityEmitter,
    IntakeService,
    NullActivityEmitter,
)
from auxima_ai.policy.enforcer import PolicyEnforcer
from auxima_ai.policy.loader import load_and_apply

logger = logging.getLogger(__name__)


class BootstrapError(RuntimeError):
    """Raised when startup composition fails — sidecar must refuse to serve."""


def build_intake_service(settings: Settings) -> IntakeService:
    """Build an :class:`IntakeService` wired with the real Ollama + tenants.

    Pure (no global mutation) so tests can drive it deterministically.
    """
    enforcer = PolicyEnforcer()

    if settings.tenants_path:
        path = Path(settings.tenants_path)
        try:
            count = load_and_apply(enforcer, path)
        except Exception as e:
            raise BootstrapError(
                f"failed to load tenants from {path}: {e}"
            ) from e
        logger.info("bootstrap: loaded %d tenant policies from %s", count, path)
    else:
        logger.warning(
            "bootstrap: tenants_path not set; /v1/* will refuse every call "
            "with UnknownTenantError until policies are registered",
        )

    ollama = OllamaLLMCaller(base_url=settings.ollama_base_url)
    activity_emitter: ActivityEmitter = _build_activity_emitter(settings)
    return IntakeService(
        enforcer=enforcer,
        llm=ollama,
        activity_emitter=activity_emitter,
    )


def _build_activity_emitter(settings: Settings) -> ActivityEmitter:
    """Construct an HTTPActivityEmitter when the callback token is set;
    otherwise fall back to the NullActivityEmitter so the CRM §4 invariant
    is at least logged (the structured log event captures the same facts)."""
    if settings.activity_emission_enabled:
        logger.info(
            "bootstrap: activity emission enabled (POST to %s)",
            settings.frappe_base_url,
        )
        return HTTPActivityEmitter(
            base_url=settings.frappe_base_url,
            token=settings.frappe_callback_token,
        )
    logger.warning(
        "bootstrap: activity emission DISABLED — frappe_callback_token unset; "
        "the structured log event still captures intake.extract.completed, "
        "but no Auxima Activity row will reach Frappe",
    )
    return NullActivityEmitter()


def build_assist_service(
    settings: Settings, enforcer: PolicyEnforcer | None = None
) -> AssistService:
    """Build the assist service with an **Ollama-first**, policy-gated chain.

    Chain order follows CLAUDE §2 (self-hosted is the default): local Ollama
    FIRST, then OpenRouter (free cloud) as an opt-in fallback included ONLY if
    an ``OPENROUTER_API_KEY`` is present. The :class:`PolicyEnforcer` gates each
    step on the tenant's tier, so an ``ollama_only`` tenant never reaches the
    cloud step even when it is wired. If no step is allowed/available every
    draft degrades cleanly (never a 500).

    ``enforcer`` is shared with the intake service so tenant policies registered
    once apply to both paths; a fresh one is created if not supplied.

    Pure (no global mutation) so tests drive it deterministically.
    """
    enforcer = enforcer or PolicyEnforcer()
    steps: list[ProviderStep] = [
        ProviderStep(
            caller=OllamaLLMCaller(
                base_url=settings.ollama_base_url,
                timeout_seconds=settings.assist_ollama_timeout_s,
            ),
            model_id=settings.assist_ollama_model,
            provider_class="self-hosted",
        )
    ]
    try:
        steps.append(
            ProviderStep(
                caller=OpenRouterLLMCaller(),
                model_id=settings.assist_openrouter_model,
                provider_class="free-cloud",
            )
        )
    except OpenRouterError as e:
        # No API key (or bad config) — skip the opt-in cloud step; Ollama-only.
        logger.warning("bootstrap: OpenRouter assist step skipped (%s); Ollama-only", e)
    logger.info("bootstrap: assist chain wired Ollama-first with %d provider step(s)", len(steps))
    return AssistService(enforcer=enforcer, steps=steps)


def bootstrap_app(settings: Settings | None = None) -> IntakeService:
    """Build + install the app-wide IntakeService + AssistService. Idempotent.

    Returns the IntakeService that was installed so callers can keep a handle
    (e.g. for graceful shutdown of the OllamaLLMCaller's HTTP pool).
    """
    s = settings or get_settings()
    service = build_intake_service(s)
    set_intake_service(service)
    # Share the intake enforcer so a tenant's tier/policy applies to BOTH paths.
    set_assist_service(build_assist_service(s, enforcer=service.enforcer))
    return service


__all__ = (
    "BootstrapError",
    "bootstrap_app",
    "build_assist_service",
    "build_intake_service",
)
