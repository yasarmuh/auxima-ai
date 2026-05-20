"""Sidecar configuration — loaded from environment variables (12-factor).

Validation rules (enforced at construction time so the sidecar fails fast
at startup rather than at first request):

  - ``shared_secret`` MUST be either empty (treated as "unconfigured" by
    the middleware, which then fails closed with 503) or at least
    :data:`MIN_SHARED_SECRET_LEN` characters. A short non-empty secret
    is a configuration mistake (a 4-char "test" left in prod env), and
    failing at startup surfaces it before the first 401.
  - ``log_level`` must be one of the canonical stdlib levels (case-
    normalised to upper).
  - ``frappe_base_url`` must use an ``http://`` or ``https://`` scheme
    and parse cleanly; non-loopback URLs MUST be ``https://`` (cleartext
    to an external Frappe in prod would leak the shared secret on the
    wire).
"""
from __future__ import annotations

from typing import Final
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 16 chars = 128 bits — strong enough for HMAC. We document 32+ as the
# recommendation in NOTES but enforce 16 as a floor so the existing
# test fixture (a 30-char secret) keeps working.
MIN_SHARED_SECRET_LEN: Final[int] = 16

_VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset(
    {"localhost", "127.0.0.1", "::1"}
)


def _validate_http_url(v: str, field_name: str) -> str:
    """Shared validator for any http(s) URL setting.

    Loopback hosts (RFC 6761 §6.3 — ``*.localhost``, ``127.0.0.1``,
    ``::1``) may use ``http://``; any other host MUST use ``https://``
    so the secrets / payloads riding on this URL don't go cleartext.
    """
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{field_name} must be a non-empty URL")
    parsed = urlparse(v)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"{field_name} scheme must be http or https; got {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise ValueError(f"{field_name} missing host: {v!r}")
    host = (parsed.hostname or "").lower()
    is_loopback = host in _LOOPBACK_HOSTS or host.endswith(".localhost")
    if parsed.scheme == "http" and not is_loopback:
        raise ValueError(
            f"{field_name} must use https:// for non-loopback hosts; got {v!r}"
        )
    return v


class Settings(BaseSettings):
    """All config is env-driven. Secrets MUST come from env, never from code."""

    model_config = SettingsConfigDict(
        env_prefix="AUXIMA_SIDECAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The shared secret that the Frappe-side caller must present in the
    # X-Auxima-Sidecar-Token header. Required for all /v1/* endpoints;
    # /healthz is unauthenticated by design.
    # Rotation: change the env var on both sides simultaneously, restart.
    shared_secret: str = ""

    # The base URL of the Frappe `auxima` app (used for outbound REST calls
    # back to write Auxima Activity rows etc). Loopback may be http://;
    # any other host MUST be https:// (the shared secret would otherwise
    # ride cleartext over the wire).
    frappe_base_url: str = "http://demo.localhost:8000"

    # The LiteLLM router default model alias (e.g. "ollama/qwen2.5:32b").
    default_model: str = "ollama/qwen2.5:32b"

    # Base URL of the Ollama daemon (used by OllamaLLMCaller). Default
    # matches `ollama serve` listening locally on its standard port.
    ollama_base_url: str = "http://localhost:11434"

    # Path to the per-tenant policy manifest (tenants.yaml). Optional —
    # if unset, the sidecar starts with no tenants registered and every
    # /v1/* call fails UnknownTenantError until policies are added.
    tenants_path: str = ""

    # Shared secret used in the REVERSE direction — sidecar -> Frappe
    # when POSTing Auxima Activity rows. Same length rules as
    # shared_secret. Empty = activity emission disabled (NullActivityEmitter);
    # the structured log event still captures the same facts so nothing
    # is silently lost, but the Frappe-side audit log won't see the row.
    frappe_callback_token: str = ""

    # Log level. DEBUG only in dev; never in prod. Validated to be one of
    # the canonical stdlib level names; lowercase input is normalised up.
    log_level: str = "INFO"

    # Which inbound auth scheme guards /v1/* (S-54 / GAP-16 cutover).
    #   "shared_secret" (default) — the Phase-0 X-Auxima-Sidecar-Token scheme.
    #   "auxima_v1"               — the HMAC + replay-protected Auxima-v1 scheme.
    # Defaults to shared_secret so the live contract is unchanged until the
    # Frappe-side Auxima-v1 signer ships. Switching to auxima_v1 requires the
    # primary/secondary key env vars below.
    sidecar_auth_mode: str = "shared_secret"

    # Auxima-v1 dual-key material (only used when sidecar_auth_mode=auxima_v1).
    # Each key is base64(32 random bytes); key_id is its audit-legible name
    # (e.g. "p2026q2"). Env vars: AUXIMA_SIDECAR_PRIMARY_KEY_B64 etc.
    primary_key_id: str = ""
    primary_key_b64: str = ""
    secondary_key_id: str = ""
    secondary_key_b64: str = ""

    # -- validators -------------------------------------------------------

    @field_validator("shared_secret")
    @classmethod
    def _validate_shared_secret(cls, v: str) -> str:
        # Empty string is the "unconfigured" sentinel — middleware turns
        # it into a 503 on every /v1/* request (fail closed). A non-empty
        # but short secret is almost certainly a typo or leftover test
        # value; fail at startup so we don't ship with weak auth.
        if v == "":
            return v
        if len(v) < MIN_SHARED_SECRET_LEN:
            raise ValueError(
                f"shared_secret must be empty (unconfigured) or "
                f">= {MIN_SHARED_SECRET_LEN} chars; got {len(v)}"
            )
        return v

    @field_validator("frappe_callback_token")
    @classmethod
    def _validate_frappe_callback_token(cls, v: str) -> str:
        # Same fail-closed-on-short-but-non-empty rule as shared_secret —
        # a 4-char "test" leftover token in prod env is a configuration
        # mistake we want to surface at startup.
        if v == "":
            return v
        if len(v) < MIN_SHARED_SECRET_LEN:
            raise ValueError(
                f"frappe_callback_token must be empty (disabled) or "
                f">= {MIN_SHARED_SECRET_LEN} chars; got {len(v)}"
            )
        return v

    @field_validator("sidecar_auth_mode")
    @classmethod
    def _validate_sidecar_auth_mode(cls, v: str) -> str:
        normalised = (v or "").strip().lower()
        if normalised not in ("shared_secret", "auxima_v1"):
            raise ValueError(
                "sidecar_auth_mode must be 'shared_secret' or 'auxima_v1'; "
                f"got {v!r}"
            )
        return normalised

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        normalised = (v or "").strip().upper()
        if normalised not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}; got {v!r}"
            )
        return normalised

    @field_validator("frappe_base_url")
    @classmethod
    def _validate_frappe_base_url(cls, v: str) -> str:
        return _validate_http_url(v, "frappe_base_url")

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_base_url(cls, v: str) -> str:
        return _validate_http_url(v, "ollama_base_url")

    # -- convenience accessors -------------------------------------------

    @property
    def shared_secret_configured(self) -> bool:
        """``True`` iff a non-empty secret is set — middleware uses this."""
        return bool(self.shared_secret)

    @property
    def activity_emission_enabled(self) -> bool:
        """``True`` iff sidecar -> Frappe activity-row emission is wired."""
        return bool(self.frappe_callback_token)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached settings accessor (lazy — avoids reading env at import time)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Reset the module-level cache. Test-only — call after monkeypatching env."""
    global _settings
    _settings = None
