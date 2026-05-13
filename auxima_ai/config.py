"""Sidecar configuration — loaded from environment variables (12-factor)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config is env-driven. Secrets MUST come from env, never from code."""

    model_config = SettingsConfigDict(
        env_prefix="AUXIMA_SIDECAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The shared secret that the Frappe-side caller must present in the X-Auxima-Sidecar-Token
    # header. Required for all /v1/* endpoints; /healthz is unauthenticated by design.
    # Rotation: change the env var on both sides simultaneously, restart.
    shared_secret: str = ""

    # The base URL of the Frappe `auxima` app (used for outbound REST calls back to write
    # Auxima Activity rows etc). Not used yet in this minimum scaffold.
    frappe_base_url: str = "http://demo.localhost:8000"

    # The LiteLLM router default model alias (e.g. "ollama/qwen2.5:32b").
    default_model: str = "ollama/qwen2.5:32b"

    # Log level. DEBUG only in dev; never in prod.
    log_level: str = "INFO"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached settings accessor (lazy — avoids reading env at import time)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
